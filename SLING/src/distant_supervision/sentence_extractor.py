import argparse
import json
import neuralcoref
import wikipedia
import spacy
import sys
import time
import random
import re
from os.path import exists
from multiprocessing import Lock
from SPARQLWrapper import SPARQLWrapper, JSON
from queue import Queue
from threading import Thread
from distant_supervision.es_client import ElasticClient
from distant_supervision.ds_utils import DistantSupervisionUtils

# setup the libraries
# Please replace this with a private DBpedia endpoint
DBPEDIA_SPARQL = 'https://dbpedia.org/sparql'
DBP_RES_PREFIX = 'http://dbpedia.org/resource/'

number_regex = '\d+\.*\d*'

nlp_coref = spacy.load('en')
neuralcoref.add_to_pipe(nlp_coref)

nlp_sent = spacy.load('en')
nlp_sent.add_pipe(nlp_sent.create_pipe('sentencizer'))

nlp = spacy.load("en")

wikipedia.set_lang("en")

possessive_pronouns = ['my', 'our', 'your', 'his', 'her', 'its', 'their']

wiki_page_cache = dict()

# queues and locks for parallel processing
relation_queue = Queue()
relation_queue_lock, wiki_page_cache_lock, output_lock = Lock(), Lock(), Lock()
num_of_threads, completed_count = 40, 0

es_client = ElasticClient()

rel_contexts = {rel.strip().split('\t')[0]: rel.strip().split('\t')[1:] for rel in open('../data/dbpedia/expanded_terms.tsv')}
for rel in rel_contexts:
    terms = rel_contexts[rel]
    rel_contexts[rel] = [term.lower().replace('_', ' ') for term in terms]

print('{} rel contexts loaded '.format(len(rel_contexts)))


def get_antecedent(token):
    if type(token) != spacy.tokens.token.Token:
        return
    if not token._.in_coref or token.text.lower() in possessive_pronouns:
        return
    token_start = token.idx
    token_end = token.idx + len(token.text)
    for cluster in token._.coref_clusters:
        if cluster.main.text == token.text:
            return
        for mention in cluster.mentions:
            if token_start == mention.start_char and token_end == mention.end_char:
                return cluster.main.text


def resolve_corefences(sentence):
    doc = nlp_coref(sentence)
    new_string = ''
    for token in doc:
        res = get_antecedent(token)
        if res:
            new_string += res
        else:
            new_string += token.text
        if token.whitespace_:
            new_string += token.whitespace_
    return new_string


def get_relation_triples(relation, limit=1000, thread_id=0):
    answer_records = list()
    sparql = SPARQLWrapper(DBPEDIA_SPARQL)
    sparql.setQuery(" PREFIX dbo: <http://dbpedia.org/ontology/>  " +
                    " PREFIX dbp: <http://dbpedia.org/property/>  " +
                    " PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>  " +
                    " SELECT DISTINCT ?subject ?pageID ?object ?subjectLabel ?objectLabel WHERE { "
                    " SELECT ?subject ?pageID ?object ?subjectLabel ?objectLabel ?subjectDegree { " +
                    "   ?subject <" + relation + "> ?object .       " +
                    "   ?subject dbo:wikiPageID ?pageID . " +
                    "   ?subject rdfs:label ?subjectLabel .     " +
                    "   OPTIONAL { ?object rdfs:label ?objectLabel }       " +
                    "   ?subject dbo:wikiPageLength ?pageLength . } ORDER BY DESC(?pageLength)}" +
                    " LIMIT " + str(limit) + "  ")
    print(sparql.queryString)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        for result in results["results"]["bindings"]:
            subject_label, object_label = None, None
            subject = result['subject']['value']
            subject_label = result['subjectLabel']['value']
            pageID = result['pageID']['value']
            if result['object']['type'] == 'typed-literal' and result['object']['datatype'] == 'http://www.w3.org/2001/XMLSchema#date':
                object_type = 'date'
                object_value = result['object']['value']
            elif result['object']['type'] == 'typed-literal' and result['object']['datatype'] == 'http://www.w3.org/2001/XMLSchema#gYear':
                object_type = 'year'
                object_value = str(result['object']['value']).lstrip("0")
            elif result['object']['type'] == 'typed-literal' and re.match(number_regex, result['object']['value']):
                object_type = 'number'
                object_value = result['object']['value']
            elif result['object']['type'] == 'uri':
                object_type = 'uri'
                object_value = result['object']['value']
            else:
                object_type = 'other'
                object_value = result['object']['value']

            if 'objectLabel' in result:
                object_label = result['objectLabel']['value']
            answer_records.append([subject, pageID, object_value, object_type, subject_label, object_label])
        print("t{}: {} - found {} triples!".format(thread_id, relation, len(answer_records)))
    except Exception as ex:
        print("SPARQL error: {}".format(ex))
    return answer_records


def get_page(page_id):
    with wiki_page_cache_lock:
        if page_id in wiki_page_cache:
            return wiki_page_cache[page_id]

    try:
        page = wikipedia.page(page_id)
    except Exception:
        try:
            page = wikipedia.page(page_id.replace('_', ' '))
        except Exception as ex:
            print(str(ex))
            return list()

    paragraphs = page.content.split('\n')
    sentences = list()
    index = -1
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if paragraph == '' or paragraph.startswith('='):
            continue
        else:
            res_para = resolve_corefences(re.sub('\(.*\)', '', paragraph))
            doc = nlp_sent(res_para)
            for sent in doc.sents:
                sentences.append(sent.text)
                index += 1

    with wiki_page_cache_lock:
        wiki_page_cache[page_id] = sentences
    return sentences


def get_relation_sentences(relation, relation_triples, limit=1000, thread_id=0):
    index = 0
    relation_instances = list()
    for subject_uri, page_id, object_value, object_type, subject_label, object_label in relation_triples:
        try:
            if not subject_uri.startswith(DBP_RES_PREFIX):
                continue
            index += 1
            print('t{}: checking:\n\t{}\n\t{}\n\t{}\n\t{}'.format(thread_id, subject_uri, relation, object_value,
                                                                subject_uri.replace('http://dbpedia.org/resource/',
                                                                                    'https://en.wikipedia.org/wiki/')))

            subj_alt_labels = DistantSupervisionUtils.get_link_text(sparql_endpoint=DBPEDIA_SPARQL,
                                                           dbpedia_uri=subject_uri)
            subj_alt_labels_scores = CorpusGenUtils.sort_by_similarity(subject_label, subj_alt_labels)
            subj_alt_labels = [term[0] for term in subj_alt_labels_scores]

            if object_type == 'uri':
                if not object_label:
                    object_label = object_value.replace(DBP_RES_PREFIX, '').replace('_', ' ')
                obj_alt_labels = DistantSupervisionUtils.get_link_text(sparql_endpoint=DBPEDIA_SPARQL,
                                                               dbpedia_uri=object_value)
                obj_alt_labels_scores = DistantSupervisionUtils.sort_by_similarity(object_label, obj_alt_labels)
                obj_alt_labels = [term[0] for term in obj_alt_labels_scores]
            elif object_type == 'number':
                obj_alt_labels = DistantSupervisionUtils.get_all_number_variants(object_value)
            elif object_type == 'date':
                obj_alt_labels = DistantSupervisionUtils.get_all_date_variants(object_value)
            else:
                obj_alt_labels = [object_value]

            subj_sentences = es_client.query_flexible(page_id, subj_alt_labels)
            obj_sentences = es_client.query_flexible(page_id, obj_alt_labels)

            final_sentences = ElasticClient.get_best_matching_setence(subj_sentences, obj_sentences, subj_alt_labels,
                                                                      obj_alt_labels)
            if final_sentences:
                sentence = final_sentences[0][3]
                print('\t {}'.format(sentence))
                sent_id = final_sentences[0][5]
                spacy_doc = nlp(sentence)
                tokenized_sentence = [token.text.lower() for token in spacy_doc]
                subject_label = final_sentences[0][1]
                object_label = final_sentences[0][2]

                num_verbs = 0
                for token in spacy_doc:
                    if token.pos_ == 'VERB':
                        num_verbs += 1

                if num_verbs == 0 or num_verbs > 4:
                    continue

                if len(tokenized_sentence) > 50:
                    continue

                if not sentence.endswith('.'):
                    continue

                subject_tokens = [token.text.lower() for token in nlp(subject_label[0])]
                object_tokens = [token.text.lower() for token in nlp(object_label[0])]

                try:
                    subj_start, subj_end = tokenized_sentence.index(subject_tokens[0]), \
                                           tokenized_sentence.index(subject_tokens[-1]) + 1
                    obj_start, obj_end = tokenized_sentence.index(object_tokens[0]), \
                                         tokenized_sentence.index(object_tokens[-1]) + 1

                    # check if subject nd object overlap and ignore those cases
                    if obj_start >= subj_start and obj_end <= subj_end:
                        continue
                    elif subj_start >= obj_start and subj_end <= obj_end:
                        continue
                    # check for incorrect cases where it accidentally find random start and end tokens
                    if obj_end < obj_start or subj_end < subj_end:
                        continue

                    relation_instances.append((' '.join(tokenized_sentence[subj_start:subj_end]), subj_start, subj_end,
                                               ' '.join(tokenized_sentence[obj_start:obj_end]), obj_start, obj_end,
                                               subject_label, object_label, object_value, object_type,
                                               tokenized_sentence, sentence, sent_id))
                    print("t{}: {} ({}) - ({}/{})".format(thread_id, relation, len(relation_triples),
                                                        len(relation_instances), index))
                except ValueError:
                    continue

                if len(relation_instances) >= limit:
                    return relation_instances

        except Exception as ex:
            print("Error {}".format(str(ex)))

    return relation_instances


def relation_sent_extractor_worker(thread_id, sparql_limit, sentence_limit):
    global completed_count
    while True:
        try:
            with relation_queue_lock:
                if relation_queue.qsize() == 0:
                    break
                relation = relation_queue.get()
            start_time = time.time()
            print('t{}: starting {}'.format(thread_id, relation))
            relation_triples = get_relation_triples(relation, sparql_limit, thread_id)
            relation_instances = get_relation_sentences(relation, relation_triples, sentence_limit, thread_id)
            instances = list()
            for inst in relation_instances:
                token = inst[6]
                h = dict(name=inst[0], id=inst[0].replace(' ', '_'), pos=[inst[1], inst[2]])
                t = dict(name=inst[3], id=inst[3].replace(' ', '_'), pos=[inst[4], inst[5]])
                instances.append(dict(token=token, h=h, t=t, relation=relation.replace('http://dbpedia.org/ontology/', 'dbo:')))

            with open('{}.txt'.format(relation.replace('http://dbpedia.org/ontology/', 'dbo_').replace('/', '_')), 'w') as relation_file:
                for inst in instances:
                    relation_file.write(json.dumps(inst))
                    relation_file.write('\n')
            print("t{} - COMPLETED {}, took {:.2f} minutes.".format(thread_id, relation, (time.time()-start_time)/60))

        except Exception as ex:
            print("t{}\tError occurred! {}".format(thread_id, str(ex)))

    with relation_queue_lock:
        completed_count += 1


def original_sent_extractor_worker(thread_id, sparql_limit, sentence_limit):
    global completed_count
    while True:
        try:
            with relation_queue_lock:
                if relation_queue.qsize() == 0:
                    break
                relation = relation_queue.get()

            file_name = '{}.txt'.format(relation.replace('http://dbpedia.org/ontology/', 'dbo_')
                                        .replace('http://dbpedia.org/property/', 'dbp_')
                                        .replace('/', '_'))

            output_file = '/Users/nandana.sampath.mihindukulasooriya@ibm.com/Src/relation-linking/data/lc-qald/sent/' \
                          + file_name

            if exists(output_file):
                continue

            if exists("output_path" + file_name):
                continue

            if exists("output_path" + file_name):
                continue

            start_time = time.time()
            print('t{}: starting {}'.format(thread_id, relation))
            relation_triples = get_relation_triples(relation, sparql_limit, thread_id)
            relation_instances = get_relation_sentences(relation, relation_triples, sentence_limit, thread_id)

            output_list = list()
            for relation_instance in relation_instances:
                rel_ins_data = dict()
                rel_ins_data['id'] = relation_instance[12]
                rel_ins_data['text'] = relation_instance[11]
                rel_ins_data['relation'] = relation
                rel_ins_data['subject'] = relation_instance[0]
                rel_ins_data['subject_labels'] = relation_instance[6]
                rel_ins_data['object'] = relation_instance[8]
                rel_ins_data['object_type'] = relation_instance[9]
                rel_ins_data['object_labels'] = relation_instance[7]
                output_list.append(rel_ins_data)

            with open(output_file, 'w', encoding='utf-8') as output_file:
                json.dump(output_list, output_file, indent=2)

            print("t{} - COMPLETED {}, took {:.2f} minutes.".format(thread_id, relation, (time.time() - start_time) / 60))

        except Exception as ex:
            print("t{}\tError occurred! {}".format(thread_id, str(ex)))

    with relation_queue_lock:
        completed_count += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rel_file')
    parser.add_argument('--output_dir')
    args = parser.parse_args()

    print('reading relations from "{}"'.format(args.rel_file))
    # adding all relations in the rel file to a queue to be used by relation sentence extractor workers
    file_list = [rel.strip().split('\t')[0] for rel in open(args.rel_file)]
    random.shuffle(file_list)
    for file in file_list:
        relation_queue.put(file)
    print('\t {} relations found.'.format(relation_queue.qsize()))

    # limits for the number of SPARQL results (triples) and the number of sentences per relation. We can't find
    # sentences for some triples, for that, the limit for the triples are higher than the sentences
    sparql_limit = 80000
    sentence_limit = 1000

    # start all workers and wait for completion
    for i in range(num_of_threads):
        print('starting thread: {}'.format(i))
        worker = Thread(target=original_sent_extractor_worker, args=[i, sparql_limit, sentence_limit])
        worker.setDaemon(True)
        worker.start()
        time.sleep(60)

    while completed_count < num_of_threads:
        time.sleep(10)


if __name__ == "__main__":
    sys.exit(main())
