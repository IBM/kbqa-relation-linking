from elasticsearch import Elasticsearch
from elasticsearch_dsl import Q, Search
from distant_supervision.ds_utils import CorpusGenUtils


class ElasticClient(object):

    # Example document from the elasticsearch index
    # {
    #    "_index":"enwiki",
    #    "_type":"sentence",
    #    "_id":"AXCbniQUQZzz3GiznsR6",
    #    "_score":21.83448,
    #    "_source":{
    #       "pageid":22177894,
    #       "position":3,
    #       "text":"Madrid, Spain: Assembly of Madrid.",
    #       "pagetitle":"List of Presidents of the Assembly of Madrid",
    #       "after":"Retrieved 30 January 2019.",
    #       "before":"\"Relación de Presidentes\" (in Spanish)."
    #    }
    # }

    def __init__(self,
                 host: str = "localhost",
                 port: int = 9200,
                 index_name: str = "enwiki",
                 field_names=["title", "text"]):

        self.client = Elasticsearch([host], port=port, timeout=45)
        self.fields = field_names
        self.index_name = index_name

    def query_sentences(self, page_title: str, term_a: str, term_b: str, size: int = 20):
        term_a, term_b = term_a.lower(), term_b.lower()
        s = Search(using=self.client, index=self.index_name)[0:size]
        s.query = Q('bool', must=[Q('match', pageid=page_title), Q('match_phrase', text=term_a),
                                  Q('match_phrase', text=term_b)])
        s.execute()

        results = list()
        for hit in s:
            sent_text = hit.text.lower()
            if term_a in sent_text and term_b in sent_text:
                results.append((hit.position, hit.text, hit.pagetitle, '{}_{}'.format(hit.pageid, hit.position)))

        filtered = list()
        for result in results:
            if result[2].lower() == page_title.lower():
                filtered.append(result)
        if filtered:
            results = filtered

        return sorted(results, key=lambda result: result[0])

    def query_flexible(self, page_id: int, search_terms: list,  size: int = 500):
        s = Search(using=self.client, index=self.index_name)[0:size]
        should_list = list()
        for term in search_terms:
            should_list.append(Q('match_phrase', text=term))
        s.query = Q('bool',
                    must=[Q('match', pageid=page_id)],
                    should=should_list,
                    minimum_should_match=1)

        results = list()
        try:
            s.execute()
        except Exception as ex:
            print("Elasticsearch error: {}".format(str(ex)))
            return results

        for hit in s:
            suface_form = None
            text = hit.text.lower()
            # we are trying to find the surface form of the most relevant term in the ranked list of search terms
            for term in search_terms:
                if term.lower() in text:
                    suface_form = term
                    break
            if suface_form:
                results.append((hit.position, suface_form, hit.text, hit.pagetitle, '{}_{}'.format(hit.pageid, hit.position)))
        return sorted(results, key=lambda result: result[0])

    @classmethod
    def get_best_matching_setence(cls, subj_sentences, obj_sentences, sub_terms, obj_terms, count: int = 3):

        final_sentences = list()
        subj_indices = {subj_sent[0]:subj_sent for subj_sent in subj_sentences}
        # obj_indices = {obj_sent[0]:obj_sent for obj_sent in obj_sentences}

        subj_term_sent_id = dict()
        obj_term_sent_id = dict()
        for sent in subj_sentences:
            term = sent[1]
            sent_ids = subj_term_sent_id.get(term, set())
            sent_ids.add(sent[0])
            subj_term_sent_id[term] = sent_ids

        for sent in obj_sentences:
            term = sent[1]
            sent_ids = obj_term_sent_id.get(term, set())
            sent_ids.add(sent[0])
            obj_term_sent_id[term] = sent_ids

        for obj_term in obj_terms:
            if obj_term in obj_term_sent_id:
                obj_sent_ids = sorted(obj_term_sent_id[obj_term])
                subj_sent_ids = list()
                for sent_id in obj_sent_ids:
                    if sent_id in subj_indices:
                        subj_sent = subj_indices[sent_id]
                        subj_term = subj_sent[1]
                        subj_term_index = sub_terms.index(subj_term)
                        subj_sent_ids.append([subj_sent, subj_term_index])
                if subj_sent_ids:
                    subj_sent_ids = sorted(subj_sent_ids, key=lambda x: x[1])
                    selected_sent_id = subj_sent_ids[0][0][0]
                    selected_sentence = subj_indices[selected_sent_id]

                    sub_term_list = list()
                    for sub_term in sub_terms:
                        if sub_term in selected_sentence[2].lower():
                            sub_term_list.append(sub_term)

                    obj_term_list = list()
                    for obj_term in obj_terms:
                        if obj_term.lower() in selected_sentence[2].lower():
                            obj_term_list.append(obj_term)

                    modified_sentence = [selected_sentence[0], sub_term_list, obj_term_list, selected_sentence[2],
                                         selected_sentence[3], selected_sentence[4]]
                    final_sentences.append(modified_sentence)
                    if len(final_sentences) > count:
                        return final_sentences
        return final_sentences


if __name__ == '__main__':
    es_client = ElasticClient()
    labels = CorpusGenUtils.get_link_text(sparql_endpoint=CorpusGenUtils.dbpedia_201610,
                                          dbpedia_uri='http://dbpedia.org/resource/Barack_Obama')
    sorted_labels = CorpusGenUtils.sort_by_similarity("Barack Obama", labels)

    print(sorted_labels)
    sub_terms = [term[0] for term in sorted_labels]
    subj_sentences = es_client.query_flexible(534366, sub_terms)

    date_variants = CorpusGenUtils.get_all_date_variants('1961-08-04')
    print(date_variants)
    obj_sentences = es_client.query_flexible(534366, date_variants)

    final_sentences = ElasticClient.get_best_matching_setence(subj_sentences, obj_sentences, sub_terms, date_variants)
    for sent in final_sentences:
        print(sent)