from collections import Counter
from gensim.parsing.preprocessing import remove_stopwords
import nltk
import numpy as np
import scipy
import re
import pickle

from nltk.tokenize import RegexpTokenizer, WordPunctTokenizer
from rapidfuzz import process, fuzz

from relation_linking_core.rel_linker_modules.rel_linker_module import RelModule


class QuestionSimilarityBasedRelRecommender(RelModule):

    def __init__(self, config=None):

        print("Initializing Question Similarity Based Rel Recommender ....")

        self.prop_map = QuestionSimilarityBasedRelRecommender.readPropertyMap("/Users/nandana.sampath.mihindukulasooriya@ibm.com/data/rel_ranking/similarity/all_relationship_labels_lemma.tsv")
        with open('/Users/nandana.sampath.mihindukulasooriya@ibm.com/Src/relation-linking/data/glove/glove_vocab.pkl', 'rb') as filein:
            embedding_dict = pickle.load(filein)

        self.fuzzywuzzysimExtractor = FuzzyWuzzySimilarityCalc(self.prop_map)
        self.similarityExtractor = SimilarityCalc(self.prop_map, embedding_dict)

        print("\tInitialized ...")

    def get_relation_candidates(self, triple_data, params=None):
        relation_scores = Counter()

        print("\n\t ------------ Checking word_embedding similarities:  ------------ ")

        listOfRelations = params["listOfRelations"]
        subj_text, subj_uri = triple_data['subj_text'], triple_data['subj_uri']
        obj_text, obj_uri = triple_data['obj_text'], triple_data['obj_uri']
        question = triple_data['text']

        word_embd_sim_q = remove_stopwords(str(question).lower()).replace('?', '')

        if subj_uri:
            print(f'\t\tsubj_uri exists, replace {subj_text.lower()} in question')
            word_embd_sim_q = word_embd_sim_q.replace(subj_text.lower(), '')
        if obj_uri:
            print(f'\t\tobj_uri exists, replace {obj_text.lower()} in question')
            word_embd_sim_q = word_embd_sim_q.replace(obj_text.lower(), '')

        reranked_relations = self.similarityExtractor.extract(word_embd_sim_q, listOfRelations)
        exact_match_reranked_relations = self.fuzzywuzzysimExtractor.extract(word_embd_sim_q, listOfRelations)

        rel_scores_combined = Counter()
        for (rel, score) in reranked_relations:
            rel_scores_combined[rel] = score
        for (rel, score) in exact_match_reranked_relations:
            if rel in rel_scores_combined:
                rel_scores_combined[rel] += score
            else:
                rel_scores_combined[rel] = score
        reranked_relations = [(key, value) for key, value in rel_scores_combined.items()]

        for (rel, score) in reranked_relations:
            relation_scores[rel] = score

        print("\t\tQuestion text: {}".format(question))
        print("\t\tQuestion word embedding scores with question text: {}\n".format(", ".join(
            ["{} ({:.2f})".format(rel, score) for (rel, score) in relation_scores.most_common(10)])))

        print("\n\t ------------ Checking word embedding similarities done  ------------ ")

        return relation_scores

    @classmethod
    def readPropertyMap(cls, fname):
        with open(fname, "r") as filein:
            prop_map = {}
            for line in filein:
                prop, label = [x.strip() for x in line.split("\t")]
                if 'dbpedia.org' in prop:
                    prop = prop.replace('http://dbpedia.org/ontology/', 'dbo:').replace('http://dbpedia.org/property/',
                                                                                        'dbp:')
                    if prop not in prop_map:
                        prop_map[prop] = label
        print("Total number of unique properties {}".format(len(prop_map)))
        return prop_map


class FuzzyWuzzySimilarityCalc:
    def __init__(self, prop_map):
        self.prop_map = prop_map

    def similarity(self, context, relation):
        try:
            relation_label = self.prop_map[relation]
            txt_in_brackets = re.findall(r'\(.*\)', relation_label)
            for txt in txt_in_brackets:
                relation_label = relation_label.replace(txt, '').strip()
        except:
            return 0.0
        tokenizer = WordPunctTokenizer()
        rel_tokens_size = len(tokenizer.tokenize(relation_label.lower()))
        context_tokens = tokenizer.tokenize(context.lower())
        if rel_tokens_size > 1:
            context_ngram_tokens = [context_tokens[i:i + rel_tokens_size] for i in
                                    range(len(context_tokens) - rel_tokens_size + 1)]
            context_tokens = context_ngram_tokens

        if process.extractOne(relation_label, context_tokens,
                              scorer=fuzz.token_sort_ratio, score_cutoff=100):
            return rel_tokens_size
        return 0

    def extract(self, context, relationList):
        res = []
        for rel in relationList:
            score = self.similarity(context, rel)
            res.append((rel, score))
        res.sort(key=lambda x: x[1], reverse=True)
        return res


class SimilarityCalc:
    def __init__(self, prop_map, embedding_dict=None, similarity_fn='cosine'):

        self.embeddings = embedding_dict
        # self.vocab = set(nltk.corpus.words.words())
        self.prop_map = prop_map
        self.local_aggregation_fn = max
        self.global_aggregation_fn = lambda x: sum(x) / len(x)
        if similarity_fn == 'cosine':
            self.token_similarity_fn = lambda x, y: (1.0 - scipy.spatial.distance.cosine(x, y))
        elif similarity_fn == 'dot':
            self.token_similarity_fn = np.dot
        else:
            raise NotImplementedError('"{}" not implemented'.format(similarity_fn))
        self.stopwords = set(nltk.corpus.stopwords.words('english'))

    def token_similarity(self, x, y):
        try:
            score = self.token_similarity_fn(self.embeddings[x], self.embeddings[y])
        except Exception as E:
            #             print('Exception in token_similarity', E)
            score = 0.0
        return score

    def similarity(self, context, relation):
        tokenizer = RegexpTokenizer(r'\w+')
        context_words = tokenizer.tokenize(context)
        contextTokens = [x for x in context_words if x.lower() not in self.stopwords]
        # contextTokens = [x for x in context.split() if x.lower() not in self.stopwords]
        try:
            relationString = self.prop_map[relation]
            txt_in_brackets = re.findall(r'\(.*\)', relationString)
            for txt in txt_in_brackets:
                relationString = relationString.replace(txt, '').strip()
        except:
            return 0.0
        relationTokens = [x for x in relationString.split()]

        global_similarities = []
        for relToken in relationTokens:
            local_similarities = []
            # if relToken not in self.vocab:
            #     continue
            for contextToken in contextTokens:
                local_similarities.append(self.token_similarity(relToken, contextToken))
            try:
                global_similarities.append(self.local_aggregation_fn(local_similarities))
            except Exception as E:
                #                 print('Exception in local aggregation', E)
                global_similarities.append(0.0)
        try:
            result = self.global_aggregation_fn(global_similarities)
        except Exception as E:
            #             print('Exception in global aggregation', E)
            result = 0.0
        return result

    # def fuzzy_wuzzy_similarity(self, context, relation):
    #     if relation in self.prop_map:
    #         relation_label = self.prop_map[relation]
    #     else:
    #         return 0
    #     rel_tokens_size = len(nltk.WordPunctTokenizer.tokenize(relation_label))
    #     context_tokens = ngrams(nltk.WordPunctTokenizer.tokenize(context), rel_tokens_size)
    #     max_similarity = 0
    #     for c_token in context_tokens:
    #         sim = fuzz.token_sort_ratio(relation_label, context_tokens)
    #         if sim > max_similarity:
    #             max_similarity = sim
    #     if max_similarity == 100:
    #         return max_similarity * rel_tokens_size
    #     return 0

    def extract(self, context, relationList):
        res = []
        for rel in relationList:
            score = self.similarity(context, rel)
            res.append((rel, score))
        res.sort(key=lambda x: x[1], reverse=True)
        return res





