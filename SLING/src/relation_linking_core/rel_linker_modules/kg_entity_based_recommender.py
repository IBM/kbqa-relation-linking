from collections import Counter
import json
import pickle
from SPARQLWrapper import SPARQLWrapper, JSON

from relation_linking_core.rel_linker_modules.rel_linker_module import RelModule


class KBEntityBasedRecommender(RelModule):

    prefix_map = {'http://dbpedia.org/ontology/': 'dbo:',
                  'http://dbpedia.org/property/': 'dbp:',
                  'http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#': 'dul:',
                  'http://dbpedia.org/class/yago/': 'yago:',
                  'http://umbel.org/umbel/rc/': 'umbel-rc:',
                  'http://www.wikidata.org/entity/': 'wd:',
                  'http://xmlns.com/foaf/0.1/': 'foaf:',
                  'http://purl.org/dc/terms/': 'dct:',
                  'http://purl.org/linguistics/gold/': 'gold:',
                  'http://www.w3.org/1999/02/22-rdf-syntax-ns#': 'rdf:',
                  'http://www.w3.org/2002/07/owl#': 'owl:',
                  'http://www.w3.org/2000/01/rdf-schema#': 'rdfs:'}

    ignored_properties = ['dbo:wikiPageOutDegree', 'dbo:wikiPageLength', 'dbo:wikiPageWikiLinkText',
                          'dbo:wikiPageID']

    def __init__(self, config=None):

        self.config = config
        self.dbpedia_endpoint = config['dbpedia_endpoint']

        print("Initializing KB Entity Based Recommender ....")
        with open(config["sparql_cache_path"]) as f:
            self.sparql_cache = json.load(f)
        print("\t{} cached SPARQL results are loaded!".format(len(self.sparql_cache)))

        with open(config["datatype_rels_path"], 'rb') as f:
            datatype_relations = pickle.load(f)

        self.numeric_relations = datatype_relations['numeric']
        self.date_relations = datatype_relations['date']

        print("Datatype relations:\n\tNumeric: {}\n\tDate: {}".format(len(self.numeric_relations),
                                                                      len(self.date_relations)))

        print("\tKB Entity Based Recommender Initialized.")

    def get_relation_candidates(self, triple_data, params=None):
        relation_scores = Counter()

        subj_text, subj_type, subj_uri, subj_type_uri = triple_data['subj_text'], triple_data['subj_type'], \
                                                        triple_data['subj_uri'], triple_data['subj_type_uri'],
        obj_text, obj_type, obj_uri, obj_type_uri = triple_data['obj_text'], triple_data['obj_type'], \
                                                    triple_data['obj_uri'], triple_data['obj_type_uri']

        print("\t------------ Fetching KB relations: ------------")
        if subj_uri or obj_uri or subj_type_uri or obj_type_uri:
            all_relations = self.get_all_relations(subj=subj_uri, obj=obj_uri)
            print('\tAll entity relations: {}'.format(len(all_relations)))
            if len(all_relations) < 20:
                print('\t\t{}'.format(all_relations))
            datatype_matched = []
            strict_relations, strict_weight = self.get_strict_relations(subj=subj_uri, subj_type=subj_type_uri,
                                                                               obj=obj_uri, obj_type=obj_type_uri)
            if triple_data['obj_id'] == triple_data['amr_unknown_var'] and 'answer_datatype' in triple_data:
                if triple_data['answer_datatype']:
                    print("\t\tDatatype range: {}".format(triple_data['answer_datatype']))
                if triple_data['answer_datatype'] in ['AGE', 'CARDINAL', 'MEASURE']:
                    datatype_matched = self.numeric_relations.intersection(set(all_relations))
                elif triple_data['answer_datatype'] == 'DATE':
                    datatype_matched = self.date_relations.intersection(set(all_relations))
                # elif triple_data['obj_type'] == 'person' or triple_data['subj_type'] == 'person':
                #     datatype_matched = ServerUtils.person_relations.intersection(set(all_relations))
                # elif triple_data['obj_type'] == 'place' or triple_data['obj_type'] == 'place':
                #     datatype_matched = ServerUtils.place_relations.intersection(set(all_relations))
            if triple_data['obj_type'] == 'ordinal-entity':
                datatype_matched = self.numeric_relations.intersection(set(all_relations))
            if len(datatype_matched) > 0:
                strict_relations = datatype_matched
                strict_weight = 2

            print('\tRelations with constraints: {}'.format(len(strict_relations)))
            if len(strict_relations) < 20:
                print('\t\t{}'.format(strict_relations))

            if strict_relations:
                for rel in strict_relations:
                    relation_scores[rel] += strict_weight
            if all_relations:
                for rel in all_relations:
                    if rel not in strict_relations:
                        relation_scores[rel] += 1

        print("\t\tRelation scores: {}".format(", ".join(
            ["{} ({:.2f})".format(count[0], count[1]) for count in relation_scores.items()])).encode("UTF-8"))

        print("\t ------------ KB relations done ------------")
        print("\n")

        return relation_scores

    def get_all_relations(self, subj=None, obj=None):

        relations = list()

        if not subj and not obj:
            return list()

        triple_pattern = ""
        if subj and obj:
            triple_pattern = " <{}> ?prop <{}> . ".format(subj, obj)
        if subj:
            triple_pattern = " <{}> ?prop ?object . ".format(subj)
        if obj:
            triple_pattern = " ?subject ?prop <{}> . ".format(obj)
        query_string = "PREFIX dbo: <http://dbpedia.org/ontology/>  SELECT DISTINCT ?prop WHERE { " + triple_pattern + \
                       " } "

        if query_string in self.sparql_cache:
            relations += self.sparql_cache[query_string]
        else:
            print("WARNING - SPARQL - Not found in cache\n\t{} ".format(query_string))
            sparql = SPARQLWrapper(self.dbpedia_endpoint)
            sparql.setQuery(query_string)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            for result in results["results"]["bindings"]:
                rel_uri = KBEntityBasedRecommender.get_curie(result['prop']['value'])
                if rel_uri.startswith("dbo:") or rel_uri.startswith("dbp:"):
                    relations.append(rel_uri)

            self.sparql_cache[query_string] = relations
            with open(self.config["sparql_cache_path"], 'w') as f:
                json.dump(self.sparql_cache, f)


        # filtering ignored relations
        relations = [relation for relation in relations if relation not in KBEntityBasedRecommender.ignored_properties]
        return relations

    def get_strict_relations(self, subj=None, obj=None, subj_type=None, obj_type=None):

        relations = list()
        weight = 0
        triple_pattern = ""
        if subj and obj:
            weight += 2
            triple_pattern += " <{}> ?prop <{}> . ".format(subj, obj)
        elif subj:
            weight += 1
            triple_pattern += " <{}> ?prop ?object . ".format(subj)
            if obj_type:
                weight += 1
                triple_pattern += " { <" + subj + "> ?prop " + obj_type + " } UNION { ?object a " + obj_type + " }  "
        elif obj:
            weight += 1
            triple_pattern += " ?subject ?prop <{}> . ".format(obj)
            if subj_type:
                weight += 1
                triple_pattern += " ?subject a {} . ".format(subj_type)
        elif subj_type and obj_type:
            weight += 1
            triple_pattern += " ?subject ?prop ?object . ?subject a {} . ?object a {} . ".format(subj_type, obj_type)

        if "" == triple_pattern:
            return list(), 0

        query_string = "PREFIX dbo: <http://dbpedia.org/ontology/>  SELECT DISTINCT ?prop WHERE { " + triple_pattern + \
                       " } "

        if query_string in self.sparql_cache:
            relations += self.sparql_cache[query_string]
        else:
            print("WARNINGG - SPARQL - Not found in cache\n\t{} ".format(query_string))
            sparql = SPARQLWrapper(self.dbpedia_endpoint)
            sparql.setQuery(query_string + " LIMIT 200")
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            for result in results["results"]["bindings"]:
                rel_uri = KBEntityBasedRecommender.get_curie(result['prop']['value'])
                if rel_uri.startswith("dbo:") or rel_uri.startswith("dbp:"):
                    relations.append(rel_uri)
            self.sparql_cache[query_string] = relations
            with open(self.config["sparql_cache_path"], 'w') as f:
                json.dump(self.sparql_cache, f)

        # filtering ignored relations
        relations = [relation for relation in relations if relation not in KBEntityBasedRecommender.ignored_properties]
        return relations, weight

    @classmethod
    def get_curie(cls, iri):
        for ns in KBEntityBasedRecommender.prefix_map:
            iri = iri.replace(ns, KBEntityBasedRecommender.prefix_map.get(ns))
        return iri



