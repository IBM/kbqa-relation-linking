from collections import Counter
import traceback

from relation_linking_core.metadata_generator.amr_utils import AMRUtils
from relation_linking_core.metadata_generator.amr_graph_to_triples import AMR2Triples
from relation_linking_core.rel_linker_modules.kg_entity_based_recommender import KBEntityBasedRecommender
from relation_linking_core.rel_linker_modules.statistical_mappings import StatisticalRelationMapping
from relation_linking_core.rel_linker_modules.neural_relation_linking import NeuralRelationLinking
from relation_linking_core.rel_linker_modules.question_similarity_based_relations import QuestionSimilarityBasedRelRecommender
from relation_linking_core.metadata_generator.entity_utils import EntityUtils
from relation_linking_core.metadata_generator.answer_type_prediction import AnswerTypePredictionService
from relation_linking_core.metadata_generator.contextual_relations import ContextualRelationsModule
from relation_linking_core.candidate_aggregators.simple_aggregator import SimpleAggregator
from relation_linking_core.triple_scorers.simple_triple_scorer import SimpleTripleScorer


class KBQARelationLinkingService:

    def __init__(self, config):
        self.kb_entity_based_linking = KBEntityBasedRecommender(config)
        self.statistical_mapping_module = StatisticalRelationMapping(config)
        self.neural_relation_linking = NeuralRelationLinking(config)
        self.similarity_based_relation_linking = QuestionSimilarityBasedRelRecommender(config)
        self.contextual_relations_module = ContextualRelationsModule(config)
        self.answer_type_prediction_module = AnswerTypePredictionService(config)

        self.aggregator = SimpleAggregator(config)
        self.triple_scorer = SimpleTripleScorer()

    def process(self, question_text, amr_graph):

        print(f"Text: {question_text}")
        print(f"EAMR: {amr_graph}")

        output_relations = list()

        try:
            amr_graph = AMRUtils.fix_amr_graph(amr_graph)
            triple_info, names, reified_to_rel, top_node = AMR2Triples.get_flat_triples(question_text, amr_graph)
            entities = EntityUtils.get_entities(amr_graph)

            amr_nodes = set()
            print("\nTriples:")
            for triple in triple_info:
                amr_nodes.update({triple['subj_text'].lower(), triple['subj_type'].lower(), triple['obj_text'].lower(),
                                  triple['obj_type'].lower()})
                print("\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(triple['subj_id'], triple['subj_text'], triple['subj_type'],
                    triple['predicate'], triple['obj_id'], triple['obj_text'], triple['obj_type']))

            amr_entity_alignments, normalized_to_surface_form = EntityUtils.align_entities(amr_nodes, entities)
            #amr_entity_alignments, normalized_to_surface_form = EntityUtils.align_entities_annotated(question_text, amr_nodes, entities)

            print("Entity alignment:")
            for alignment in amr_entity_alignments:
                print("\t{}: {}".format(alignment, amr_entity_alignments[alignment]))
            print('\n')

            answer_types = self.answer_type_prediction_module.get_answer_types(question_text)
            print("Answer Types:\n{}".format(answer_types))

            # check if answer type was a literal/data type
            answer_datatype = None
            if len(answer_types) > 0 and answer_types[0][0]:
                answer_type = answer_types[0][0]
                if answer_type in ['AGE', 'CARDINAL', 'DATE', 'MEASURE']:
                    answer_datatype = answer_type

            contextual_relations = self.contextual_relations_module.get_contextual_relations(question_text)
            contextual_relation_scores = Counter()
            for index, rel in enumerate(contextual_relations):
                contextual_relation_scores[rel] = 0.6 if index < 5 else 0.4
                if index + 1 == len(contextual_relations): contextual_relation_scores[rel] = 0.3

            response_list = list()

            triple_id = 0
            for triple in triple_info:

                if triple['subj_type'] == 'multi-sentence':
                    continue

                triple_id += 1
                rel = triple['predicate']
                triple['rel_split'] = rel.split('.')
                triple['text'] = question_text

                if triple['rel_split'][0] == 'give-01' or triple['rel_split'][0] == 'list-01':
                    # TODO handle imperative cases better, check for imperative
                    continue

                # try to link subject, subject type, object and object type to KB entities
                EntityUtils.link_entities_types(triple, amr_entity_alignments, answer_types)

                if answer_datatype:
                    triple['answer_datatype'] = answer_datatype

                KBQARelationLinkingService.print_triple(triple_id, "direct", triple)
                scores_dict = self.do_relation_linking(triple, contextual_relation_scores, normalized_to_surface_form,
                                                       reified_to_rel)


                inverse_triple = KBQARelationLinkingService.get_inverse_triple(triple)
                KBQARelationLinkingService.print_triple(triple_id, "inverse", inverse_triple)
                inverse_scores_dict = self.do_relation_linking(inverse_triple, contextual_relation_scores,
                                                               normalized_to_surface_form, reified_to_rel)

                # Do min-max normalization of scores from each dict here (from the union of direct and inverse directions,
                # to get a better triple scoring)
                scores_dict, inverse_scores_dict = KBQARelationLinkingService.do_normalization(scores_dict, inverse_scores_dict)

                direct_amr_scores = scores_dict['statistical_rel_mapping_scores']
                inverse_amr_scores = inverse_scores_dict['statistical_rel_mapping_scores']
                union_amr_scores = Counter()
                for rel, score in direct_amr_scores.items():
                    union_amr_scores[rel] = score
                    for rel, score in inverse_amr_scores.items():
                        union_amr_scores[rel] = max(score, union_amr_scores[rel])
                    scores_dict['statistical_rel_mapping_scores'] = union_amr_scores
                    inverse_scores_dict['statistical_rel_mapping_scores'] = union_amr_scores

                relations_with_scores = self.aggregator.aggregate(scores_dict)
                triple_score = self.triple_scorer.score(scores_dict, relations_with_scores)

                inverse_relations_with_scores = self.aggregator.aggregate(inverse_scores_dict)
                inverse_triple_score = self.triple_scorer.score(inverse_scores_dict, inverse_relations_with_scores)

                KBQARelationLinkingService.print_relation_scores(triple_score, inverse_triple_score, relations_with_scores,
                                                  inverse_relations_with_scores)


                # we only pick one direction for each triple based on the score
                if triple_score > inverse_triple_score:
                    response_list.append([triple, relations_with_scores, triple_score])
                else:
                    response_list.append([inverse_triple, inverse_relations_with_scores, inverse_triple_score])

            # sort the response list based on
            response_list = sorted(response_list, reverse=True, key=lambda x: x[2])

            pruned_triple_count = KBQARelationLinkingService.pruned_triple_count(response_list)

            for response_item in response_list:
                print('{} {}\n'.format(response_item[0]['predicate'], ", ".join(
                    ["{} ({:.2f})".format(rel[0], rel[1]) for rel in response_item[1].most_common(10)])))

            return self.prepare_final_relation_list(response_list, pruned_triple_count)

        except Exception as ex:
            print("ERROR - {}".format(str(ex)))
            print(traceback.format_exc())
            raise ex

        return output_relations

    def do_relation_linking(self, triple_data, contextual_relations, normalized_to_surface_form, reified_to_rel):

        # if the subject is a literal, we don't consider that triple
        if (triple_data['subj_id'] == triple_data['amr_unknown_var'] and 'answer_datatype' in triple_data) or triple_data['subj_type'] == 'ordinal-entity':
            if 'answer_datatype' in triple_data:
                print("\t skipping the triple with datatype subject: {}".format(triple_data['answer_datatype']))
            empty_scores_dict = {
                'kg_entity_recommender_scores': Counter(),
                'contextual_rel_recommender_scores': Counter(),
                'statistical_rel_mapping_scores': Counter(),
                'neural_model_scores': Counter(),
                'similarity_based_scores': Counter()
            }
            return empty_scores_dict

        kg_entity_recommender_scores = self.kb_entity_based_linking.get_relation_candidates(triple_data, {})

        statistical_rel_mapping_scores = self.statistical_mapping_module.get_relation_candidates(triple_data, {
            "reified_to_rel": reified_to_rel})

        neural_model_scores = self.neural_relation_linking.get_relation_candidates(triple_data, {
            "normalized_to_surface_form": normalized_to_surface_form })

        list_of_relations = set().union(set(kg_entity_recommender_scores.keys()),
                                      set(statistical_rel_mapping_scores.keys()),
                                      set(neural_model_scores.keys()))

        # if kg_entity_recommender_scores.keys():
        #     list_of_relations = kg_entity_recommender_scores.keys()
        # else:
        #     list_of_relations = set().union(set(statistical_rel_mapping_scores.keys()),
        #                               set(neural_model_scores.keys()))

        similarity_based_scores = self.similarity_based_relation_linking.get_relation_candidates(triple_data, {
            "listOfRelations": list_of_relations})

        scores_dict = {
            'kg_entity_recommender_scores': kg_entity_recommender_scores,
            'contextual_rel_recommender_scores': contextual_relations,
            'statistical_rel_mapping_scores': statistical_rel_mapping_scores,
            'neural_model_scores': neural_model_scores,
            'similarity_based_scores': similarity_based_scores
        }

        return scores_dict

    @classmethod
    def get_inverse_triple(cls, triple):
        inverse_triple = dict()
        # swapping subject and object
        inverse_triple['text'] = triple['text']
        inverse_triple['amr_unknown_var'] = triple['amr_unknown_var']

        inverse_triple['subj_id'] = triple['obj_id']
        inverse_triple['subj_text'] = triple['obj_text']
        inverse_triple['subj_type'] = triple['obj_type']
        inverse_triple['subj_uri'] = triple['obj_uri']
        inverse_triple['subj_type_uri'] = triple['obj_type_uri']

        inverse_triple['obj_id'] = triple['subj_id']
        inverse_triple['obj_text'] = triple['subj_text']
        inverse_triple['obj_type'] = triple['subj_type']
        inverse_triple['obj_uri'] = triple['subj_uri']
        inverse_triple['obj_type_uri'] = triple['subj_type_uri']

        if len(triple['rel_split']) == 3:
            inverse_triple['rel_split'] = [triple['rel_split'][0], triple['rel_split'][2], triple['rel_split'][1]]
        elif len(triple['rel_split']) == 2:
            inverse_triple['rel_split'] = [triple['rel_split'][1], triple['rel_split'][0]]
        inverse_triple['predicate_id'] = triple['predicate_id']
        inverse_triple['predicate'] = '.'.join(inverse_triple['rel_split'])

        return inverse_triple

    @classmethod
    def do_normalization(cls, scores_dict, inverse_scores_dict):
        normalized_scores_dict, normalized_inverse_scores_dict = {}, {}

        if set(scores_dict.keys()) != set(inverse_scores_dict.keys()):
            raise ValueError('There is a scoring measure that is not common to both direct and inverse mappings')

        for module_name in scores_dict.keys():
            if module_name != 'corrected_question_similarity_based_rel_recommender_scores':
                normalized_scores_dict[module_name] = scores_dict[module_name]
                normalized_inverse_scores_dict[module_name] = inverse_scores_dict[module_name]
                continue

            direct_scores, inverse_scores = scores_dict[module_name], inverse_scores_dict[module_name]
            normalized_direct_scores, normalized_inverse_scores = Counter(), Counter()

            direct_scores_set = set(direct_scores.values())
            inverse_scores_set = set(inverse_scores.values())
            all_scores = direct_scores_set.union(inverse_scores_set)

            if len(all_scores) == 0:
                normalized_scores_dict[module_name] = normalized_direct_scores
                normalized_inverse_scores_dict[module_name] = normalized_inverse_scores
                continue

            max_val, min_val = max(all_scores), min(all_scores)
            for rel, score in direct_scores.items():
                normalized_direct_scores[rel] = 0.0 if (max_val - min_val == 0) else (score - min_val) / (
                            max_val - min_val)
            for rel, score in inverse_scores.items():
                normalized_inverse_scores[rel] = 0.0 if (max_val - min_val == 0) else (score - min_val) / (
                            max_val - min_val)

            normalized_scores_dict[module_name] = normalized_direct_scores
            normalized_inverse_scores_dict[module_name] = normalized_inverse_scores

        return normalized_scores_dict, normalized_inverse_scores_dict

    def prepare_final_relation_list(self, response_list, relation_count):

        output_relation_list = list()
        unified_counter = Counter()

        for triple in response_list:
            if triple[1]:
                for rel in triple[1].most_common(1):
                    if rel[0] not in output_relation_list:
                        output_relation_list.append(rel[0])
                        break
                if len(output_relation_list) == relation_count:
                    break
                unified_counter += triple[1]

        other_rels = unified_counter.most_common()
        for rel in other_rels:
            if len(output_relation_list) < relation_count and rel[0] not in output_relation_list:
                output_relation_list.append(rel[0])

        dbo_rels = []
        for rel in output_relation_list:
            if rel.startswith("dbp:"):
                dbo_rel = rel.replace("dbp:", "dbo:")
                if dbo_rel in self.similarity_based_relation_linking.prop_map:
                    dbo_rels.append(dbo_rel)

        return output_relation_list + dbo_rels

    @classmethod
    def pruned_triple_count(cls, response_list):
        #min_triples = 2 if len(response_list) > 2 else 1

        new_response_list = list()
        seen_relations = set()
        for response in response_list:
            triple, relations_with_scores, score = response
            top_k = relations_with_scores.most_common(2)
            top_k_size = len(top_k)
            if len(seen_relations.intersection({rel[0] for rel in top_k})) == top_k_size:
                continue
            new_response_list.append(response)
            seen_relations.update({rel[0] for rel in relations_with_scores.most_common(2)})
        #return max(min_triples, len(new_response_list))
        return len(new_response_list)

    @classmethod
    def print_triple(cls, triple_id, comment, triple_data):
        print(
            "triple_{}_{}:\n\t\tpredicate: {}\n\t\tsubj: {},\t{},\t{},\t{}\n\t\tobject: {},\t{},\t{},\t{}\n".format(
                triple_id,
                comment,
                triple_data['predicate'],
                triple_data['subj_text'], triple_data['subj_type'], triple_data['subj_uri'],
                triple_data['subj_type_uri'],
                triple_data['obj_text'], triple_data['obj_type'], triple_data['obj_uri'],
                triple_data['obj_type_uri']))

    @classmethod
    def print_relation_scores(cls, triple_score, inverse_triple_score, direct_scores, inverse_scorees):
        print("\nTriple score:\n\tdirect: {}\n\t\t{}\n\tinverse: {}\n\t\t{}\n".format(triple_score,
              ", ".join([ "{} ({:.2f})".format(count[0], count[1]) for count in direct_scores.most_common(10)]),
                                                                                      inverse_triple_score,
              ", ".join(["{} ({:.2f})".format(count[0], count[1]) for count in inverse_scorees.most_common(10)])))

