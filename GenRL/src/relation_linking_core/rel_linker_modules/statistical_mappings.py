import pickle
from collections import Counter
from relation_linking_core.rel_linker_modules.rel_linker_module import RelModule


class StatisticalRelationMapping(RelModule):

    def __init__(self, config=None):

        print("Initializing Statistical Relation Mapping ....")

        with open('../data/probbank-dbpedia.pkl', 'rb') as f:
            db = pickle.load(f)
        self.relation_scores = db['relation_scores']
        self.rel_arg_scores = db['rel_arg_scores']
        self.binary_relation_scores = db['binary_relation_scores']
        print("Statistical Mappings: {} rel arg, {} binary, {} predicates".format(len(self.rel_arg_scores),
                                                                                  len(self.binary_relation_scores),
                                                                                  len(self.relation_scores)))
        print("\tStatistical Relation Mapping Initialized.")

    def get_relation_candidates(self, triple_data, params=None):

        relation_scores = Counter()
        rel_list = list()
        reified_to_rel = params['reified_to_rel']

        print("\n\t ------------ AMR statistical mapping ------------")

        rel_split = triple_data['rel_split']
        rel_args_predicate = '.'.join(rel_split)

        if len(rel_split) == 3:
            if rel_split[1] not in ['time', 'quant']:
                # skipping combinations with these arguments as subject
                propbank_predicate = rel_split[0]
                if propbank_predicate == 'give-01' or propbank_predicate == 'list-01':
                    # TODO handle imperative cases better, check for imperative
                    pass
                else:
                    if triple_data['predicate_id'] in reified_to_rel:
                        modified_predicate = "{}.{}".format(rel_args_predicate, reified_to_rel[triple_data['predicate_id']])
                    else:
                        modified_predicate = rel_args_predicate

                    if modified_predicate in self.rel_arg_scores:
                        rel_list += self.rel_arg_scores[modified_predicate]
                    else:
                        if propbank_predicate in self.relation_scores:
                            rel_list += self.relation_scores[propbank_predicate]
                        else:
                            # checking if other senses are present
                            for i in range(1, 5):
                                other_sense = "{}0{}".format(propbank_predicate[:-2], i)
                                if other_sense in self.relation_scores:
                                    rel_list += self.rel_arg_scores[other_sense]
                                    break
        elif len(rel_split) == 2:
            if rel_args_predicate in self.binary_relation_scores:
                rel_list += self.binary_relation_scores[rel_args_predicate]
        elif len(rel_split) == 1:
            if rel_split[0] in self.relation_scores:
                rel_list += self.relation_scores[rel_split[0]]

        for rel in rel_list:
            relation_scores[rel['rel']] += float(rel['score'])

        print("\n\t\tStatistical relation scores: {}".format(", ".join(
            ["{} ({:.2f})".format(count[0], count[1]) for count in relation_scores.most_common(10)])))

        print("\t ------------ AMR statistical mapping done ------------\n")

        return relation_scores