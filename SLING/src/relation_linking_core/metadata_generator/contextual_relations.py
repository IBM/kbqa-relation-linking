import pickle


class ContextualRelationsModule:

    def __init__(self, config=None):
        with open('../data/contextual-relations.pkl', 'rb') as f:
            self.contextual_relations_cache = pickle.load(f)
        print("Contextual relations :\n\t{} loaded.".format(len(self.contextual_relations_cache)))

    def get_contextual_relations(self, q_text):
        if q_text in self.contextual_relations_cache:
            return self.contextual_relations_cache[q_text]
        else:
            print("WARNING: question not found in cache.\n\t{}".format(q_text))
            return list()

