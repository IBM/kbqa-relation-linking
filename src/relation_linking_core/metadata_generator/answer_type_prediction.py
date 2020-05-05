import pickle


class AnswerTypePredictionService:

    def __init__(self, config=None):
        with open('../data/answer-types.pkl', 'rb') as f:
            self.answer_type_cache = pickle.load(f)
        print("Answer Type Prediction:\n\tloaded {} cached answer types!".format(len(self.answer_type_cache)))

    def get_answer_types(self, q_text):
        if q_text in self.answer_type_cache:
            return self.answer_type_cache[q_text]
        else:
            print("WARNING: question not found in cache.\n\t{}".format(q_text))
            return list()





