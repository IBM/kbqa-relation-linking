from relation_linking_core.triple_scorers.triple_scorer import TripleScorer


class SimpleTripleScorer(TripleScorer):

    def __init__(self):
        pass

    def score(self, scores_dict, relations_with_scores, params=None):
        return max(relations_with_scores.values()) if len(relations_with_scores) > 0 else 0.0