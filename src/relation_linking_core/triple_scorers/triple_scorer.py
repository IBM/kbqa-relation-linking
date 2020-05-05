# Interface which returns a triple score based on relation candidate scores
class TripleScorer:
    """
    Basic Interface for Triple Scorer.
    """
    def __init__(self):
        pass

    def score(self, scores_dict, params=None):
        """
        Takes the relation candidate scores and returns a score for the triple
        :param scores_dict:
        :return: dict of str --> dict
        """
        raise NotImplementedError('Please implement the scorer function')