# Interface which relation linking candidates with scores
class RelModule:
    """
    Basic Interface for Relation Extractors and Linkers
    """
    def __init__(self, config):
        pass

    def get_relation_candidates(self, text: str, params=None):
        """
        Takes in text and returns surface forms with top K entities
        :param text
        :param params
        :return: Counter a counter where the key is the candidate relation and count is the score
        """
        raise NotImplementedError('Please implement REL function')