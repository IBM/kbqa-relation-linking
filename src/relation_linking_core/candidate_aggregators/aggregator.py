# Interface which returns final rel scores after aggregation of different module scores
class Aggregator:
    """
    Basic Interface for Aggregators of different relation linking module scores
    """
    def __init__(self):
        pass

    def aggregate(self, scores_dict, params=None):
        """
        Takes in scores_dict (dict of dicts) corresponding to different modules
        and returns a single dict of {rel : score}
        :param scores_dict:
        :return: dict of str --> dict
        """
        raise NotImplementedError('Please implement an aggregate function')