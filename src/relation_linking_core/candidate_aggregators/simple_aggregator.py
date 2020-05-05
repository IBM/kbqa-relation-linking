from relation_linking_core.candidate_aggregators.aggregator import Aggregator
from collections import Counter


# Interface which returns final rel scores after aggregation of different module scores
class SimpleAggregator(Aggregator):
    """
    Uses just the rels from kg_entity_recommender scores, and reranks among them using neural model
    """

    def __init__(self, config=None):
        self.module_weights = config["module_weights"]

    def aggregate(self, scores_dict, params=None):
        """
        Takes in scores_dict (dict of dicts) corresponding to different modules
        and returns a list of (rel, score) tuples ordered in descending scores.
        :param scores_dict:
        :return: dict of str --> dict
        """
        aggregated_scores = Counter()
        for module in scores_dict:
            module_weight = self.module_weights[module] if module in self.module_weights else 1
            module_dict = scores_dict[module]
            for k, v in module_dict.items():
                aggregated_scores[k] += (v * module_weight)
        return aggregated_scores