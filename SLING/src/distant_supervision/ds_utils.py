from SPARQLWrapper import SPARQLWrapper, JSON
from datetime import datetime
import nltk
import decimal


class DistantSupervisionUtils:

    # DBpedia endpoint with anchor text for each entity
    dbpedia_201610 = 'fill in endpoint'
    # https://en.wikipedia.org/wiki/Wikipedia:Manual_of_Style/Dates_and_numbers
    date_format_list = ['%B %d, %Y', '%b %d, %Y', '%d %B %Y', '%d %b %Y', '%B %d', '%b %d', '%d %B', '%d %b', '%B %Y',
                       '%B %Y', '%Y-%m-%-d', '%Y-%m-%d', '%d-%m-%Y', '%-d-%m-%Y', '%m-%d', '%Y']

    @classmethod
    def get_all_date_variants(cls, date_value):
        date_object = datetime.strptime(date_value, '%Y-%m-%d')
        date_variants = list()
        for date_format in DistantSupervisionUtils.date_format_list:
            formatted = date_object.strftime(date_format).lstrip("0").replace(" 0", " ")
            if formatted not in date_variants:
                date_variants.append(formatted)
        return date_variants

    @classmethod
    def get_all_number_variants(cls, num_value):
        num_variants = list()
        num_value = float(num_value)
        if int(num_value) == num_value:
            num_variants.append(str(int(num_value)))
        else:
            num_variants.append(str(num_value))
            tup = decimal.Decimal(str(num_value)).as_tuple()
            decimals = abs(tup.exponent) - 1
            while decimals > 0:
                format_var = "{:." + str(decimals)+"f}"
                num_variants.append(format_var.format(float(num_value)))
                decimals -= 1
        while num_value > 1000:
            num_value = num_value/1000
            num_variants.append("{:.2f}".format(num_value))
            num_variants.append("{:.1f}".format(num_value))
            if num_value > 10:
                num_variants.append(str(int(num_value)))
        return num_variants

    @classmethod
    def sort_by_similarity(cls, label, alias_list):
        alias_with_scores = list()
        alias_list.append(label)
        label_set = set(label)

        for alias in alias_list:
            score = nltk.jaccard_distance(label_set, set(alias))
            alias_with_scores.append([alias, score])

        alias_with_scores = sorted(alias_with_scores, key=lambda alias_with_score: alias_with_score[1])

        return alias_with_scores

    @classmethod
    def get_link_text(cls, sparql_endpoint, dbpedia_uri):
        labels = set()
        sparql = SPARQLWrapper(sparql_endpoint, agent='sparqlwrapper')
        sparql.setQuery("PREFIX dbo: <http://dbpedia.org/ontology/> " +
                        "SELECT DISTINCT ?label { "
                        " <" + dbpedia_uri + "> dbo:wikiPageWikiLinkText ?label "
                                             "} ")
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        for result in results["results"]["bindings"]:
            label = result['label']['value']
            if label and label != '':
                labels.add(label.strip().lower())

        labels = list(set(labels))
        return labels


if __name__ == '__main__':
    labels = DistantSupervisionUtils.get_link_text(sparql_endpoint=DistantSupervisionUtils.dbpedia_201610,
                                          dbpedia_uri='http://dbpedia.org/resource/Barack_Obama')
    sorted_labels = DistantSupervisionUtils.sort_by_similarity("Barack Obama", labels)
    for label in sorted_labels:
        print(label)