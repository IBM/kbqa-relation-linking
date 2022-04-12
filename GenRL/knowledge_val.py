import argparse
import json
import itertools
import traceback
import numpy as np
from os.path import exists
from SPARQLWrapper import SPARQLWrapper, JSON

wh_terms = ["Who", "What", "Where", "Which", "How many", "Count", "When", "List", "Name", "Give", "Was", "Is",
            "Does", "Did"]


def expand_paths(paths, include_invalid=False):
    new_triples = list()
    kg_to_node = dict()
    for triple in paths:
        subj_node = triple['a1_amr']
        obj_node = triple['a2_amr']
        subj = triple['a1_kg']
        obj = triple['a2_kg']
        relation = triple['relation']
        kg_to_node[subj] = subj_node
        kg_to_node[obj] = obj_node

        one_hop = list()
        new_triple = [None, None, None]
        if not subj.startswith("?"):
            new_triple[0] = "<" + subj + ">"
        else:
            new_triple[0] = subj
        if not obj.startswith("?"):
            new_triple[2] = "<" + obj + ">"
        else:
            new_triple[2] = obj

        tr_candidates = [[new_triple[0], "dbo:" + relation, new_triple[2], 1],
                         [new_triple[0], "dbp:" + relation, new_triple[2], 1],
                         [new_triple[2], "dbo:" + relation, new_triple[0], 1],
                         [new_triple[2], "dbp:" + relation, new_triple[0], 1]]

        for tr in tr_candidates:
            if run_ask_query([tr]):
                one_hop.append(tr)
            elif include_invalid:
                tr[3] = 0.5
                one_hop.append(tr)
        new_triples.append(one_hop)
    return new_triples, kg_to_node


def run_ask_query(triple_patterns, validation_cache_path, validation_cache, sparql_endpoint):
    query_string = "PREFIX dbo: <http://dbpedia.org/ontology/> \
                    PREFIX dbp: <http://dbpedia.org/property/> ASK WHERE {"
    for triple_pattern in triple_patterns:
        query_string += f"{triple_pattern[0]} {triple_pattern[1]} {triple_pattern[2]} . "
    query_string += " } "

    if query_string in validation_cache:
        return validation_cache[query_string]

    sparql = SPARQLWrapper(sparql_endpoint)
    sparql.setQuery(query_string)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        validation_cache[query_string] = results["boolean"]
        if len(validation_cache) % 10 == 0:
                with open(validation_cache_path, 'w', encoding='utf-8') as output_file:
                    json.dump(validation_cache, output_file)
        return results["boolean"]
    except Exception as ex:
        print(f"RUN ASK QUERY ERROR: {query_string}\n{traceback.format_exc()}")
        return False


def is_ask_question(q_text):
    wh_term = get_wh_term(q_text)
    if wh_term:
        if wh_term.lower() in ["was", "is", "does", "did"]:
            return True
        else:
            return False
    else:
        False


def get_wh_term(q_text):
    selected_term = None
    for term in wh_terms:
        if term in q_text.split():
                selected_term = term
                break
        if not selected_term:
            for term in wh_terms:
                if term.lower() in q_text.lower().split():
                    selected_term = term.lower()
                    break
        if not selected_term:
            for term in wh_terms:
                if term.lower() in q_text.lower():
                    selected_term = term.lower()
                    break
    return selected_term


def multihop_validation(triple_patterns,  validation_cache_path, validation_cache, sparql_endpoint,
                        top_k=1, include_invalid=False):
    ask_positive_threshold = 10
    ask_count = 0
    graphs_to_val = list()
    for can_list in itertools.product(*triple_patterns):
        graph_score = np.prod([can[3] for can in can_list])
        graphs_to_val.append([can_list, graph_score])
    graphs_to_val = sorted(graphs_to_val, key=lambda a: a[1], reverse=True)

    print(f"\nValidating graphs:")
    last_score = 0
    candidate_graphs = list()
    for graph, score in graphs_to_val:
        ask_count += 1
        if len(candidate_graphs) >= top_k and last_score > score:
            return candidate_graphs
        last_score = score

        print(f"\tcan_size: {len(candidate_graphs)} graph: {graph} score: {score}")
        if len(graph) >= 2 and graph[1][0].startswith("?") and graph[1][2].startswith("?"):
            if graph[0][0] + graph[0][1] == graph[1][0] + graph[1][1]:
                print(f"\t\tskipping the graph. Symmetric mapping.")
                continue
            elif graph[0][1] + graph[0][2] == graph[1][1] + graph[1][2]:
                print(f"\t\tskipping the graph. Symmetric mapping.")
                continue

        if run_ask_query(graph, validation_cache_path, validation_cache, sparql_endpoint):
            print(f"\t\tgraph validated!")
            candidate_graphs.append(graph)

        if include_invalid and ask_count > ask_positive_threshold:
            print("\nNo valid graph found, using invalid graphs.")
            return [graph for graph, score in graphs_to_val[:top_k]]

    if not candidate_graphs and graphs_to_val:
        return [graph for graph, score in graphs_to_val[:top_k]]

    return candidate_graphs


def validate(ques, validation_cache_path, validation_cache, sparql_endpoint):
    q_id = ques['id']
    text = ques['text']
    paths = ques['path']
    for path in paths:
        path['relation'] = path['relation'].replace("http://dbpedia.org/ontology/", "").replace(
            "http://dbpedia.org/property/", "")
    ask_question = True if is_ask_question(text) else False
    # print(f"ask: {ask_question}, text: {text}")
    ask_question = False

    candidates, kg_to_node = expand_paths(paths, ask_question)
    candidate_graphs = multihop_validation(candidates, validation_cache_path, validation_cache, sparql_endpoint,
                                           include_invalid=ask_question)
    validated_triples = list()
    for graph in candidate_graphs:
        valid_graph = list()
        for tr in graph:
            valid_graph.append(tr[:3])
        validated_triples.append(valid_graph)
    ques_output = {"q_id": q_id, "text": text, "validated_triples": validated_triples, "kg2node": kg_to_node,
                   "paths": paths}
    return ques_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_output')
    parser.add_argument('--val_cache')
    parser.add_argument('--sparql_endpoint')
    parser.add_argument('--val_output')

    args = parser.parse_args()
    sparql_endpoint = args.sparql_endpoint
    model_output_path = args.model_output
    validation_cache_path = args.val_cache
    validation_output_path = args.val_output

    with open(model_output_path) as train_file:
        data = json.load(train_file)
    print(f"loaded {len(data)} model results to validate!")

    with open(validation_cache_path) as json_file:
        validation_cache = json.load(json_file)
    print(f"loaded {len(validation_cache)} cached query responses!")

    validated_triple_results = list()
    for i, ques in enumerate(data):
        validated_output = validate(ques, validation_cache_path, validation_cache, sparql_endpoint)
        validated_triple_results.append(validated_output)

    with open(validation_output_path, 'w', encoding='utf-8') as output_file:
        json.dump(validated_triple_results, output_file, indent=2, ensure_ascii=False)
