import argparse
import json
from relation_linking_core.relation_linking_service import KBQARelationLinkingService


def precision_recall_f1(predictions, golds):
    p, r, f1 = 0.0, 0.0, 0.0
    if len(predictions) > 0 and len(golds) > 0:
        p = (len(set(predictions) & set(golds))) / len(set(predictions))
        r = (len(set(predictions) & set(golds))) / len(golds)
        if p + r > 0:
            f1 = f1_score(p, r)
    return p, r, f1


def f1_score(p, r):
    if p + r == 0:
        return 0
    return 2 * ((p * r) / (p + r))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path')
    parser.add_argument('--config_path')
    args = parser.parse_args()

    print(f"Input Path: {args.input_path}")
    print(f"Config Path: {args.config_path}")

    with open(args.input_path) as json_file:
        input_data = json.load(json_file)

    with open(args.config_path) as json_file:
        config = json.load(json_file)

    service = KBQARelationLinkingService(config)

    p_tot, r_tot, f1_tot = 0.0, 0.0, 0.0
    q_count = 0

    for q_id in list(input_data.keys()):

        # if q_id != 'train_53':
        #     continue

        question = input_data[q_id]

        question_text = question['text']
        eamr = question['extended_amr']
        gold_relations = question['relations']
        if len(gold_relations) == 0:
            continue

        print("QID: {}".format(q_id))
        predicted_relations = service.process(question_text, eamr)

        print("SPARQL: {}".format(question["sparql"]))
        print("gold: {}".format(", ".join(gold_relations)))
        print("predicted: {}\n".format(", ".join(predicted_relations)))

        p, r, f1 = precision_recall_f1(predicted_relations, gold_relations)
        print('---------------------------------------')
        print("QID: {}\nQuestion: {}\n".format(q_id, question_text))
        print("P: {}, R: {}, F1: {}".format(p, r, f1))
        print('---------------------------------------\n\n')

        p_tot += p
        r_tot += r
        f1_tot += f1
        q_count += 1

        print('Global: {} questions'.format(q_count))
        print("P: {}, R: {}, F1: {}".format(p_tot/q_count, r_tot/q_count, f1_score(p_tot/q_count, r_tot/q_count)))
        print('---------------------------------------\n\n')

    p_tot /= q_count
    r_tot /= q_count
    f1_tot = f1_score(p_tot, r_tot)

    print("\n\n\nFinal results:\n\t# of Qs: {}\t\nPrecision: {}\n\tRecall: {}\n\tF1: {}".format(q_count, p_tot, r_tot,
                                                                                                f1_tot))



