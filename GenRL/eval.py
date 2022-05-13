import argparse
import json


def get_p_r_f1(gold, pred):
    if len(pred) == 0:
        return 0, 0, 0
    gold = set(gold)
    pred = set(pred)
    p = len(gold.intersection(pred)) / len(pred)
    r = len(gold.intersection(pred)) / len(gold)
    if p + r > 0:
        f1 = 2 * ((p * r) / (p + r))
    else:
        f1 = 0
    return p, r, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_output')

    args = parser.parse_args()
    validation_output_path = args.val_output
    print(f"Evaluating file: {validation_output_path}")

    total_q, total_p, total_r, total_f1 = 0, 0, 0, 0
    for i, line in enumerate(open(validation_output_path)):
        ques_data = json.loads(line)
        gold_rels = ques_data['gold_rels']
        pred_rels = ques_data['pred_rels']
        p, r, f1 = get_p_r_f1(gold_rels, pred_rels)
        total_p += p
        total_r += r
        total_f1 += f1
        total_q += 1

    print(f"Evaluation:\n  P: {total_p /total_q}\n  R: {total_r /total_q}\n  F1: {total_f1 / total_q}")


if __name__ == "__main__":
    main()