import json
import torch
from collections import Counter

from opennre.encoder.bert_encoder import BERTEntityEncoder
from opennre.model.ranking_nn import RankingNN

from relation_linking_core.rel_linker_modules.rel_linker_module import RelModule


class NeuralRelationLinking(RelModule):

    question_terms = ['what', 'when', 'which', 'who', 'how', 'list', 'give', 'show', 'do', 'does']

    def __init__(self, config):
        rel_id_path = "/Users/nandana.sampath.mihindukulasooriya@ibm.com/Src/relation-linking/data/lcquad/rel2id.json"
        rel2id = json.load(open(rel_id_path))
        pretrain_path = "/Users/nandana.sampath.mihindukulasooriya@ibm.com/Src/relation-linking/data/bert-base-uncased"
        ckpt_path = "/Users/nandana.sampath.mihindukulasooriya@ibm.com/Src/relation-linking/data/lcquad/nre4qald_v4_10_bertentity_softmax.pth.tar"

        sentence_encoder = BERTEntityEncoder(max_length=80, pretrain_path=pretrain_path)

        print("Loading neural model ...\n\trel2id path: {}\n\trels: {}\n\tpretrain_path: {}\n\tckpt: {}".
              format(rel_id_path, len(rel2id), pretrain_path, ckpt_path))

        self.neural_model = RankingNN(sentence_encoder, len(rel2id), rel2id)

        if torch.cuda.is_available():
            self.neural_model.load_state_dict(torch.load(ckpt_path, map_location='cuda')['state_dict'])
            self.neural_model.cuda()
        else:
            self.neural_model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['state_dict'])

        with torch.no_grad():
            self.neural_model.r_hiddens = self.neural_model.forward_all_relations()

    def get_relation_candidates(self, triple_data, params=None):

        relation_scores = Counter()

        normalized_to_surface_form = params['normalized_to_surface_form']
        subj_text, subj_type = triple_data['subj_text'], triple_data['subj_type']
        obj_text, obj_type = triple_data['obj_text'], triple_data['obj_type']

        head = subj_text if subj_text else subj_type
        tail = obj_text if obj_text else obj_type

        if triple_data['amr_unknown_var'] == triple_data['subj_id'] or head == 'amr-unknown' or head == 'unknown':
            amr_unkown = head
        elif triple_data['amr_unknown_var'] == triple_data['obj_id'] or tail == 'amr-unknown' or tail == 'unknown':
            amr_unkown = tail
        else:
            amr_unkown = None

        print("OpenNRE:\n\thead: {}\n\ttail: {}\n\tamr-unknown: {}".format(head, tail, amr_unkown))
        input = NeuralRelationLinking.prepare_opennre_input(triple_data['text'], head, tail, normalized_to_surface_form,
                                                  amr_unkown)
        if input:
            openre_response = self.neural_model.infer_ranking(input)
            opennre_relations = [(rel[0], rel[1]) for rel in openre_response[:10]]
            for rel in opennre_relations:
                relation_scores[rel[0]] += rel[1]
            print("\topennre relations: {}".format(opennre_relations))
        else:
            print("\topennre input error:\n\tsent: {}\n\th: {} \t: {}".format(triple_data['text'], head, tail))

        return relation_scores

    @classmethod
    def prepare_opennre_input(cls, sentence, head, tail, normalized_to_surface_form, amr_unkown):
        sentence = sentence.lower()
        head_start, tail_start = sentence.find(head.lower()), sentence.find(tail.lower())

        if head_start == -1:
            if head in normalized_to_surface_form:
                head_start = sentence.find(normalized_to_surface_form[head].lower())
            if head_start == -1 and amr_unkown == head:
                for term in NeuralRelationLinking.question_terms:
                    head_start = sentence.find(term)
                    if head_start != -1:
                        head = term
                        break
        if tail_start == -1:
            if tail in normalized_to_surface_form:
                tail_start = sentence.find(normalized_to_surface_form[tail].lower())
            if tail_start == -1 and amr_unkown == tail:
                for term in NeuralRelationLinking.question_terms:
                    tail_start = sentence.find(term)
                    if tail_start != -1:
                        tail = term
                        break

        if head_start == -1 or tail_start == -1:
            return
        else:
            head_end, tail_end = head_start + len(head) + 1, tail_start + len(tail) + 1

        return {"text": sentence, "h": {"pos": (head_start, head_end)}, "t": {"pos": (tail_start, tail_end)}}




