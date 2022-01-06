#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch import nn, optim
import numpy as np
from .base_model import SentenceRE


# In[ ]:


def process_relation_name(relation):

    rel_info = relation.split(':')

    rel_name = rel_info[1]
    rel_words = []
    start = 0
    for i in range(len(rel_name)):
        if (rel_name[i] >= 'A' and rel_name[i] <= 'Z'):
            rel_words.append(rel_name[start:i])
            start = i
        elif rel_name[i] == '/':
            rel_words.append(rel_name[start:i])
            start = i + 1

    rel_words.append(rel_name[start:])

    return ' '.join(rel_words).lower()


# In[ ]:


class RankingNN(SentenceRE):
    """
    Softmax classifier for sentence-level relation extraction.
    """

    def __init__(self, sentence_encoder, num_class, rel2id, use_entity_pos=True):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        
        self.softmax = nn.Softmax(-1)
        self.rel2id = rel2id
        self.id2rel = {}
        self.drop = nn.Dropout()
        for rel, id in rel2id.items():
            self.id2rel[id] = rel
        
        self.relid2name = {}
        self.CLS_TOKEN = 101
        self.SEP_TOKEN = 102
        
        self.r_hiddens = None
        
        self.zero_position = torch.tensor([[0]]).long()
        
        self.get_rel_words()
        
    def get_rel_words(self):
        for rel, rel_id in self.rel2id.items():
            rel_tokens = self.sentence_encoder.tokenizer.tokenize(process_relation_name(rel))
            self.relid2name[rel_id] = rel_tokens

    def infer(self, item):
        self.eval()
        item = self.sentence_encoder.tokenize(item)
        logits = self.forward(('emb', self.r_hiddens), *item)
        logits = self.softmax(logits)
        score, pred = logits.max(-1)
        score = score.item()
        pred = pred.item()
        return self.id2rel[pred], score

    def infer_ranking(self, item):
        self.eval()
        item = self.sentence_encoder.tokenize(item)
        new_item = []
        if torch.cuda.is_available():
            for i in range(len(item)):
                new_item.append(item[i].cuda())
        else:
            new_item = item

#         print(self.r_hiddens)
#         print(new_item)
        logits = self.forward(('emb', self.r_hiddens), *new_item)
        logits = self.softmax(logits)
        ranked_list = [(self.id2rel[i], x.item()) for i, x in enumerate(logits[0, :])]
        ranked_list = sorted(ranked_list, key=lambda x: x[1], reverse=True)
        return ranked_list
    
    def get_rws(self, r):
        """
        Args:
            r, a 2d-list of relation ids (B, R)
        Return:
            logits, (B, R)
        """
        
        rws = []

        max_rw_len = -1

        for i in range(r.shape[0]):
            tmp_rw = []
            for j in range(r.shape[1]):
                rid = r[i][j]

                tmp_rw.append(self.relid2name[rid])
            
                if len(tmp_rw[-1]) > max_rw_len:
                    max_rw_len = len(tmp_rw[-1])

            rws.append(tmp_rw)

        rw_masks_ = []
        for i, rws_ in enumerate(rws):
            rw_masks_.append([])
            for j, rw in enumerate(rws_):
                rws[i][j] = ['[CLS]'] + rw + ['[SEP]'] + (max_rw_len - len(rw)) * [0]
                rws[i][j] = self.sentence_encoder.tokenizer.convert_tokens_to_ids(rws[i][j])
                rw_masks_[i].append([1] * (len(rw) + 2) + [0] * (max_rw_len - len(rw)))
                
        rw_tensor = np.array(rws, dtype=np.int64)
        rw_mask = np.array(rw_masks_, dtype=np.int64)
        
        rw_tensor = torch.tensor(rw_tensor).long()
        rw_mask = torch.tensor(rw_mask).long()
        
        if torch.cuda.is_available():
            rw_tensor = rw_tensor.cuda()
            rw_mask = rw_mask.cuda()
        
        return rw_tensor, rw_mask
    
    def forward(self, r, *args):
        """
        Args:
            r = (str:identifier, matrix of relation), matrix of relation ids (B, R), R is the number of relations sampled for each sentence
            args: depends on the encoder
        Return:
            logits, (B, R)
        """
        
        if r[0] == 'emb' and self.training == False:
            # evaluation mode
            
#             # pre-compute the relation embeddings
#             if self.r_hiddens is None:
#                 print(self.r_hiddens)
#                 print('forward all relations ...')
#                 self.forward_all_relations()
#                 print(self.r_hiddens)
            r_hiddens = r[1]
#             print(args)
#             print(r_hiddens)
            q_hiddens = self.sentence_encoder(*args) # (B, H)
            logits = torch.mm(q_hiddens, r_hiddens.transpose(0, 1)) # (B, N)
            return logits
        elif r[0] == 'rid':
    #         _, q_hiddens = self.sentence_encoder(q, attention_mask=q_mask) # (B, H)
            q_hiddens = self.sentence_encoder(*args) # (B, H)

            rw, rw_mask = self.get_rws(r[1].cpu().numpy())

            rw_flat = rw.view(rw.size(0)*rw.size(1), rw.size(2)) # (B x R, Lr)
            rw_mask_flat = rw_mask.view(rw.size(0)*rw.size(1), rw.size(2)) # (B x R, Lr)

    #         print(q_hiddens.size())
    #         print(rw_flat.size())
    #         print(rw_mask_flat.size())
    #         print(self.zero_position.size())

            zero_position = torch.zeros(rw_flat.size(0), 1).long()

            rw_hiddens_ = self.sentence_encoder(rw_flat, rw_mask_flat, 
                                                zero_position, zero_position) # (B x R, H)
            rw_hiddens = rw_hiddens_.view(rw.size(0), rw.size(1), -1) # (B, R, H)

    #         q_hiddens = self.drop(q_hiddens)
    #         rw_hiddens = self.drop(rw_hiddens)

            logits = torch.bmm(rw_hiddens, q_hiddens.unsqueeze(-1)).squeeze(-1) # (B, R)
            return logits
        else:
            print('Error')
            return None

    def forward_all_relations(self):
        rs = np.array(list(range(len(self.id2rel))), dtype=np.int64)
#         rs = torch.tensor(rs).long().cuda().unsqueeze(-1)
        rs = np.expand_dims(rs, axis=-1)
        rw, rw_mask = self.get_rws(rs)
        
        zero_position = torch.zeros(rw.size(0), 1).long()
        
        rw = rw.squeeze(1)
        rw_mask = rw_mask.squeeze(1)
#         print(rw.size())
#         print(zero_position.size())
        
        r_hiddens = self.sentence_encoder(rw, rw_mask, zero_position, zero_position) # (N, H)
        r_hiddens = r_hiddens.data
        return r_hiddens
    
    def forward_relations(self, r, rw, r_mask):
        """
        Args:

        Return:

        """
        
        _, r_hiddens = self.bert(rw, attention_mask=r_mask) # (N, H)
        r_hiddens = r_hiddens.data
        return r_hiddens
    
#     def predict(self, *args):
#         _, q_hiddens = self.bert(q, attention_mask=q_mask) # (B, H)
#         logits = torch.mm(q_hiddens, self.r_hiddens.transpose(0, 1)) # (B, N)
#         return logits
    


# In[ ]:


def test(x, *args):
    print(x)
    print(args)
    
# test(0,1,2)


# In[ ]:




