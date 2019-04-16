from constants import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(2018)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrainedWordEmbeddingLayer(nn.Module):

    def __init__(self, trained_we_tensor, d_word):
        super(TrainedWordEmbeddingLayer, self).__init__()
        self.we = nn.Embedding.from_pretrained(trained_we_tensor)
        self.d_word = d_word

    def forward(self, spans):
        outputs = torch.zeros([len(spans), self.d_word], dtype=torch.float).to(device)
        for i, span in enumerate(spans):
            if len(span) > 0:
                outputs[i] = torch.sum(self.we(torch.LongTensor(span).to(device)), 0)
        return outputs
    
class NounAttentionLayer_SingleQuery(nn.Module):

    def __init__(self, trained_we_tensor, d_word, d_noun_hidden, d_desc):
        super(NounAttentionLayer_SingleQuery, self).__init__()
        self.we = nn.Embedding.from_pretrained(trained_we_tensor).to(device)
        self.d_word = d_word
        self.d_noun_hidden = d_noun_hidden
        self.d_desc = 1
        self.linear_key = nn.Linear(d_noun_hidden, d_noun_hidden, bias=False)
        nn.init.xavier_uniform_(self.linear_key.weight)
        self.query = nn.Embedding(1, d_noun_hidden)

    def forward(self, noun_spans, months, month_info_encode=1):
        outputs = torch.zeros([len(noun_spans), self.d_noun_hidden * self.d_desc], dtype=torch.float).to(device)
        for i, (noun_span, month) in enumerate(zip(noun_spans, months)):
            if len(noun_span) > 0:
                noun_part_mat = self.we(torch.LongTensor(noun_span).to(device))
                month_part_mat = torch.zeros([len(noun_part_mat), self.d_noun_hidden - self.d_word],
                                             dtype=torch.float).to(device)
                for j in range(len(month_part_mat)):
                    month_part_mat[j][month] = month_info_encode

                hidden_mat = torch.cat((noun_part_mat, month_part_mat), 1)

                key_mat = self.linear_key(hidden_mat)
                key_mat = torch.tanh(key_mat)

                query_mat = self.query(torch.LongTensor([j for j in range(self.d_desc)]).to(device))

                alpha_mat = torch.mm(query_mat, torch.t(key_mat))
                alpha_mat = F.softmax(alpha_mat, dim=1)

                result_mat = torch.mm(alpha_mat, hidden_mat)
                result_cat = result_mat.view(self.d_noun_hidden * self.d_desc)

                outputs[i] = result_cat

        return outputs

class MixingLayer(nn.Module):

    def __init__(self, d_word, d_ent, d_meta, d_mix):
        super(MixingLayer, self).__init__()
        self.linear_word = nn.Linear(d_word, d_mix)
        self.linear_ent = nn.Linear(d_ent, d_mix, bias=False)
        self.linear_meta = nn.Linear(d_meta, d_mix, bias=False)
        self.func = F.relu
        nn.init.xavier_uniform_(self.linear_word.weight)
        nn.init.constant_(self.linear_word.bias, 0.)
        nn.init.xavier_uniform_(self.linear_ent.weight)
        nn.init.xavier_uniform_(self.linear_meta.weight)

    def forward(self, input_word, input_ent1, input_ent2, input_meta):
        return self.func(self.linear_word(input_word) + self.linear_ent(input_ent1 + input_ent2)
                         + self.linear_meta(input_meta))
    
class MixingLayer_Attention_SingleQuery_Concat(nn.Module):

    def __init__(self, d_word, d_noun_hidden, d_ent, d_mix):
        super(MixingLayer_Attention_SingleQuery_Concat, self).__init__()
        self.linear_cat = nn.Linear(d_word + d_noun_hidden + d_ent, d_mix)
        self.func = F.relu
        nn.init.xavier_uniform_(self.linear_cat.weight)
        nn.init.constant_(self.linear_cat.bias, 0.)
        self.d_noun_hidden = d_noun_hidden

    def forward(self, input_word, input_noun_cat, input_ent1, input_ent2):
        noun_mats = input_noun_cat.view([len(input_noun_cat), 1, self.d_noun_hidden])
        from_noun_mats = []
        from_noun_mats.append(torch.index_select(noun_mats, 1, torch.LongTensor([0]).to(device)).view([len(input_noun_cat), self.d_noun_hidden]))
        from_nouns = torch.cat(from_noun_mats, dim=1)
        cat_inputs = torch.cat([input_word, from_nouns, input_ent1 + input_ent2], dim=1)
        return self.func(self.linear_cat(cat_inputs))

class LinearRNN(nn.Module):

    def __init__(self, d_input, d_hidden):
        super(LinearRNN, self).__init__()
        self.d_hidden = d_hidden
        self.begin = True
        self.hidden = torch.zeros([self.d_hidden], dtype=torch.float).to(device)
        self.linear_input = nn.Linear(d_input, d_hidden, bias=False)
        self.linear_hidden = nn.Linear(d_hidden, d_hidden, bias=False)
        self.func = F.softmax
        self.alpha = 0.5 # inherited from RMN

    def forward(self, inp, hid):
        from_inp = self.linear_input(inp)
        if self.begin:
            output = self.func(from_inp, dim=0)
            self.begin = False
        else:
            from_hid = self.linear_hidden(hid)
            output = self.func(from_inp + from_hid, dim=0)
            output = output * self.alpha + hid * (1 - self.alpha)
        return output, output
    
class DistributionLayer(nn.Module):

    def __init__(self, d_input, d_hidden):
        super(DistributionLayer, self).__init__()
        self.linear_input = nn.Linear(d_input, d_hidden, bias=False)
        nn.init.xavier_uniform_(self.linear_input.weight)
        self.func = F.softmax

    def forward(self, inps):
        from_inps = self.linear_input(inps)
        outputs = self.func(from_inps, dim=1)
        return outputs

class Contrastive_Max_Margin_Loss(nn.Module):

    def __init__(self):
        super(Contrastive_Max_Margin_Loss, self).__init__()

    def forward(self, outputs, pos_labels, neg_labels, traj_length, R, eps, d_word):
        norm_outputs = outputs / torch.norm(outputs, 2, 1, True)
        nan_masks = torch.isnan(norm_outputs)
        for i in range(len(nan_masks)):
            if torch.sum(nan_masks[i]) > 0:
                norm_outputs[i] = torch.zeros([d_word], dtype=torch.float).to(device)

        norm_pos_labels = pos_labels / torch.norm(pos_labels, 2, 1, True)
        nan_masks = torch.isnan(norm_pos_labels)
        for i in range(len(nan_masks)):
            if torch.sum(nan_masks[i]) > 0:
                norm_pos_labels[i] = torch.zeros([d_word], dtype=torch.float).to(device)

        norm_neg_labels = neg_labels / torch.norm(neg_labels, 2, 1, True)
        nan_masks = torch.isnan(norm_neg_labels)
        for i in range(len(nan_masks)):
            if torch.sum(nan_masks[i]) > 0:
                norm_neg_labels[i] = torch.zeros([d_word], dtype=torch.float).to(device)

        correct = torch.sum(norm_outputs * norm_pos_labels, 1, True)
        wrong = torch.mm(norm_outputs, torch.t(norm_neg_labels))

        loss = torch.sum(torch.max(torch.zeros(traj_length).to(device),
                                   torch.sum(1. - correct + wrong, 1)))

        norm_R = R / torch.norm(R, 2, 1, True)
        ortho_p = eps * torch.sum((torch.mm(norm_R, torch.t(norm_R)) - torch.eye(norm_R.shape[0]).to(device)) ** 2)

        loss += ortho_p.to(device)

        return loss
