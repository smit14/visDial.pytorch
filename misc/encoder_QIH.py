import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb
import math
import numpy as np
import torch.nn.functional as F
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

class _netE(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ninp, nhid, nlayers, dropout, img_feat_size):
        super(_netE, self).__init__()

        self.d = dropout
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.ninp = ninp
        self.img_embed = nn.Linear(img_feat_size, nhid).cuda()
        self.bert_feat_size = 768

        self.ques_rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout).cuda()
        self.his_rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout).cuda()

        # Bert model
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()
        # Bert feature size: 768

        self.Wb2q = nn.Linear(self.bert_feat_size, self.nhid).cuda()
        self.Wb2h = nn.Linear(self.bert_feat_size, self.nhid).cuda()

        self.Wb2qc = nn.Linear(self.bert_feat_size, self.nhid).cuda()
        self.Wb2hc = nn.Linear(self.bert_feat_size, self.nhid).cuda()

        self.Wq_1 = nn.Linear(self.nhid, self.nhid).cuda()
        self.Wh_1 = nn.Linear(self.nhid, self.nhid).cuda()
        self.Wa_1 = nn.Linear(self.nhid, 1).cuda()

        self.Wq_2 = nn.Linear(self.nhid, self.nhid).cuda()
        self.Wh_2 = nn.Linear(self.nhid, self.nhid).cuda()
        self.Wi_2 = nn.Linear(self.nhid, self.nhid).cuda()
        self.Wa_2 = nn.Linear(self.nhid, 1).cuda()

        self.fc1 = nn.Linear(self.nhid*3, self.ninp).cuda()

    def forward(self, ques_tokens_tensor, ques_segments_tensor, question_attention_mask, \
                his_tokens_tensor, his_segments_tensor, his_attention_mask, img_raw, rnd):

        img_emb = F.tanh(self.img_embed(img_raw))

        with torch.no_grad():
            ques_feat, _ = self.model(ques_tokens_tensor,ques_segments_tensor,question_attention_mask)
        ques_feat = torch.mean(ques_feat[11],1)
        ques_c = self.Wb2qc(ques_feat)
        ques_feat = self.Wb2q(ques_feat)
        ques_hidden = (ques_feat, ques_c)

        with torch.no_grad():
            his_feat, _ = self.model(his_tokens_tensor, his_segments_tensor,his_attention_mask)
        his_feat = torch.mean(his_feat[11],1)
        his_c = self.Wb2hc(his_feat)
        his_feat = self.Wb2h(his_feat)
        his_hidden = (his_feat, his_c)

        ques_emb_1 = self.Wq_1(ques_feat).view(-1, 1, self.nhid)
        his_emb_1 = self.Wh_1(his_feat).view(-1, rnd, self.nhid)

        atten_emb_1 = F.tanh(his_emb_1 + ques_emb_1.expand_as(his_emb_1))
        his_atten_weight = F.softmax(self.Wa_1(F.dropout(atten_emb_1, self.d, training=self.training
                                                ).view(-1, self.nhid)).view(-1, rnd))

        his_attn_feat = torch.bmm(his_atten_weight.view(-1, 1, rnd),
                                        his_feat.view(-1, rnd, self.nhid))

        his_attn_feat = his_attn_feat.view(-1, self.nhid)
        ques_emb_2 = self.Wq_2(ques_feat).view(-1, 1, self.nhid)
        his_emb_2 = self.Wh_2(his_attn_feat).view(-1, 1, self.nhid)
        img_emb_2 = self.Wi_2(img_emb).view(-1, 49, self.nhid)

        atten_emb_2 = F.tanh(img_emb_2 + ques_emb_2.expand_as(img_emb_2) + \
                                    his_emb_2.expand_as(img_emb_2))

        img_atten_weight = F.softmax(self.Wa_2(F.dropout(atten_emb_2, self.d, training=self.training
                                                ).view(-1, self.nhid)).view(-1, 49))

        img_attn_feat = torch.bmm(img_atten_weight.view(-1, 1, 49),
                                        img_emb.view(-1, 49, self.nhid))

        concat_feat = torch.cat((ques_feat, his_attn_feat.view(-1, self.nhid), \
                                 img_attn_feat.view(-1, self.nhid)),1)

        encoder_feat = F.tanh(self.fc1(F.dropout(concat_feat, self.d, training=self.training)))

        return encoder_feat, ques_hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()).cuda(),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()).cuda())
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
