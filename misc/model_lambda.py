import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb
import math
import numpy as np
import torch.nn.functional as F
from misc.share_Linear import share_Linear


class _netW(nn.Module):
    def __init__(self, ntoken, ninp, dropout):
        super(_netW, self).__init__()
        self.word_embed = nn.Embedding(ntoken + 1, ninp).cuda()
        self.Linear = share_Linear(self.word_embed.weight).cuda()
        self.init_weights()
        self.d = dropout

    def init_weights(self):
        initrange = 0.1
        self.word_embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, format='index'):
        if format == 'onehot':
            out = F.dropout(self.Linear(input), self.d, training=self.training)
        elif format == 'index':
            out = F.dropout(self.word_embed(input), self.d, training=self.training)

        return out

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())


class _netD(nn.Module):
    """
    Given the real/wrong/fake answer, use a RNN (LSTM) to embed the answer.
    """

    def __init__(self, rnn_type, ninp, nhid, nlayers, ntoken, dropout):
        super(_netD, self).__init__()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.ntoken = ntoken
        self.ninp = ninp
        self.d = dropout

        self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers)
        self.W1 = nn.Linear(self.nhid, self.nhid)
        self.W2 = nn.Linear(self.nhid, 1)
        self.fc = nn.Linear(nhid, ninp)

    def forward(self, input_feat, idx, hidden, vocab_size):

        output, _ = self.rnn(input_feat, hidden)
        mask = idx.data.eq(0)  # generate the mask
        mask[idx.data == vocab_size] = 1  # also set the last token to be 1
        if isinstance(input_feat, Variable):
            mask = Variable(mask, volatile=input_feat.volatile)

        # Doing self attention here.
        atten = self.W2(F.dropout(F.tanh(self.W1(output.view(-1, self.nhid))), self.d, training=self.training)).view(
            idx.size())
        atten.masked_fill_(mask, -99999)
        weight = F.softmax(atten.t()).view(-1, 1, idx.size(0))
        feat = torch.bmm(weight, output.transpose(0, 1)).view(-1, self.nhid)
        feat = F.dropout(feat, self.d, training=self.training)
        transform_output = F.tanh(self.fc(feat))

        return transform_output

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())


class LMCriterion(nn.Module):

    def __init__(self):
        super(LMCriterion, self).__init__()

    def forward(self, input, target):
        logprob_select = torch.gather(input, 1, target)

        mask = target.data.gt(0)  # generate the mask
        if isinstance(input, Variable):
            mask = Variable(mask, volatile=input.volatile)

        out = torch.masked_select(logprob_select, mask)

        loss = -torch.sum(out)  # get the average loss.
        return loss


class mixture_of_softmaxes(torch.nn.Module):
    """
    Breaking the Softmax Bottleneck: A High-Rank RNN Language Model (ICLR 2018)
    """

    def __init__(self, nhid, n_experts, ntoken):
        super(mixture_of_softmaxes, self).__init__()

        self.nhid = nhid
        self.ntoken = ntoken
        self.n_experts = n_experts

        self.prior = nn.Linear(nhid, n_experts, bias=False)
        self.latent = nn.Sequential(nn.Linear(nhid, n_experts * nhid), nn.Tanh())
        self.decoder = nn.Linear(nhid, ntoken)

    def forward(self, x):
        latent = self.latent(x)
        logit = self.decoder(latent.view(-1, self.nhid))

        prior_logit = self.prior(x).view(-1, self.n_experts)
        prior = nn.functional.softmax(prior_logit)

        prob = nn.functional.softmax(logit.view(-1, self.ntoken)).view(-1, self.n_experts, self.ntoken)
        prob = (prob * prior.unsqueeze(2).expand_as(prob)).sum(1)

        return prob


class nPairLoss(nn.Module):
    """
    Given the right, fake, wrong, wrong_sampled embedding, use the N Pair Loss
    objective (which is an extension to the triplet loss)

    Loss = log(1+exp(feat*wrong - feat*right + feat*fake - feat*right)) + L2 norm.

    Improved Deep Metric Learning with Multi-class N-pair Loss Objective (NIPS)
    """

    def __init__(self, ninp, margin):
        super(nPairLoss, self).__init__()
        self.ninp = ninp
        self.margin = np.log(margin)

    def forward(self, combined, negative_samples, batch_negative_samples):

        batch_size = combined.size(0)
        total_ans = combined.size(1)

        #rank
        (sorted_score, sorted_idx) = combined.sort(dim=1,descending=True)

        assignment = torch.arange(total_ans).reshape(1, total_ans).expand(batch_size, total_ans).cuda()
        row_idx = torch.arange(batch_size).reshape(batch_size, 1).expand(batch_size, total_ans).reshape(
            total_ans * batch_size).cuda()
        col_idx = sorted_idx.reshape(-1)
        rank = torch.LongTensor(batch_size * total_ans).cuda()
        flattened_idx = row_idx * total_ans + col_idx
        rank.put_(flattened_idx, 1 + assignment.reshape(-1))
        rank = rank.reshape(batch_size, total_ans)

        score_diff = combined[:,:1] - combined[:,1:].view(batch_size,total_ans-1)
        exped = score_diff.exp()
        mrr_dif = 1/rank[:,:1].float() - 1/rank[:,1:].view(batch_size,total_ans-1).float()

        # lamb updates
        lamb_updates = (-1/(1 + exped)) * mrr_dif.abs()
        lambs1 = lamb_updates.sum(dim=1, keepdim=True)
        lambs_rest = -1*lamb_updates
        lambs = torch.cat((lambs1,lambs_rest),dim=1)
        lambs = lambs / batch_size      # dividing lambs loss by batch size because actual lamb loss will be summed over while backpropogating
       # -------------------------------------------------------------------------------------------------
        right_dis = combined[:,:1]
        wrong_dis = combined[:,1:negative_samples+1]
        batch_wrong_dis = combined[:,negative_samples+1:]

        wrong_score = torch.sum(torch.exp(wrong_dis - right_dis.expand_as(wrong_dis)), 1) \
                      + torch.sum(torch.exp(batch_wrong_dis - right_dis.expand_as(batch_wrong_dis)), 1)

        loss_dis = torch.sum(torch.log(wrong_score + 1))

        loss = loss_dis / batch_size
        return lambs, loss

class feat_to_score_and_norm(nn.Module):
    def __init__(self,ninp, margin):
        super(feat_to_score_and_norm, self).__init__()
        self.ninp = ninp
        self.margin = np.log(margin)

    def forward(self, feat, right, wrong, batch_wrong):
        num_wrong = wrong.size(1)
        batch_size = feat.size(0)
        total_ans = 1 + num_wrong + batch_wrong.size(1)

        feat = feat.view(-1, self.ninp, 1)  # (batch_size, ninp, 1)
        right_dis = torch.bmm(right.view(-1, 1, self.ninp), feat)
        wrong_dis = torch.bmm(wrong, feat)
        batch_wrong_dis = torch.bmm(batch_wrong, feat)

        # combined score
        combined = torch.cat((right_dis, wrong_dis, batch_wrong_dis),dim=1)  # (batch_size, 1 + n_s + b_n_s, 1) # n_s : negatives samples , b_n_s: batch_negative_samples
        loss_norm = right.norm() + feat.norm() + wrong.norm() + batch_wrong.norm()
        loss_norm = loss_norm / batch_size
        return combined.squeeze(), loss_norm

class G_loss(nn.Module):
    """
    Generator loss:
    minimize right feature and fake feature L2 norm.
    maximinze the fake feature and wrong feature.
    """

    def __init__(self, ninp):
        super(G_loss, self).__init__()
        self.ninp = ninp

    def forward(self, feat, right, fake):
        # num_wrong = wrong.size(1)
        batch_size = feat.size(0)

        feat = feat.view(-1, self.ninp, 1)
        # wrong_dis = torch.bmm(wrong, feat)
        # batch_wrong_dis = torch.bmm(batch_wrong, feat)
        fake_dis = torch.bmm(fake.view(-1, 1, self.ninp), feat)
        right_dis = torch.bmm(right.view(-1, 1, self.ninp), feat)

        fake_score = torch.exp(right_dis - fake_dis)
        loss_fake = torch.sum(torch.log(fake_score + 1))

        loss_norm = feat.norm() + fake.norm() + right.norm()
        loss = (loss_fake + 0.1 * loss_norm) / batch_size

        return loss, loss_fake.data[0] / batch_size


class gumbel_sampler(nn.Module):
    def __init__(self):
        super(gumbel_sampler, self).__init__()

    def forward(self, input, noise, temperature=0.5):
        eps = 1e-20
        noise.data.add_(eps).log_().neg_()
        noise.data.add_(eps).log_().neg_()
        y = (input + noise) / temperature
        y = F.softmax(y)

        max_val, max_idx = torch.max(y, y.dim() - 1)
        y_hard = y == max_val.view(-1, 1).expand_as(y)
        y = (y_hard.float() - y).detach() + y

        # log_prob = input.gather(1, max_idx.view(-1,1)) # gather the logprobs at sampled positions

        return y, max_idx.view(1, -1)  # , log_prob


class AxB(nn.Module):
    def __init__(self, nhid):
        super(AxB, self).__init__()
        self.nhid = nhid

    def forward(self, nhA, nhB):
        mat = torch.bmm(nhB.view(-1, 100, self.nhid), nhA.view(-1, self.nhid, 1))
        return mat.view(-1, 100)
