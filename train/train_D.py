from __future__ import print_function

import argparse
import os
import random
import sys
sys.path.append(os.getcwd())

import pdb
import time
import numpy as np
import json
import progressbar
import sys
# sys.path.insert(1, '/home/hitarth/gpu/visDial.pytorch')


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable


# from misc.utils import repackage_hidden, repackage_hidden_new, clip_gradient, adjust_learning_rate, \
#                     decode_txt, sample_batch_neg, l2_norm
#
# import misc.dataLoader as dl
# import misc.model as model
# from misc.encoder_QIH import _netE
import datetime

parser = argparse.ArgumentParser()

parser.add_argument('--input_img_h5', default='../script/data/vdl_img_vgg.h5', help='path to image feature, now hdf5 file')
parser.add_argument('--input_ques_h5', default='../script/data/visdial_data.h5', help='path to label, now hdf5 file')
parser.add_argument('--input_json', default='../script/data/visdial_params.json', help='path to dataset, now json file')
parser.add_argument('--input_probs_h5', default='../script/data/visdial_data_prob.h5')
parser.add_argument('--outf', default='./save', help='folder to output model checkpoints')
parser.add_argument('--decoder', default='D', help='what decoder to use.')
parser.add_argument('--model_path', default='', help='folder to output images and model checkpoints')
parser.add_argument('--num_val', type=int, default=1000, help='number of image split out as validation set.')

parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--negative_sample', type=int, default=20, help='folder to output images and model checkpoints')
parser.add_argument('--neg_batch_sample', type=int, default=0, help='folder to output images and model checkpoints')
parser.add_argument('--start_epoch', type=int, default=1, help='start of epochs to train for')
parser.add_argument('--teacher_forcing', type=int, default=1, help='start of epochs to train for')

parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--save_iter', type=int, default=5, help='number of epochs to train for')

parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--lr', type=float, default=0.0004, help='learning rate for, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.8, help='beta1 for adam. default=0.5')
parser.add_argument('--alpha_norm', type=float, default=0.1)
parser.add_argument('--sigma', type=float, default=1.0)
parser.add_argument('--alphaC', type=float, default=2.0)
parser.add_argument('--alphaE', type=float, default=0.1)
parser.add_argument('--alphaN', type=float, default=1.0)
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--debug', action='store_true', help='prints stuff in debug mode')
parser.add_argument('--sample_dist', action='store_true', help='Used for seeing distribution of C,E,N in samples')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--verbose'  , action='store_true', help='show the sampled caption')

parser.add_argument('--conv_feat_size', type=int, default=512, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=512, help='input batch size')
parser.add_argument('--model', type=str, default='LSTM', help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--ninp', type=int, default=300, help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=512, help='humber of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1, help='number of layers')
parser.add_argument('--dropout', type=int, default=0.5, help='number of layers')
parser.add_argument('--clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--margin', type=float, default=2, help='number of epochs to train for')
parser.add_argument('--log_interval', type=int, default=5, help='how many iterations show the log info')
parser.add_argument('--path_to_home',type=str)
parser.add_argument('--exp_name', type=str, help='name of the expemriment')
parser.add_argument('--early_stop', type=int, default='1000000', help='datapoints to consider')

opt = parser.parse_args()
print(opt)
sys.path.insert(1, opt.path_to_home)

from misc.utils import repackage_hidden, repackage_hidden_new, clip_gradient, adjust_learning_rate, \
                    decode_txt, sample_batch_neg, l2_norm

import misc.dataLoader as dl
import misc.model as model
from misc.encoder_QIH import _netE
from script.test_data import check_data

# ---------------------- check for data correctnes -------------------------------------
if check_data() == False:
    print("data is not up-to-date")
    exit(255)

# ---------------------- ---------------------------------------------------------------



opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed(opt.manualSeed)

cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.model_path != '':
    print("=> loading checkpoint '{}'".format(opt.model_path))
    checkpoint = torch.load(opt.model_path)
    model_path = opt.model_path
    opt.start_epoch = checkpoint['epoch']
    # opt.model_path = model_path
    # opt.batchSize = 1
    opt.start_epoch = checkpoint['epoch']
    save_path = opt.save_path
else:
    # create new folder.
    t = datetime.datetime.now()
    if(opt.exp_name is None or opt.exp_name == ''):
        print('Model path is not provided. Cannot run experiment without providing --exp_name')
        exit(255)
    # cur_time = '%s-%s-%s' %(t.day, t.month, t.hour)
    save_path = os.path.join(opt.outf, opt.exp_name)
    opt.save_path = save_path
    try:
        os.makedirs(save_path)
    except OSError:
        pass

####################################################################################
# Data Loader
####################################################################################

dataset = dl.train(input_img_h5=opt.input_img_h5, input_ques_h5=opt.input_ques_h5,
                input_json=opt.input_json, negative_sample = opt.negative_sample,
                num_val = opt.num_val, data_split = 'train', input_probs=opt.input_probs_h5)

dataset_val = dl.validate(input_img_h5=opt.input_img_h5, input_ques_h5=opt.input_ques_h5,
                input_json=opt.input_json, negative_sample = opt.negative_sample,
                num_val = opt.num_val, data_split = 'val')

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=int(opt.workers))

dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1,
                                         shuffle=False, num_workers=int(opt.workers))

####################################################################################
# Build the Model
####################################################################################
n_neg = opt.negative_sample
vocab_size = dataset.vocab_size
ques_length = dataset.ques_length
ans_length = dataset.ans_length + 1
his_length = dataset.ans_length + dataset.ques_length
itow = dataset.itow
img_feat_size = 512

netE = _netE(opt.model, opt.ninp, opt.nhid, opt.nlayers, opt.dropout, img_feat_size)
netW = model._netW(vocab_size, opt.ninp, opt.dropout)
netD = model._netD(opt.model, opt.ninp, opt.nhid, opt.nlayers, vocab_size, opt.dropout)
critD =model.nPairLoss(opt.ninp, opt.margin, opt.alpha_norm, opt.sigma, opt.alphaC, opt.alphaE, opt.alphaN, opt.debug, opt.log_interval)

if opt.model_path != '': # load the pre-trained model.
    netW.load_state_dict(checkpoint['netW'])
    netE.load_state_dict(checkpoint['netE'])
    netD.load_state_dict(checkpoint['netD'])

if opt.cuda: # ship to cuda, if has GPU
    netW.cuda(), netE.cuda(),
    netD.cuda(), critD.cuda()

####################################################################################
# training model
####################################################################################
def train(epoch):
    netW.train()
    netE.train()
    netD.train()

    lr = adjust_learning_rate(optimizer, epoch, opt.lr)

    ques_hidden = netE.init_hidden(opt.batchSize)
    hist_hidden = netE.init_hidden(opt.batchSize)

    real_hidden = netD.init_hidden(opt.batchSize)
    wrong_hidden = netD.init_hidden(opt.batchSize)

    data_iter = iter(dataloader)

    average_loss = 0
    avg_dist_summary = np.zeros(3, dtype=float)
    smooth_avg_dist_summary = np.zeros(3, dtype=float)
    count = 0
    i = 0

    # size of data to work on
    early_stop = int(opt.early_stop / opt.batchSize)
    dataloader_size = min(len(dataloader), early_stop)
    while i < dataloader_size: #len(dataloader):

        t1 = time.time()
        data = data_iter.next()
        image, history, question, answer, answerT, answerLen, answerIdx, questionL, \
                                    opt_answerT, opt_answerLen, opt_answerIdx, opt_selected_probs = data

        batch_size = question.size(0)
        image = image.view(-1, img_feat_size)

        with torch.no_grad():
            img_input.resize_(image.size()).copy_(image)

        for rnd in range(10):
            netW.zero_grad()
            netE.zero_grad()
            netD.zero_grad()
            # get the corresponding round QA and history.
            ques = question[:,rnd,:].t()
            his = history[:,:rnd+1,:].clone().view(-1, his_length).t()

            ans = answer[:,rnd,:].t()
            tans = answerT[:,rnd,:].t()
            wrong_ans = opt_answerT[:,rnd,:].clone().view(-1, ans_length).t()
            opt_selected_probs_for_rnd = opt_selected_probs[:, rnd, :, :]

            real_len = answerLen[:,rnd]
            wrong_len = opt_answerLen[:,rnd,:].clone().view(-1)

            ques_input = torch.LongTensor(ques.size()).cuda()
            ques_input.copy_(ques)

            his_input = torch.LongTensor(his.size()).cuda()
            his_input.copy_(his)

            ans_input = torch.LongTensor(ans.size()).cuda()
            ans_input.copy_(ans)

            ans_target = torch.LongTensor(tans.size()).cuda()
            ans_target.copy_(tans)

            wrong_ans_input = torch.LongTensor(wrong_ans.size()).cuda()
            wrong_ans_input.copy_(wrong_ans)

            opt_selected_probs_for_rnd_input = torch.LongTensor(opt_selected_probs_for_rnd.size()).cuda()
            opt_selected_probs_for_rnd_input.copy_(opt_selected_probs_for_rnd)

            # # sample in-batch negative index
            # batch_sample_idx = torch.zeros(batch_size, opt.neg_batch_sample, dtype=torch.long).cuda()
            # sample_batch_neg(answerIdx[:,rnd], opt_answerIdx[:,rnd,:], batch_sample_idx, opt.neg_batch_sample)

            ques_emb = netW(ques_input, format = 'index')
            his_emb = netW(his_input, format = 'index')

            ques_hidden = repackage_hidden_new(ques_hidden, batch_size)
            hist_hidden = repackage_hidden_new(hist_hidden, his_input.size(1))

            featD, ques_hidden = netE(ques_emb, his_emb, img_input, \
                                                ques_hidden, hist_hidden, rnd+1)

            ans_real_emb = netW(ans_target, format='index')
            ans_wrong_emb = netW(wrong_ans_input, format='index')

            real_hidden = repackage_hidden_new(real_hidden, batch_size)
            wrong_hidden = repackage_hidden_new(wrong_hidden, ans_wrong_emb.size(1))

            real_feat = netD(ans_real_emb, ans_target, real_hidden, vocab_size)
            wrong_feat = netD(ans_wrong_emb, wrong_ans_input, wrong_hidden, vocab_size)

            # batch_wrong_feat = wrong_feat.index_select(0, batch_sample_idx.view(-1))
            wrong_feat = wrong_feat.view(batch_size, -1, opt.ninp)
            # batch_wrong_feat = batch_wrong_feat.view(batch_size, -1, opt.ninp)

            nPairLoss, dist_summary = critD(featD, real_feat, wrong_feat, opt_selected_probs_for_rnd_input)

            average_loss += nPairLoss.data.item()
            avg_dist_summary += dist_summary.cpu().detach().numpy()
            smooth_avg_dist_summary += smooth_avg_dist_summary.cpu().detach().numpy()
            nPairLoss.backward()
            optimizer.step()
            count += 1

        i += 1
        if i % opt.log_interval == 0:
            average_loss /= count
            avg_dist_summary = avg_dist_summary/np.sum(avg_dist_summary)
            smooth_avg_dist_summary = smooth_avg_dist_summary/np.sum(smooth_avg_dist_summary)
            print("step {} / {} (epoch {}), g_loss {:.3f}, lr = {:.6f}, CEN dist: {}, CEN smooth: {}"\
                .format(i, len(dataloader), epoch, average_loss, lr, avg_dist_summary, smooth_avg_dist_summary))
            average_loss = 0
            avg_dist_summary = np.zeros(3, dtype=float)
            smooth_avg_dist_summary = np.zeros(3, dtype=float)
            count = 0

    return average_loss, lr


def val():
    # ques_hidden = netE.init_hidden(opt.batchSize)

    netE.eval()
    netW.eval()
    netD.eval()

    n_neg = 100
    data_iter_val = iter(dataloader_val)
    ques_hidden = netE.init_hidden(opt.batchSize)
    print('DSuccess')
    hist_hidden = netE.init_hidden(opt.batchSize)

    opt_hidden = netD.init_hidden(opt.batchSize)
    i = 0

    average_loss = 0
    rank_all_tmp = []

    while i < len(dataloader_val):
        data = data_iter_val.next()
        image, history, question, answer, answerT, questionL, opt_answer, \
                opt_answerT, answer_ids, answerLen, opt_answerLen, img_id  = data

        batch_size = question.size(0)
        image = image.view(-1, img_feat_size)
        #image = l2_norm(image)
        with torch.no_grad():
            img_input.resize_(image.size()).copy_(image)

        for rnd in range(10):
            # get the corresponding round QA and history.
            ques = question[:,rnd,:].t()
            his = history[:,:rnd+1,:].clone().view(-1, his_length).t()

            opt_ans = opt_answerT[:,rnd,:].clone().view(-1, ans_length).t()
            gt_id = answer_ids[:,rnd]

            ques_input = torch.LongTensor(ques.size()).cuda()
            ques_input.copy_(ques)

            his_input = torch.LongTensor(his.size()).cuda()
            his_input.copy_(his)

            opt_ans_input = torch.LongTensor(opt_ans.size()).cuda()
            opt_ans_input.copy_(opt_ans)

            gt_index = torch.LongTensor(gt_id.size()).cuda()
            gt_index.copy_(gt_id)

            opt_len = opt_answerLen[:,rnd,:].clone().view(-1)

            ques_emb = netW(ques_input, format = 'index')
            his_emb = netW(his_input, format = 'index')

            ques_hidden = repackage_hidden_new(ques_hidden, batch_size)
            hist_hidden = repackage_hidden_new(hist_hidden, his_input.size(1))

            featD, ques_hidden = netE(ques_emb, his_emb, img_input, \
                                                ques_hidden, hist_hidden, rnd+1)

            opt_ans_emb = netW(opt_ans_input, format = 'index')
            opt_hidden = repackage_hidden_new(opt_hidden, opt_ans_input.size(1))
            opt_feat = netD(opt_ans_emb, opt_ans_input, opt_hidden, vocab_size)
            opt_feat = opt_feat.view(batch_size, -1, opt.ninp)

            #ans_emb = ans_emb.view(ans_length, -1, 100, opt.nhid)
            featD = featD.view(-1, opt.ninp, 1)
            score = torch.bmm(opt_feat, featD)
            score = score.view(-1, 100)

            for b in range(batch_size):
                gt_index.data[b] = gt_index.data[b] + b*100

            gt_score = score.view(-1).index_select(0, gt_index)
            sort_score, sort_idx = torch.sort(score, 1, descending=True)

            count = sort_score.gt(gt_score.view(-1,1).expand_as(sort_score))
            rank = count.sum(1) + 1
            rank_all_tmp += list(rank.view(-1).data.cpu().numpy())
            
        i += 1
        sys.stdout.write('Evaluating: {:d}/{:d}  \r' \
          .format(i, len(dataloader_val)))
        sys.stdout.flush()

    return rank_all_tmp


####################################################################################
# Main
####################################################################################
img_input = torch.FloatTensor(opt.batchSize)
ques_input = torch.LongTensor(ques_length, opt.batchSize)
his_input = torch.LongTensor(his_length, opt.batchSize)

# answer input
ans_input = torch.LongTensor(ans_length, opt.batchSize)
ans_target = torch.LongTensor(ans_length, opt.batchSize)
wrong_ans_input = torch.LongTensor(ans_length, opt.batchSize)
sample_ans_input = torch.LongTensor(1, opt.batchSize)
opt_ans_input = torch.LongTensor(ans_length, opt.batchSize)

batch_sample_idx = torch.LongTensor(opt.batchSize)
fake_diff_mask = torch.ByteTensor(opt.batchSize)
fake_len = torch.LongTensor(opt.batchSize)
noise_input = torch.FloatTensor(opt.batchSize)
gt_index = torch.LongTensor(opt.batchSize)


if opt.cuda:
    ques_input, his_input, img_input = ques_input.cuda(), his_input.cuda(), img_input.cuda()
    ans_input, ans_target = ans_input.cuda(), ans_target.cuda()
    wrong_ans_input = wrong_ans_input.cuda()
    sample_ans_input = sample_ans_input.cuda()

    fake_len = fake_len.cuda()
    noise_input = noise_input.cuda()
    batch_sample_idx = batch_sample_idx.cuda()
    fake_diff_mask = fake_diff_mask.cuda()
    opt_ans_input = opt_ans_input.cuda()
    gt_index = gt_index.cuda()

ques_input = Variable(ques_input)
img_input = Variable(img_input)
his_input = Variable(his_input)

ans_input = Variable(ans_input)
ans_target = Variable(ans_target)
wrong_ans_input = Variable(wrong_ans_input)
sample_ans_input = Variable(sample_ans_input)

noise_input = Variable(noise_input)
batch_sample_idx = Variable(batch_sample_idx)
fake_diff_mask = Variable(fake_diff_mask)
opt_ans_input = Variable(opt_ans_input)
gt_index = Variable(gt_index)

optimizer = optim.Adam([{'params': netW.parameters()},
                        {'params': netE.parameters()},
                        {'params': netD.parameters()}], lr=opt.lr, betas=(opt.beta1, 0.999))


history = []

for epoch in range(opt.start_epoch+1, opt.niter):

    t = time.time()
    train_loss, lr = train(epoch)
    print ('Epoch: %d learningRate %4f train loss %4f Time: %3f' % (epoch, lr, train_loss, time.time()-t))
    train_his = {'loss': train_loss}

    print('Evaluating ... ')
    rank_all = val()
    R1 = np.sum(np.array(rank_all)==1) / float(len(rank_all))
    R5 =  np.sum(np.array(rank_all)<=5) / float(len(rank_all))
    R10 = np.sum(np.array(rank_all)<=10) / float(len(rank_all))
    ave = np.sum(np.array(rank_all)) / float(len(rank_all))
    mrr = np.sum(1/(np.array(rank_all, dtype='float'))) / float(len(rank_all))
    print ('%d/%d: mrr: %f R1: %f R5 %f R10 %f Mean %f' %(epoch, len(dataloader_val), mrr, R1, R5, R10, ave))
    val_his = {'R1': R1, 'R5':R5, 'R10': R10, 'Mean':ave, 'mrr':mrr}
    history.append({'epoch':epoch, 'train': train_his, 'val': val_his})

    # saving the model.
    if epoch % opt.save_iter == 0:
        torch.save({'epoch': epoch,
                    'opt': opt,
                    'netW': netW.state_dict(),
                    'netD': netD.state_dict(),
                    'netE': netE.state_dict()},
                    '%s/epoch_%d.pth' % (save_path, epoch))

        json.dump(history, open('%s/log.json' %(save_path), 'w'))
