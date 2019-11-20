import torch.utils.data as data
from PIL import Image
import torch
import numpy as np
# import torch.multiprocessing as mp
# mp.set_start_method('spawn')
import h5py
import json
import pdb
import random
from misc.utils import repackage_hidden, clip_gradient, adjust_learning_rate, decode_txt
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM


class train(data.Dataset) :  # torch wrapper
    def __init__(self, input_img_h5, input_ques_h5, input_json, negative_sample, num_val, data_split) :
        #This is the number of images for which we have copied the new vgg features to the parallely
        #accessible h5 file. DO NOT CHANGE THIS!!!
        TOTAL_VALID_IMAGES = 82000

        print(h5py.version.info)
        print('DataLoader loading: %s' % data_split)
        print('Loading image feature from %s' % input_img_h5)

        if data_split == 'test' :
            split = 'val'
        else :
            split = 'train'  # train and val split both corresponding to 'train'

        f = json.load(open(input_json, 'r'))
        self.itow = f['itow']
        self.img_info = f['img_' + split]

        self.f_image = h5py.File(input_img_h5, 'r')
        self.imgs = self.f_image['images_' + split]

        # get the data split.
        total_num = TOTAL_VALID_IMAGES
        if data_split == 'train' :
            s = 0
            e = total_num - num_val
        elif data_split == 'val' :
            s = total_num - num_val
            e = total_num
        else :
            s = 0
            e = total_num

        self.img_info = self.img_info[s :e]

        print('%s number of data: %d' % (data_split, e - s))
        # load the data.

        # f.close()

        print('Loading txt from %s' % input_ques_h5)
        f = h5py.File(input_ques_h5, 'r')
        self.ques = f['ques_' + split][s :e]
        self.ans = f['ans_' + split][s :e]
        self.cap = f['cap_' + split][s :e]

        self.ques_len = f['ques_len_' + split][s :e]
        self.ans_len = f['ans_len_' + split][s :e]
        self.cap_len = f['cap_len_' + split][s :e]

        self.ans_ids = f['ans_index_' + split][s :e]
        self.opt_ids = f['opt_' + split][s :e]
        self.opt_list = f['opt_list_' + split][:]
        self.opt_len = f['opt_len_' + split][:]
        f.close()

        self.ques_length = self.ques.shape[2]  # Max word length of a question. Current Value is 16
        self.ans_length = self.ans.shape[2]  # Max word length of answer. Current value is 8
        self.his_length = self.ques_length + self.ans_length  # Max word length of question and answer combined. Current value is 16+8 = 24
        self.vocab_size = len(self.itow) + 1

        print('Vocab Size: %d' % self.vocab_size)
        self.split = split
        self.total_qa_pairs = 10
        self.negative_sample = negative_sample

        # bert tokenzier
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_token_size = 100

    def __getitem__(self, index) :
        # get the image
        img = torch.from_numpy(self.imgs[index])

        # decode text for question
        ques_text = decode_txt(self.itow,
                               torch.from_numpy(self.ques[index].T.astype(dtype='int32')))  # List of strings # len: 10

        # decode text for history and caption
        caption_text = decode_txt(self.itow, torch.from_numpy(self.cap[index:index + 1].T.astype(dtype='int32')))
        ques_list = decode_txt(self.itow, torch.from_numpy(self.ques[index, :9].T.astype(dtype='int32')))
        ans_list = decode_txt(self.itow, torch.from_numpy(self.ans[index, :9].T.astype(dtype='int32')))

        # apply CLS SEP for question
        ques_text = ['[CLS] ' + ques + ' [SEP]' for ques in ques_text]

        # apply CLS SEP for caption
        caption_text = '[CLS] ' + caption_text[0] + ' [SEP]'

        # apply CLS SEP SEP for history
        history_text = []
        for i in range(9):
            history_text.append('[CLS] ' + ques_list[i] + ' [SEP] ' + ans_list[i] + ' [SEP]')

        # tokens for questions
        ques_tokens = [self.tokenizer.tokenize(ques) for ques in ques_text]
        ques_tokens_len = [len(ques_token) for ques_token in ques_tokens]

        # tokens for captions
        caption_tokens = self.tokenizer.tokenize(caption_text)
        capiton_tokens_len = len(caption_tokens)

        # tokens for history
        history_tokens = [self.tokenizer.tokenize(his) for his in history_text]
        history_tokens_len = [len(his_token) for his_token in history_tokens]

        # token ids for question
        question_token_ids = [self.tokenizer.convert_tokens_to_ids(x) for x in ques_tokens]

        # token ids for caption and history
        caption_token_ids = self.tokenizer.convert_tokens_to_ids(caption_tokens)
        history_token_ids = [self.tokenizer.convert_tokens_to_ids(x) for x in history_tokens]

        # add padding to question
        # attention mask for question
        question_array = np.zeros((10, self.max_token_size))
        question_attention_array = np.ones((10, self.max_token_size))
        for i in range(10):
            question_array[i, :ques_tokens_len[i]] = np.array(question_token_ids[i])
        question_attention_array[question_array == 0] = 0

        # add padding to caption and history
        # attention mask for history and caption
        history_array = np.zeros((10, self.max_token_size))
        history_attention_array = np.ones((10, self.max_token_size))
        history_array[0, :capiton_tokens_len] = np.array(caption_token_ids)
        for i in range(1, 10):
            history_array[i, :history_tokens_len[i - 1]] = np.array(history_token_ids[i - 1])
        history_attention_array[history_array == 0] = 0

        # segment ids for question
        question_segment = np.ones((10, self.max_token_size))

        # segment ids for history and caption
        history_segment = np.zeros((10, self.max_token_size))
        sep = np.argmax(history_array[1:10, :] == 102, axis=1)
        for i in range(1, 10):
            history_segment[i, :sep[i - 1] + 1] = 0
            history_segment[i, sep[i - 1] + 1:] = 1

        question_array = torch.from_numpy(question_array)
        question_segment = torch.from_numpy(question_segment)
        question_attention_array = torch.from_numpy(question_attention_array)

        history_array = torch.from_numpy(history_array)
        history_segment = torch.from_numpy(history_segment)
        history_attention_array = torch.from_numpy(history_attention_array)
        ############################################################################################
        # get the history
        #Format of one row of his:
        #0 0 .. 0 0 q q q q a a a a
        #leading zero - (question words - answer words) OR leading zero - caption words
        his = np.zeros((self.total_qa_pairs, self.his_length))
        his[0, self.his_length - self.cap_len[index] :] = self.cap[index, :self.cap_len[index]]

        ques = np.zeros((self.total_qa_pairs, self.ques_length))
        ans_vocab_first = np.zeros((self.total_qa_pairs, self.ans_length + 1))
        ans_vocab_last = np.zeros((self.total_qa_pairs, self.ans_length + 1))
        ques_trailing_zeros = np.zeros((self.total_qa_pairs, self.ques_length))

        opt_ans_vocab_first = np.zeros((self.total_qa_pairs, self.negative_sample, self.ans_length + 1))
        ans_len = np.zeros((self.total_qa_pairs))
        opt_ans_len = np.zeros((self.total_qa_pairs, self.negative_sample))

        ans_idx = np.zeros((self.total_qa_pairs))
        opt_ans_idx = np.zeros((self.total_qa_pairs, self.negative_sample))

        for i in range(self.total_qa_pairs) :
            # get the index
            q_len = self.ques_len[index, i]
            a_len = self.ans_len[index, i]
            qa_len = q_len + a_len

            if i + 1 < self.total_qa_pairs :
                his[i + 1, self.his_length - qa_len :self.his_length - a_len] = self.ques[index, i, :q_len]
                his[i + 1, self.his_length - a_len :] = self.ans[index, i, :a_len]

            ques[i, self.ques_length - q_len :] = self.ques[index, i, :q_len]

            ques_trailing_zeros[i, :q_len] = self.ques[index, i, :q_len]
            ans_vocab_first[i, 1 :a_len + 1] = self.ans[index, i, :a_len]
            ans_vocab_first[i, 0] = self.vocab_size

            ans_vocab_last[i, :a_len] = self.ans[index, i, :a_len]
            ans_vocab_last[i, a_len] = self.vocab_size
            ans_len[i] = self.ans_len[index, i]

            opt_ids = self.opt_ids[index, i]  # since python start from 0
            # random select the negative samples.
            ans_idx[i] = opt_ids[self.ans_ids[index, i]]
            # exclude the gt index.
            opt_ids = np.delete(opt_ids, ans_idx[i], 0)
            random.shuffle(opt_ids)
            for j in range(self.negative_sample) :
                ids = opt_ids[j]
                opt_ans_idx[i, j] = ids

                opt_len = self.opt_len[ids]

                opt_ans_len[i, j] = opt_len
                opt_ans_vocab_first[i, j, :opt_len] = self.opt_list[ids, :opt_len]
                opt_ans_vocab_first[i, j, opt_len] = self.vocab_size

        his = torch.from_numpy(his)
        ques = torch.from_numpy(ques)
        ans_vocab_first = torch.from_numpy(ans_vocab_first)
        ans_vocab_last = torch.from_numpy(ans_vocab_last)
        ques_trailing_zeros = torch.from_numpy(ques_trailing_zeros)
        ans_len = torch.from_numpy(ans_len)
        opt_ans_len = torch.from_numpy(opt_ans_len)
        opt_ans_vocab_first = torch.from_numpy(opt_ans_vocab_first)
        ans_idx = torch.from_numpy(ans_idx)
        opt_ans_idx = torch.from_numpy(opt_ans_idx)

        # return img, his, ques, ans_vocab_first, ans_vocab_last, ans_len, ans_idx, ques_trailing_zeros, \
        #        opt_ans_vocab_first, opt_ans_len, opt_ans_idx

        return img, question_array, question_segment, question_attention_array, history_array, history_segment, history_attention_array ,ans_vocab_first, ans_vocab_last, ans_len, ans_idx, ques_trailing_zeros, \
               opt_ans_vocab_first, opt_ans_len, opt_ans_idx

    def __len__(self) :
        return self.ques.shape[0]


class validate(data.Dataset) :  # torch wrapper
    def __init__(self, input_img_h5, input_ques_h5, input_json, negative_sample, num_val, data_split) :
        # This is the number of images for which we have copied the new vgg features to the parallely
        # accessible h5 file. DO NOT CHANGE THIS!!!
        TOTAL_VALID_TEST_IMAGES = 40000
        TOTAL_VALID_TRAIN_IMAGES = 82000
        print('DataLoader loading: %s' % data_split)
        print('Loading image feature from %s' % input_img_h5)
        total_num = 0
        if data_split == 'test' :
            total_num = TOTAL_VALID_TEST_IMAGES
            split = 'val'
        else :
            total_num = TOTAL_VALID_TRAIN_IMAGES
            split = 'train'  # train and val split both corresponding to 'train'

        f = json.load(open(input_json, 'r'))
        self.itow = f['itow']
        self.img_info = f['img_' + split]

        self.f_image = h5py.File(input_img_h5, 'r')
        self.imgs = self.f_image['images_' + split]

        # get the data split.

        if data_split == 'train' :
            e = total_num - num_val
        elif data_split == 'val' :
            s = total_num - num_val
            e = total_num
        else :
            s = 0
            e = total_num

        self.img_info = self.img_info[s :e]
        print('%s number of data: %d' % (data_split, e - s))

        # load the data.

        print('Loading txt from %s' % input_ques_h5)
        f = h5py.File(input_ques_h5, 'r')
        self.ques = f['ques_' + split][s :e]
        self.ans = f['ans_' + split][s :e]
        self.cap = f['cap_' + split][s :e]

        self.ques_len = f['ques_len_' + split][s :e]
        self.ans_len = f['ans_len_' + split][s :e]
        self.cap_len = f['cap_len_' + split][s :e]

        self.ans_ids = f['ans_index_' + split][s :e]
        self.opt_ids = f['opt_' + split][s :e]
        self.opt_list = f['opt_list_' + split][:]
        self.opt_len = f['opt_len_' + split][:]
        f.close()

        self.ques_length = self.ques.shape[2]
        self.ans_length = self.ans.shape[2]
        self.his_length = self.ques_length + self.ans_length
        self.vocab_size = len(self.itow) + 1

        print('Vocab Size: %d' % self.vocab_size)
        self.split = split
        self.total_qa_pairs = 10
        self.negative_sample = negative_sample

    def __getitem__(self, index) :

        # get the image
        img_id = self.img_info[index]['imgId']
        img = torch.from_numpy(self.imgs[index])
        # get the history
        his = np.zeros((self.total_qa_pairs, self.his_length))
        his[0, self.his_length - self.cap_len[index] :] = self.cap[index, :self.cap_len[index]]

        ques = np.zeros((self.total_qa_pairs, self.ques_length))
        ans_vocab_first = np.zeros((self.total_qa_pairs, self.ans_length + 1))
        ans_vocab_last = np.zeros((self.total_qa_pairs, self.ans_length + 1))
        ques_trailing_zeros = np.zeros((self.total_qa_pairs, self.ques_length))

        opt_ans_vocab_first = np.zeros((self.total_qa_pairs, 100, self.ans_length + 1))
        ans_idx = np.zeros(self.total_qa_pairs)
        opt_ans_vocab_last = np.zeros((self.total_qa_pairs, 100, self.ans_length + 1))

        ans_len = np.zeros((self.total_qa_pairs))
        opt_ans_len = np.zeros((self.total_qa_pairs, 100))

        for i in range(self.total_qa_pairs) :
            # get the index
            q_len = self.ques_len[index, i]
            a_len = self.ans_len[index, i]
            qa_len = q_len + a_len

            if i + 1 < self.total_qa_pairs :
                ques_ans = np.concatenate([self.ques[index, i, :q_len], self.ans[index, i, :a_len]])
                his[i + 1, self.his_length - qa_len :] = ques_ans

            ques[i, self.ques_length - q_len :] = self.ques[index, i, :q_len]
            ques_trailing_zeros[i, :q_len] = self.ques[index, i, :q_len]
            ans_vocab_first[i, 1 :a_len + 1] = self.ans[index, i, :a_len]
            ans_vocab_first[i, 0] = self.vocab_size

            ans_vocab_last[i, :a_len] = self.ans[index, i, :a_len]
            ans_vocab_last[i, a_len] = self.vocab_size

            ans_idx[i] = self.ans_ids[index, i]  # since python start from 0
            opt_ids = self.opt_ids[index, i]  # since python start from 0
            ans_len[i] = self.ans_len[index, i]

            for j, ids in enumerate(opt_ids) :
                opt_len = self.opt_len[ids]
                opt_ans_vocab_first[i, j, 1 :opt_len + 1] = self.opt_list[ids, :opt_len]
                opt_ans_vocab_first[i, j, 0] = self.vocab_size

                opt_ans_vocab_last[i, j, :opt_len] = self.opt_list[ids, :opt_len]
                opt_ans_vocab_last[i, j, opt_len] = self.vocab_size
                opt_ans_len[i, j] = opt_len

        opt_ans_vocab_first = torch.from_numpy(opt_ans_vocab_first)
        opt_ans_vocab_last = torch.from_numpy(opt_ans_vocab_last)
        ans_idx = torch.from_numpy(ans_idx)

        his = torch.from_numpy(his)
        ques = torch.from_numpy(ques)
        ans_vocab_first = torch.from_numpy(ans_vocab_first)
        ans_vocab_last = torch.from_numpy(ans_vocab_last)
        ques_trailing_zeros = torch.from_numpy(ques_trailing_zeros)

        ans_len = torch.from_numpy(ans_len)
        opt_ans_len = torch.from_numpy(opt_ans_len)

        return img, his, ques, ans_vocab_first, ans_vocab_last, ques_trailing_zeros, opt_ans_vocab_first, \
               opt_ans_vocab_last, ans_idx, ans_len, opt_ans_len, img_id

    def __len__(self) :
        return self.ques.shape[0]
