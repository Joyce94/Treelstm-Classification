import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model_att.attention_batch import Attention_b
import data.utils as utils


class Vanilla_att(nn.Module):
    def __init__(self, config, params):
        super(Vanilla_att, self).__init__()
        self.word_num = params.word_num
        self.label_num = params.label_num
        self.char_num = params.char_num

        self.id2word = params.word_alphabet.id2word
        self.word2id = params.word_alphabet.word2id
        self.padID = params.word_alphabet.word2id['<pad>']
        self.unkID = params.word_alphabet.word2id['<unk>']

        self.use_cuda = params.use_cuda
        self.add_char = params.add_char
        self.static = params.static

        self.feature_count = config.shrink_feature_thresholds
        self.word_dims = config.word_dims
        self.char_dims = config.char_dims

        self.lstm_hiddens = config.lstm_hiddens
        self.attention_size = config.attention_size

        self.dropout_emb = nn.Dropout(p=config.dropout_emb)
        self.dropout_lstm = nn.Dropout(p=config.dropout_lstm)

        self.lstm_layers = config.lstm_layers
        self.batch_size = config.train_batch_size

        self.embedding = nn.Embedding(self.word_num, self.word_dims)
        self.embedding.weight.requires_grad = True
        if self.static:
            self.embedding_static = nn.Embedding(self.word_num, self.word_dims)
            self.embedding_static.weight.requires_grad = False

        if params.pretrain_word_embedding is not None:
            # pretrain_weight = np.array(params.pretrain_word_embedding)
            # self.embedding.weight.data.copy_(torch.from_numpy(pretrain_weight))
            # pretrain_weight = np.array(params.pretrain_embed)
            pretrain_weight = torch.FloatTensor(params.pretrain_word_embedding)
            self.embedding.weight.data.copy_(pretrain_weight)

        # for id in range(self.word_dims):
        #     self.embedding.weight.data[self.eofID][id] = 0

        if params.static:
            self.lstm = nn.LSTM(self.word_dims * 2, self.lstm_hiddens // 2, num_layers=self.lstm_layers,
                                bidirectional=True, dropout=config.dropout_lstm)
        else:
            self.lstm = nn.LSTM(self.word_dims, self.lstm_hiddens // 2, num_layers=self.lstm_layers, bidirectional=True,
                                dropout=config.dropout_lstm)

        self.hidden2label = nn.Linear(self.lstm_hiddens, self.label_num)
        self.hidden = self.init_hidden(self.batch_size, self.lstm_layers)

        # self.attention = Attention(self.lstm_hiddens, self.attention_size, self.use_cuda)
        # self.attention_l = Attention(self.lstm_hiddens, self.attention_size, self.use_cuda)
        # self.attention_r = Attention(self.lstm_hiddens, self.attention_size, self.use_cuda)

        self.attention = Attention_b(self.lstm_hiddens, self.attention_size, self.use_cuda)
        self.attention_l = Attention_b(self.lstm_hiddens, self.attention_size, self.use_cuda)
        self.attention_r = Attention_b(self.lstm_hiddens, self.attention_size, self.use_cuda)

        self.linear = nn.Linear(self.lstm_hiddens, self.label_num, bias=True)
        self.linear_l = nn.Linear(self.lstm_hiddens, self.label_num, bias=True)
        self.linear_r = nn.Linear(self.lstm_hiddens, self.label_num, bias=True)

    def init_hidden(self, batch_size, num_layers):
        if self.use_cuda:
            return (Variable(torch.zeros(2 * num_layers, batch_size, self.lstm_hiddens // 2)).cuda(),
                    Variable(torch.zeros(2 * num_layers, batch_size, self.lstm_hiddens // 2)).cuda())
        else:
            return (Variable(torch.zeros(2 * num_layers, batch_size, self.lstm_hiddens // 2)),
                    Variable(torch.zeros(2 * num_layers, batch_size, self.lstm_hiddens // 2)))

    def forward(self, fea_v, length, target_start, target_end):
        if self.add_char:
            word_v = fea_v[0]
            char_v = fea_v[1]
        else:
            word_v = fea_v
        batch_size = word_v.size(0)
        seq_length = word_v.size(1)

        word_emb = self.embedding(word_v)
        word_emb = self.dropout_emb(word_emb)
        if self.static:
            word_static = self.embedding_static(word_v)
            word_static = self.dropout_emb(word_static)
            word_emb = torch.cat([word_emb, word_static], 2)

        x = torch.transpose(word_emb, 0, 1)
        packed_words = pack_padded_sequence(x, length)
        lstm_out, self.hidden = self.lstm(packed_words, self.hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        ##### lstm_out: (seq_len, batch_size, hidden_size)
        lstm_out = self.dropout_lstm(lstm_out)
        x = lstm_out
        x = x.transpose(0, 1)
        ##### batch version
        # x = torch.squeeze(lstm_out, 1)
        # x: variable (seq_len, batch_size, hidden_size)
        # target_start: variable (batch_size)
        # _, start = torch.max(target_start.unsqueeze(0), dim=1)
        # max_start = utils.to_scalar(target_start[start])
        # _, end = torch.min(target_end.unsqueeze(0), dim=1)
        # min_end = utils.to_scalar(target_end[end])
        max_length = 0
        for index in range(batch_size):
            x_len = x[index].size(0)
            start = utils.to_scalar(target_start[index])
            end = utils.to_scalar(target_end[index])
            none_t = x_len-(end-start+1)
            if none_t > max_length: max_length = none_t


        # left_save = []
        # mask_left_save = []
        # right_save = []
        # mask_right_save = []
        none_target = []
        mask_none_target = []
        target_save = []
        for idx in range(batch_size):
            mask_none_t = []
            none_t = None
            x_len_cur = x[idx].size(0)
            start_cur = utils.to_scalar(target_start[idx])
            end_cur = utils.to_scalar(target_end[idx])
            # left_len_cur = start_cur
            # left_len_max = max_start
            x_target = x[idx][start_cur:(end_cur + 1)]
            x_average_target = torch.mean(x_target, 0)
            target_save.append(x_average_target.unsqueeze(0))
            if start_cur != 0:
                left = x[idx][:start_cur]
                none_t = left
                mask_none_t.extend([1]*start_cur)
            if end_cur != (x_len_cur-1):
                right = x[idx][(end_cur+1):]
                if none_t is not None: none_t = torch.cat([none_t, right], 0)
                else: none_t = right
                mask_none_t.extend([1]*(x_len_cur-end_cur-1))
            if len(mask_none_t) != max_length:
                add_t = Variable(torch.zeros((max_length - len(mask_none_t)), self.lstm_hiddens))
                if self.use_cuda: add_t = add_t.cuda()
                mask_none_t.extend([0]*(max_length-len(mask_none_t)))
                # print(add_t)
                none_t = torch.cat([none_t, add_t], 0)
            mask_none_target.append(mask_none_t)
            none_target.append(none_t.unsqueeze(0))
            # if start_cur != 0:
            #     x_cur_left = x[idx][:start_cur]
            #     left_len_sub = left_len_max - left_len_cur
            #     mask_cur_left = [1 for _ in range(left_len_cur)]
            # else:
            #     x_cur_left = x[idx][0].unsqueeze(0)
            #     left_len_sub = left_len_max - 1
            #     # mask_cur_left = [-1e+20]
            #     mask_cur_left = [0]
            # # x_cur_left: variable (start_cur, two_hidden_size)
            # # mask_cur_left = [1 for _ in range(start_cur)]
            # # mask_cur_left: list (start_cur)
            # if start_cur < max_start:
            #     add = Variable(torch.zeros(left_len_sub, self.lstm_hiddens))
            #     if self.use_cuda: add = add.cuda()
            #     x_cur_left = torch.cat([x_cur_left, add], 0)
            #     # x_cur_left: variable (max_start, two_hidden_size)
            #     left_save.append(x_cur_left.unsqueeze(0))
            #     # mask_cur_left.extend([-1e+20 for _ in range(left_len_sub)])
            #     mask_cur_left.extend([0 for _ in range(left_len_sub)])
            #     # mask_cur_left: list (max_start)
            #     mask_left_save.append(mask_cur_left)
            # else:
            #     left_save.append(x_cur_left.unsqueeze(0))
            #     mask_left_save.append(mask_cur_left)
            #
            # end_cur = utils.to_scalar(target_end[idx])
            # right_len_cur = x_len_cur - end_cur - 1
            # right_len_max = x_len_cur - min_end - 1
            # if (end_cur + 1) != x_len_cur:
            #     x_cur_right = x[idx][(end_cur + 1):]
            #     right_len_sub = right_len_max - right_len_cur
            #     mask_cur_right = [1 for _ in range(right_len_cur)]
            # else:
            #     x_cur_right = x[idx][end_cur].unsqueeze(0)
            #     right_len_sub = right_len_max - right_len_cur - 1
            #     # mask_cur_right = [-1e+20]
            #     mask_cur_right = [0]
            # # x_cur_right: variable ((x_len_cur-end_cur-1), two_hidden_size)
            # # mask_cur_right = [1 for _ in range(right_len_cur)]
            # # mask_cur_right: list (x_len_cur-end_cur-1==right_len)
            # if end_cur > min_end:
            #     add = Variable(torch.zeros(right_len_sub, self.lstm_hiddens))
            #     if self.use_cuda: add = add.cuda()
            #     x_cur_right = torch.cat([x_cur_right, add], 0)
            #     right_save.append(x_cur_right.unsqueeze(0))
            #     # mask_cur_right.extend([-1e+20 for _ in range(right_len_sub)])
            #     mask_cur_right.extend([0 for _ in range(right_len_sub)])
            #     mask_right_save.append(mask_cur_right)
            # else:
            #     right_save.append(x_cur_right.unsqueeze(0))
            #     mask_right_save.append(mask_cur_right)

        # mask_left_save = Variable(torch.ByteTensor(mask_left_save))
        # # mask_left_save: variable (batch_size, left_len_max)
        # mask_right_save = Variable(torch.ByteTensor(mask_right_save))
        # # mask_right_save: variable (batch_size, right_len_max)
        # left_save = torch.cat(left_save, 0)
        # right_save = torch.cat(right_save, 0)
        target_save = torch.cat(target_save, 0)
        # print(none_target)
        none_target = torch.cat(none_target, 0)
        mask_none_target = Variable(torch.ByteTensor(mask_none_target))
        # left_save: variable (batch_size, left_len_max, two_hidden_size)
        # right_save: variable (batch_size, right_len_max, two_hidden_size)
        # target_save: variable (batch_size, two_hidden_size)
        if self.use_cuda:
            # mask_right_save = mask_right_save.cuda()
            # mask_left_save = mask_left_save.cuda()
            # left_save = left_save.cuda()
            # right_save = right_save.cuda()
            target_save = target_save.cuda()
            mask_none_target = mask_none_target.cuda()
            none_target = none_target.cuda()

        # squence = torch.cat(none_target, 1)
        s = self.attention(none_target, target_save, mask_none_target)
        # s = self.attention(x, target_save, None)
        # s_l = self.attention_l(left_save, target_save, mask_left_save)
        # s_r = self.attention_r(right_save, target_save, mask_right_save)

        result = self.linear(s)  # result: variable (1, label_num)
        # result = self.linear_l(s_l)
        # result = torch.add(result, self.linear_l(s_l))
        # result = torch.add(result, self.linear_r(s_r))
        # result: variable (batch_size, label_num)
        # print(result)
        return result


