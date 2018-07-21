import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model_att.attention_batch import Attention_b
import data.utils as utils


class Context_att_gate(nn.Module):
    def __init__(self, config, params):
        super(Context_att_gate, self).__init__()
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
            self.lstm = nn.LSTM(self.word_dims * 2, self.lstm_hiddens // 2, num_layers=self.lstm_layers, bidirectional=True, dropout=config.dropout_lstm)
        else:
            self.lstm = nn.LSTM(self.word_dims, self.lstm_hiddens // 2, num_layers=self.lstm_layers, bidirectional=True, dropout=config.dropout_lstm)

        self.hidden2label = nn.Linear(self.lstm_hiddens, self.label_num)
        self.hidden = self.init_hidden(self.batch_size, self.lstm_layers)

        self.attention = Attention_b(self.lstm_hiddens, self.attention_size, self.use_cuda)
        self.attention_l = Attention_b(self.lstm_hiddens, self.attention_size, self.use_cuda)
        self.attention_r = Attention_b(self.lstm_hiddens, self.attention_size, self.use_cuda)

        self.w1 = Parameter(torch.randn(self.lstm_hiddens, self.lstm_hiddens))
        self.w2 = Parameter(torch.randn(self.lstm_hiddens, self.lstm_hiddens))
        self.w3 = Parameter(torch.randn(self.lstm_hiddens, self.lstm_hiddens))

        self.u1 = Parameter(torch.randn(self.lstm_hiddens, self.lstm_hiddens))
        self.u2 = Parameter(torch.randn(self.lstm_hiddens, self.lstm_hiddens))
        self.u3 = Parameter(torch.randn(self.lstm_hiddens, self.lstm_hiddens))

        self.b1 = Parameter(torch.randn(self.lstm_hiddens, self.batch_size))
        self.b2 = Parameter(torch.randn(self.lstm_hiddens, self.batch_size))
        self.b3 = Parameter(torch.randn(self.lstm_hiddens, self.batch_size))

        nn.init.xavier_uniform(self.w1)
        nn.init.xavier_uniform(self.w2)
        nn.init.xavier_uniform(self.w3)

        nn.init.xavier_uniform(self.u1)
        nn.init.xavier_uniform(self.u2)
        nn.init.xavier_uniform(self.u3)

        nn.init.xavier_uniform(self.b1)
        nn.init.xavier_uniform(self.b2)
        nn.init.xavier_uniform(self.b3)

        self.linear_2 = nn.Linear(self.lstm_hiddens, self.label_num, bias=True)
        nn.init.xavier_uniform(self.linear_2.weight)


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
        else: word_v = fea_v
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

        ##### batch version
        # x: variable (seq_len, batch_size, hidden_size)
        # target_start: variable (batch_size)
        _, start = torch.max(target_start.unsqueeze(0), dim=1)
        max_start = utils.to_scalar(target_start[start])
        _, end = torch.min(target_end.unsqueeze(0), dim=1)
        min_end = utils.to_scalar(target_end[end])

        x = x.transpose(0, 1)
        left_save = []
        mask_left_save = []
        right_save = []
        mask_right_save = []
        target_save = []
        for idx in range(batch_size):
            x_len_cur = x[idx].size(0)
            start_cur = utils.to_scalar(target_start[idx])
            left_len_cur = start_cur
            left_len_max = max_start
            if start_cur != 0:
                x_cur_left = x[idx][:start_cur]
                left_len_sub = left_len_max - left_len_cur
                mask_cur_left = [1 for _ in range(left_len_cur)]
            else:
                x_cur_left = x[idx][0].unsqueeze(0)
                left_len_sub = left_len_max - 1
                # mask_cur_left = [-1e+20]
                mask_cur_left = [0]
            # x_cur_left: variable (start_cur, two_hidden_size)
            # mask_cur_left = [1 for _ in range(start_cur)]
            # mask_cur_left: list (start_cur)
            if start_cur < max_start:
                if left_len_sub == 0: print('error')
                add = Variable(torch.rand(left_len_sub, self.lstm_hiddens))
                if self.use_cuda: add = add.cuda()
                x_cur_left = torch.cat([x_cur_left, add], dim=0)
                # x_cur_left: variable (max_start, two_hidden_size)
                left_save.append(x_cur_left.unsqueeze(0))
                # mask_cur_left.extend([-1e+20 for _ in range(left_len_sub)])
                mask_cur_left.extend([0 for _ in range(left_len_sub)])
                # mask_cur_left: list (max_start)
                mask_left_save.append(mask_cur_left)
            else:
                left_save.append(x_cur_left.unsqueeze(0))
                mask_left_save.append(mask_cur_left)

            end_cur = utils.to_scalar(target_end[idx])
            right_len_cur = x_len_cur - end_cur - 1
            right_len_max = x_len_cur - min_end - 1
            if (end_cur + 1) != x_len_cur:
                x_cur_right = x[idx][(end_cur + 1):]
                right_len_sub = right_len_max - right_len_cur
                mask_cur_right = [1 for _ in range(right_len_cur)]
            else:
                x_cur_right = x[idx][end_cur].unsqueeze(0)
                right_len_sub = right_len_max - right_len_cur - 1
                # mask_cur_right = [-1e+20]
                mask_cur_right = [0]
            # x_cur_right: variable ((x_len_cur-end_cur-1), two_hidden_size)
            # mask_cur_right = [1 for _ in range(right_len_cur)]
            # mask_cur_right: list (x_len_cur-end_cur-1==right_len)
            if end_cur > min_end:
                if right_len_sub == 0: print('error2')
                add = Variable(torch.rand(right_len_sub, self.lstm_hiddens))
                if self.use_cuda: add = add.cuda()
                x_cur_right = torch.cat([x_cur_right, add], dim=0)
                right_save.append(x_cur_right.unsqueeze(0))
                # mask_cur_right.extend([-1e+20 for _ in range(right_len_sub)])
                mask_cur_right.extend([0 for _ in range(right_len_sub)])
                mask_right_save.append(mask_cur_right)
            else:
                right_save.append(x_cur_right.unsqueeze(0))
                mask_right_save.append(mask_cur_right)

            # target_sub = end_cur-start_cur
            x_target = x[idx][start_cur:(end_cur + 1)]
            x_average_target = torch.mean(x_target, 0)
            target_save.append(x_average_target.unsqueeze(0))
        mask_left_save = Variable(torch.ByteTensor(mask_left_save))
        # mask_left_save: variable (batch_size, left_len_max)
        mask_right_save = Variable(torch.ByteTensor(mask_right_save))
        # mask_right_save: variable (batch_size, right_len_max)
        left_save = torch.cat(left_save, dim=0)
        right_save = torch.cat(right_save, dim=0)
        target_save = torch.cat(target_save, dim=0)
        # left_save: variable (batch_size, left_len_max, two_hidden_size)
        # right_save: variable (batch_size, right_len_max, two_hidden_size)
        # target_save: variable (batch_size, two_hidden_size)
        if self.use_cuda:
            mask_right_save = mask_right_save.cuda()
            mask_left_save = mask_left_save.cuda()
            left_save = left_save.cuda()
            right_save = right_save.cuda()
            target_save = target_save.cuda()

        s, s_alpha = self.attention(x, target_save, None)
        s_l, s_l_alpha = self.attention_l(left_save, target_save, mask_left_save)
        s_r, s_r_alpha = self.attention_r(right_save, target_save, mask_right_save)

        w1s = torch.mm(self.w1, torch.t(s))
        u1t = torch.mm(self.u1, torch.t(target_save))
        if self.use_cuda:
            w1s = w1s.cuda()
            u1t = u1t.cuda()

        if batch_size == self.batch_size:
            z = torch.exp(w1s + u1t + self.b1)
        else:
            z = torch.exp(w1s + u1t)

        z_all = z
        # z_all: variable (two_hidden_size, batch_size)
        z_all = z_all.unsqueeze(2)

        w2s = torch.mm(self.w2, torch.t(s_l))
        u2t = torch.mm(self.u2, torch.t(target_save))
        if self.use_cuda:
            w2s = w2s.cuda()
            u2t = u2t.cuda()
        if batch_size == self.batch_size:
            z_l = torch.exp(w2s + u2t + self.b2)
        else:
            z_l = torch.exp(w2s + u2t)
        # print(z_all)
        # print(z_l)
        z_all = torch.cat([z_all, z_l.unsqueeze(2)], dim=2)

        w3s = torch.mm(self.w3, torch.t(s_r))
        u3t = torch.mm(self.u3, torch.t(target_save))
        if self.use_cuda:
            w3s = w3s.cuda()
            u3t = u3t.cuda()
        if batch_size == self.batch_size:
            z_r = torch.exp(w3s + u3t + self.b3)
        else:
            z_r = torch.exp(w3s + u3t)
        z_all = torch.cat([z_all, z_r.unsqueeze(2)], dim=2)

        # z_all: variable (two_hidden_size, batch_size, 3)
        if self.use_cuda:
            z_all = F.softmax(z_all, dim=2)
        else:
            z_all = F.softmax(z_all)
        # z_all = torch.t(z_all)
        z_all = z_all.permute(2, 1, 0)
        # z = torch.unsqueeze(z_all[:batch_size], 0)
        # z_l = torch.unsqueeze(z_all[batch_size:(2*batch_size)], 0)
        # z_r = torch.unsqueeze(z_all[(2*batch_size):], 0)
        # z = z_all[:batch_size]
        # z_l = z_all[batch_size:(2*batch_size)]
        # z_r = z_all[(2*batch_size):]
        z = z_all[0]
        z_l = z_all[1]
        z_r = z_all[2]

        ss = torch.mul(z, s)
        ss = torch.add(ss, torch.mul(z_l, s_l))
        ss = torch.add(ss, torch.mul(z_r, s_r))

        logit = self.linear_2(ss)
        # print(logit)
        alpha = [s_alpha, s_l_alpha, s_r_alpha]
        return logit, alpha











