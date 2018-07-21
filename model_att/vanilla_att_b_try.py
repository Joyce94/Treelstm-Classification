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

        self.linear = nn.Linear(self.lstm_hiddens*2, self.label_num, bias=True)

        self.linear_try = nn.Linear(self.lstm_hiddens, 100, bias=True)
        self.u_try = Parameter(torch.randn(100, 1))


    def init_hidden(self, batch_size, num_layers):
        if self.use_cuda:
            return (Variable(torch.zeros(2 * num_layers, batch_size, self.lstm_hiddens // 2)).cuda(),
                    Variable(torch.zeros(2 * num_layers, batch_size, self.lstm_hiddens // 2)).cuda())
        else:
            return (Variable(torch.zeros(2 * num_layers, batch_size, self.lstm_hiddens // 2)),
                    Variable(torch.zeros(2 * num_layers, batch_size, self.lstm_hiddens // 2)))

    def forward(self, fea_v, length, target_start, target_end):
        word_v = fea_v
        batch_size = word_v.size(0)
        seq_length = word_v.size(1)

        word_emb = self.embedding(word_v)
        word_emb = self.dropout_emb(word_emb)

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
        target_save_mean = []
        for idx in range(batch_size):
            mask_none_t = []
            none_t = None
            x_len_cur = x[idx].size(0)
            start_cur = utils.to_scalar(target_start[idx])
            end_cur = utils.to_scalar(target_end[idx])
            x_target = x[idx][start_cur:(end_cur + 1)]
            x_target_mean = torch.mean(x_target, 0)
            target_save_mean.append(x_target_mean.unsqueeze(0))

            x_target_fc = F.tanh(self.linear_try(x_target))
            beta = torch.mm(x_target_fc, self.u_try)
            beta = torch.t(beta)
            if self.use_cuda:
                alpha = F.softmax(beta, dim=1)  # alpha: variable (1, len)
            else:
                alpha = F.softmax(beta)  # alpha: variable (1, len)
            x_average_target = torch.mm(alpha, x_target)
            x_average_target = x_average_target.squeeze(0)
            # print(x_average_target)

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

        target_save = torch.cat(target_save, 0)
        target_save_mean = torch.cat(target_save_mean, 0)
        # print(none_target)
        none_target = torch.cat(none_target, 0)
        mask_none_target = Variable(torch.ByteTensor(mask_none_target))
        # target_save: variable (batch_size, two_hidden_size)
        if self.use_cuda:
            target_save = target_save.cuda()
            mask_none_target = mask_none_target.cuda()
            none_target = none_target.cuda()
            target_save_mean = target_save_mean.cuda()

        # squence = torch.cat(none_target, 1)
        # s = self.attention(none_target, target_save_mean, mask_none_target)
        # # s: variable (batch_size, two_hidden_size)
        # # s = self.attention(x, target_save, None)
        # # target_save_mean: variable (batch_size, two_hidden_size)
        # s = torch.cat([s, target_save], 1)

        # none_target: variable (batch_size, max_length, two_hidden_size)
        none_target = none_target.transpose(1, 2)
        none_target = F.max_pool1d(none_target, none_target.size(2)).squeeze(2)
        # print(none_target)
        s = torch.cat([none_target, target_save_mean], 1)

        result = self.linear(s)  # result: variable (1, label_num)
        # result: variable (batch_size, label_num)
        return result


