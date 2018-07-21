import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model_att.attention_batch import Attention_b
import data.utils as utils


class LSTM_att(nn.Module):
    def __init__(self, config, params):
        super(LSTM_att, self).__init__()
        self.word_num = params.word_num
        self.label_num = params.label_num
        self.char_num = params.char_num
        self.category_num = params.category_num
        self.parse_num = params.parse_num

        # self.id2label = params.label_alphabet.id2word
        # self.label2id = params.label_alphabet.word2id

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
        self.embedding_label = nn.Embedding(self.label_num, self.word_dims)
        self.embedding_label.weight.requires_grad = True
        self.embedding_parse = nn.Embedding(self.parse_num, self.word_dims)
        self.embedding_parse.weight.requires_grad = True

        if params.pretrain_word_embedding is not None:
            # pretrain_weight = np.array(params.pretrain_word_embedding)
            # self.embedding.weight.data.copy_(torch.from_numpy(pretrain_weight))
            # pretrain_weight = np.array(params.pretrain_embed)
            pretrain_weight = torch.FloatTensor(params.pretrain_word_embedding)
            self.embedding.weight.data.copy_(pretrain_weight)

        # for id in range(self.word_dims):
        #     self.embedding.weight.data[self.eofID][id] = 0

        self.lstm = nn.LSTM(self.word_dims*2, self.lstm_hiddens // 2, num_layers=self.lstm_layers, bidirectional=True,
                                dropout=config.dropout_lstm)

        self.hidden2label = nn.Linear(self.lstm_hiddens, self.label_num)
        self.hidden = self.init_hidden(self.batch_size, self.lstm_layers)

        # self.attention = Attention(self.lstm_hiddens, self.attention_size, self.use_cuda)
        # self.attention_l = Attention(self.lstm_hiddens, self.attention_size, self.use_cuda)
        # self.attention_r = Attention(self.lstm_hiddens, self.attention_size, self.use_cuda)

        self.attention = Attention_b(self.lstm_hiddens, self.attention_size, self.use_cuda)
        self.attention_l = Attention_b(self.lstm_hiddens, self.attention_size, self.use_cuda)
        self.attention_r = Attention_b(self.lstm_hiddens, self.attention_size, self.use_cuda)

        self.linear = nn.Linear(self.lstm_hiddens, self.category_num, bias=True)

        self.linear_try = nn.Linear(self.lstm_hiddens, 100, bias=True)
        self.u_try = Parameter(torch.randn(100, 1))


    def init_hidden(self, batch_size, num_layers):
        if self.use_cuda:
            return (Variable(torch.zeros(2 * num_layers, batch_size, self.lstm_hiddens // 2)).cuda(),
                    Variable(torch.zeros(2 * num_layers, batch_size, self.lstm_hiddens // 2)).cuda())
        else:
            return (Variable(torch.zeros(2 * num_layers, batch_size, self.lstm_hiddens // 2)),
                    Variable(torch.zeros(2 * num_layers, batch_size, self.lstm_hiddens // 2)))

    def forward(self, fea_v, parse_v, length, target_start, target_end, label_v):
        # print(self.label_num)
        # print(self.id2label)
        # print(self.label2id)
        # print(label_v)

        word_v = fea_v
        batch_size = word_v.size(0)
        seq_length = word_v.size(1)

        word_emb = self.embedding(word_v)
        word_emb = self.dropout_emb(word_emb)

        label_emb = self.embedding_label(label_v)
        label_emb = self.dropout_emb(label_emb)

        parse_emb = self.embedding_parse(parse_v)
        parse_emb = self.dropout_emb(parse_emb)

        #x = torch.cat([word_emb, label_emb, parse_emb], 2)
        x = torch.cat([word_emb, label_emb], 2)
        x = torch.transpose(x, 0, 1)
        packed_words = pack_padded_sequence(x, length)
        lstm_out, self.hidden = self.lstm(packed_words, self.hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        ##### lstm_out: (seq_len, batch_size, hidden_size)
        lstm_out = self.dropout_lstm(lstm_out)
        x = lstm_out.transpose(0, 1)
        x = x.transpose(1, 2)

        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        # print(x)
        result = self.linear(x)  # result: variable (1, label_num)
        # result: variable (batch_size, label_num)
        return result


