import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
import torch.autograd as autograd
# from pack_embedding import LoadEmbedding
import numpy as np
import time
import torch.optim as optim
from model_att.tree import *


class ChildSumTreeLSTM(nn.Module):
    def __init__(self, config, params):
        super(ChildSumTreeLSTM, self).__init__()
        self.word_num = params.n_embed
        self.label_num = 4
        self.char_num = 0
        self.category_num = 4
        self.pos_num = 0
        self.parse_num = 0
        self.n_embed = params.n_embed

        self.id2word = params.word_alphabet.id2word
        self.word2id = params.word_alphabet.word2id
        self.padID = params.word_alphabet.word2id['<pad>']
        self.unkID = params.word_alphabet.word2id['<unk>']

        self.use_cuda = params.use_cuda
        self.feature_count = config.shrink_feature_thresholds
        self.word_dims = config.word_dims
        self.char_dims = config.char_dims

        self.lstm_hiddens = config.lstm_hiddens
        self.dropout_emb = nn.Dropout(p=config.dropout_emb)
        self.dropout = nn.Dropout(p=config.dropout_lstm)

        self.lstm_layers = config.lstm_layers
        self.batch_size = config.train_batch_size

        self.embedding = nn.Embedding(self.n_embed, self.word_dims)
        self.embedding.weight.requires_grad = True
        self.embedding_label = nn.Embedding(self.label_num, self.word_dims)
        self.embedding_label.weight.requires_grad = True
        # self.embedding_pos = nn.Embedding(self.pos_num, self.word_dims)
        # self.embedding_pos.weight.requires_grad = True

        if params.pretrain_word_embedding is not None:
            # pretrain_weight = np.array(params.pretrain_word_embedding)
            # self.embedding.weight.data.copy_(torch.from_numpy(pretrain_weight))
            # pretrain_weight = np.array(params.pretrain_embed)
            pretrain_weight = torch.FloatTensor(params.pretrain_word_embedding)
            self.embedding.weight.data.copy_(pretrain_weight)

        # self.in_dim = hyperparameter.embed_dim
        # self.hidden_dim = hyperparameter.hidden_dim
        # self.out_dim = hyperparameter.n_label
        self.in_dim = self.word_dims
        self.hidden_dim = self.lstm_hiddens
        self.out_dim = self.category_num

        self.ix = nn.Linear(self.in_dim, self.hidden_dim)
        self.ih = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.fx = nn.Linear(self.in_dim, self.hidden_dim)
        self.fh = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.ox = nn.Linear(self.in_dim, self.hidden_dim)
        self.oh = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.ux = nn.Linear(self.in_dim, self.hidden_dim)
        self.uh = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.out = nn.Linear(self.hidden_dim, self.out_dim)

        self.softmax = nn.LogSoftmax(dim=1)
        self.loss_func = nn.NLLLoss()
        self.dropout = nn.Dropout(config.dropout_emb)

        if self.use_cuda:
            self.loss_func.cuda()

    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = F.torch.sum(torch.squeeze(child_h, 1), 0)

        i = F.sigmoid(self.ix(inputs) + self.ih(child_h_sum))
        o = F.sigmoid(self.ox(inputs) + self.oh(child_h_sum))
        u = F.tanh(self.ux(inputs) + self.uh(child_h_sum))

        fx = F.torch.unsqueeze(self.fx(inputs), 1)
        f = F.torch.cat([self.fh(child_hi) + fx for child_hi in child_h], 0)
        f = F.sigmoid(f)

        fc = F.torch.squeeze(F.torch.mul(f, child_c), 1)

        c = F.torch.mul(i, u) + F.torch.sum(fc, 0)
        h = F.torch.mul(o, F.tanh(c))

        return c, h

    def forward(self, xs, heads, rels, labels, poss, xlengths, target_start, target_end, x_mask):
        word_emb = self.embedding(xs)
        label_emb = self.embedding_label(labels)
        # pos_emb = self.embedding(poss)
        # emb = torch.cat([word_emb, label_emb, pos_emb], 2)
        ##### emb: variable(batch_size, length, 3*word_dims)
        emb = torch.cat([word_emb, label_emb], 2)

        rel_emb = self.rel_embedding(rels)

        max_length = emb.size(1)
        batch_size = emb.size(0)

        trees = []
        indexes = np.zeros((max_length, batch_size), dtype=np.int32)
        for b, head in enumerate(heads):
            root, tree = creatTree(head)
            root.traverse()
            for step, index in enumerate(root.order):
                indexes[step, b] = index
            trees.append(tree)

        output, loss = self.treelstm_loss(trees[0], emb)

        return output, loss

    def treelstm_loss(self, tree, embeds):
        # embeds = self.embedding(embeds)
        if self.use_cuda:
            loss = autograd.Variable(torch.zeros(1)).cuda()
        else:
            loss = autograd.Variable(torch.zeros(1))
        for child in tree.left_children:
            _, child_loss = self.treelstm_loss(child, embeds)
            loss = loss + child_loss
        for child in tree.right_children:
            _, child_loss = self.treelstm_loss(child, embeds)
            loss = loss + child_loss
        child_c, child_h = self.get_child_states(tree)
        tree.state = self.node_forward(embeds[tree.index - 1], child_c, child_h)        #####

        output = self.out(self.dropout(tree.state[1]))
        output = self.softmax(output)
        return output, loss

    def get_child_states(self, tree):
        """
        get c and h of all children
        :param tree:
        :return:
        """
        num_children = len(tree.left_children) + len(tree.right_children)
        if num_children == 0:
            child_c = autograd.Variable(torch.zeros(1, 1, self.hidden_dim))
            child_h = autograd.Variable(torch.zeros(1, 1, self.hidden_dim))
            if self.use_cuda:
                child_c, child_h = child_c.cuda(), child_h.cuda()
        else:
            child_c = autograd.Variable(torch.Tensor(num_children, 1, self.hidden_dim))
            child_h = autograd.Variable(torch.Tensor(num_children, 1, self.hidden_dim))
            children = tree.left_children + tree.right_children
            for idx, child in enumerate(children):
                child_c[idx] = child.state[0]
                child_h[idx] = child.state[1]
            if self.use_cuda:
                child_c, child_h = child_c.cuda(), child_h.cuda()
        return child_c, child_h


