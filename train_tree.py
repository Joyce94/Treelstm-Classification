import torch
import torch.nn as nn
import time
import random
import torch.autograd as autograd
import numpy as np
import data.utils as utils
import model_joint.evaluation as evaluation
import data.vocab as vocab
import torch.nn.functional as F
from model_att.tree_batch import Forest


def train(train_datasets, dev_datasets, test_datasets, model_att, config, params):
    parameters = filter(lambda p: p.requires_grad, model_att.parameters())
    optimizer = torch.optim.Adam(params=parameters, lr=config.learning_rate, weight_decay=config.decay)
    best_micro_f1 = float('-inf')
    best_macro_f1 = float('-inf')

    train_trees = train_datasets[0]
    data_len = len(train_trees)
    batch_block = data_len // config.train_batch_size
    if data_len % config.train_batch_size:
        batch_block += 1

    for epoch in range(config.maxIters):
        loss_sum = 0
        model_att.train()
        # train_insts, train_insts_index = utils.random_data(train_insts, train_insts_index)
        # epoch_loss_e = 0
        # train_buckets, train_labels_raw, train_category_raw, train_target_start, train_target_end = params.generate_batch_buckets(config.train_batch_size, train_insts_index, char=params.add_char)

        # for index in range(len(train_buckets)):
        #     batch_length = np.array([np.sum(mask) for mask in train_buckets[index][-1]])
        #     fea_v, label_v, parse_v, mask_v, length_v, target_v, start_v, end_v = utils.patch_var(train_buckets[index], batch_length.tolist(), train_category_raw[index], train_target_start[index], train_target_end[index], params)
        #     # model.zero_grad()
        #     model_att.zero_grad()
        num_sentences = cor_sentences = 0
        start = time.time()
        random.shuffle(train_trees)
        # train_trees = utils.random_data_tree(train_trees)
        for idx in range(batch_block):
            model_att.zero_grad()
            left = idx * config.train_batch_size
            right = left + config.train_batch_size
            if right < data_len:
                forest = Forest(train_trees[left:right])
            else:
                forest = Forest(train_trees[left:])
            if params.use_cuda:
                sen = autograd.Variable(torch.LongTensor([n.word_idx for n in forest.node_list])).cuda()
                y = autograd.Variable(torch.LongTensor([t.category for t in forest.trees])).cuda()
            else:
                sen = autograd.Variable(torch.LongTensor([n.word_idx for n in forest.node_list]))
                y = autograd.Variable(torch.LongTensor([t.category for t in forest.trees]))
            # print(sen)      # [torch.LongTensor of size 2975]
            out = model_att.forward(forest, sen)    # out: [torch.FloatTensor of size 100x4]
            loss = F.cross_entropy(out, y)
            loss.backward()
            loss_sum += loss.data[0]
            optimizer.step()
            # optimizer.zero_grad()
            out.data[:, 1] = -1e+7
            val, pred = torch.max(out.data, 1)
            for x in range(len(forest.trees)):
                if forest.trees[x].category == pred[x]:
                    cor_sentences += 1
                num_sentences += 1
            forest.clean_state()
            if params.use_cuda:
                torch.cuda.empty_cache()
            # print("test time:",time.time()-test_start)
        print("this {} epoch loss :{},train_accuracy : {},consume time:{}".format(epoch, loss_sum / data_len, cor_sentences / num_sentences, time.time() - start))


def test(embed_model, model, dataset, dataset_name):
    model.eval()
    embed_model.eval()
    cor = total = 0
    # no batch
    data_len = len(dataset[0])
    data_trees = dataset[0]
    data_sentences = dataset[1]
    for idx in range(data_len):
        sen = autograd.Variable(torch.LongTensor(data_sentences[idx]), volatile=True)

        embeds = torch.unsqueeze(embed_model(sen), 1)
        output, loss = model(data_trees[idx], embeds)

        output[:, 1] = -9999
        val, pred = torch.max(output.data, 1)
        if pred[0] == data_trees[idx].label:
            cor += 1
        total += 1
    # batch
    # forest = dataset
    # y = torch.LongTensor([t.label for t in forest.trees])
    # out, loss = model(forest)
    # out[:,1] = -9999
    # val,pred = torch.max(out.data,1)
    # for x in range(len(forest.trees)):
    #     if y[x] == pred[x]:
    #         cor += 1
    #     total += 1
    print(dataset_name, total)
    print("{}_set accuracy:{}".format(dataset_name, cor / total))

def batchtest(embed_model, model, dataset, dataset_name):
    model.eval()
    embed_model.eval()
    cor = total = 0
    data_len = len(dataset)
    batch_block = data_len // self.args.batch_size
    if data_len % self.args.batch_size:
        batch_block += 1
    for idx in range(batch_block):
        left = idx * self.args.batch_size
        right = left + self.args.batch_size
        if right < data_len:
            forest = Forest(dataset[left:right])
        else:
            forest = Forest(dataset[left:])
        # y = torch.LongTensor([t.label for t in forest.trees])
        if self.args.cuda:
            sen = autograd.Variable(torch.LongTensor([n.word_idx for n in forest.node_list]).cuda())
        else:
            sen = autograd.Variable(torch.LongTensor([n.word_idx for n in forest.node_list]))
        # embeds = torch.unsqueeze(embed_model(sen),1)
        # out,loss = model(forest)
        # out,loss = model(forest,embed_model(sen))
        if self.args.bidirectional is False:
            out = model.forward(forest, embed_model(sen))
        else:
            out = model.bid_forward(forest, embed_model(sen))
        if not self.args.five:
            out.data[:, 1] = -1e+7
        val, pred = torch.max(out.data, 1)
        for x in range(len(forest.trees)):
            if forest.trees[x].label == pred[x]:
                cor += 1
            total += 1
        forest.clean_state()
    print("{}_set accuracy:{}".format(dataset_name, cor / total))
    return cor / total

def get_max_index(output):
    max_list = []
    row = output.size()[0]
    col = output.size()[1]
    for i in range(row):
        max_index, max_num = 0, output[i][0]
        for j in range(col):
            tmp = output[i][j]
            if max_num < tmp:
                max_num = tmp
                max_index = j
        max_list.append(max_index)
    return max_list

