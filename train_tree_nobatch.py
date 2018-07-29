import torch
import torch.nn as nn
import time
import random
import numpy as np
import data.utils as utils
import torch.nn.functional as F
from calc_loss import FocalLoss

def to_scalar(vec):
    return vec.view(-1).data.tolist()[0]

def adjust_learning_rate(optimizer, lr):
    """
    shrink learning rate for pytorch
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_att(train_insts, train_insts_index, dev_insts, dev_insts_index, test_insts, test_insts_index, model_att, config, params):
    print('training...')
    # parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters_att = filter(lambda p: p.requires_grad, model_att.parameters())
    # optimizer = torch.optim.Adam(params=parameters, lr=config.learning_rate, weight_decay=config.decay)
    optimizer_att = torch.optim.Adam(params=parameters_att, lr=config.learning_rate, weight_decay=config.decay)
    best_micro_f1 = float('-inf')
    best_macro_f1 = float('-inf')
    # fl = FocalLoss(class_num=params.label_num)

    for epoch in range(config.maxIters):
        correct = total = 0
        model_att.train()
        train_insts, train_insts_index = utils.random_data(train_insts, train_insts_index)

        epoch_loss_e = 0
        train_buckets, train_labels_raw, train_category_raw, train_target_start, train_target_end = params.generate_batch_buckets(config.train_batch_size, train_insts_index, char=params.add_char)

        for index in range(len(train_buckets)):
            batch_length = np.array([np.sum(mask) for mask in train_buckets[index][-1]])
            fea_v, label_v, pos_v, parse_v, rel_v, mask_v, length_v, target_v, start_v, end_v = utils.patch_var(train_buckets[index], batch_length.tolist(), train_category_raw[index], train_target_start[index], train_target_end[index], params)
            model_att.zero_grad()

            out, loss_e = model_att.forward(fea_v, parse_v, rel_v, label_v, pos_v, batch_length.tolist(), start_v, end_v, mask_v)

            loss_e.backward()
            optimizer_att.step()
            epoch_loss_e += to_scalar(loss_e)

            # pre = to_scalar(torch.max(out, dim=1)[1])
            # tar = to_scalar(target_v)
            #
            # if pre == tar:
            #     correct += 1
            # total += 1
            # print('sentence is {}, loss is {}'.format(index, to_scalar(loss_e)))

        # print('\nepoch is {}, average loss_c is {} '.format(epoch, (epoch_loss_c / config.train_batch_size)))
        print('\nepoch is {}, average loss_e is {}, train_accuracy is {} '.format(epoch, (epoch_loss_e / (config.train_batch_size * len(train_buckets))), (correct / total)))

        print('Dev...')
        dev_micro, dev_macro = eval_att_nobatch(dev_insts, dev_insts_index, model_att, config, params)
        if dev_micro > best_micro_f1:
            best_micro_f1 = dev_micro
            # print('\nTest...')
            # test_acc = eval_att(test_insts, test_insts_index, model_att, config, params)
        if dev_macro > best_macro_f1:
            best_macro_f1 = dev_macro
        print('now, best micro fscore is {}, best macro fscore is {}'.format(best_micro_f1, best_macro_f1))


def eval_att_nobatch(insts, insts_index, model, config, params):
    model.eval()
    insts, insts_index = utils.random_data(insts, insts_index)
    buckets, labels_raw, categorys_raw, target_start, target_end = params.generate_batch_buckets(config.train_batch_size, insts_index, char=params.add_char)

    size = len(insts)
    logit_total = []
    target_total = []
    for index in range(size):
        batch_length = np.array([np.sum(mask) for mask in buckets[index][-1]])
        fea_v, label_v, pos_v, parse_v, rel_v, mask_v, length_v, target_v, start_v, end_v = utils.patch_var(buckets[index], batch_length.tolist(), categorys_raw[index], target_start[index], target_end[index], params)

        target_v = target_v.squeeze(0)
        start_v = start_v.squeeze(0)
        end_v = end_v.squeeze(0)

        logit, _ = model.forward(fea_v, parse_v, rel_v, label_v, pos_v, batch_length.tolist(), start_v, end_v, mask_v)
        logit_total.append(logit)
        target_total.append(target_v)
    logit_total = torch.cat(logit_total, 0)
    target_total = torch.cat(target_total, 0)
    ##### logit_total: variable(size, category_num)
    ##### target_total: variable(size)

    micro_fscore, macro_fscore = calc_fscore(logit_total, target_total, size, params)
    return micro_fscore, macro_fscore


def eval_att(insts, insts_index, model, config, params):
    model.eval()
    insts, insts_index = utils.random_data(insts, insts_index)
    buckets, labels_raw, categorys_raw, target_start, target_end = params.generate_batch_buckets(len(insts), insts_index, char=params.add_char)

    size = len(insts)
    batch_length = np.array([np.sum(mask) for mask in buckets[0][-1]])
    fea_v, label_v, pos_v, parse_v, rel_v, mask_v, length_v, target_v, start_v, end_v = utils.patch_var(buckets[0], batch_length.tolist(), categorys_raw, target_start, target_end, params)

    target_v = target_v.squeeze(0)
    start_v = start_v.squeeze(0)
    end_v = end_v.squeeze(0)

    # if mask_v.size(0) != config.test_batch_size:
    #     model.hidden = model.init_hidden(mask_v.size(0), config.lstm_layers)
    # else:
    #     model.hidden = model.init_hidden(config.test_batch_size, config.lstm_layers)
    # if mask_v.size(0) != config.test_batch_size:
    #     model_e.hidden = model_e.init_hidden(mask_v.size(0), config.lstm_layers)
    # else:
    #     model_e.hidden = model_e.init_hidden(config.test_batch_size, config.lstm_layers)
    # fea_v, parse_v, rel_v, label_v, pos_v, batch_length.tolist(), start_v, end_v, mask_v
    logit = model.forward(fea_v, parse_v, rel_v, label_v, pos_v, batch_length.tolist(), start_v, end_v, mask_v)
    ##### lstm_out: (seq_length, batch_size, label_num)
    ##### label_v: (batch_size, seq_length)
    # lstm_out_e, lstm_out_h = model_e.forward(fea_v, batch_length.tolist())

    # max_index = torch.max(logit, dim=1)[1].view(target_v.size())
    # rel_list = [[], [], [], []]
    # pre_list = [[], [], [], []]
    # corrects_list = [[], [], [], []]
    #
    # corrects = 0
    # for x in range(max_index.size(0)):
    #     y = int(params.category_alphabet.id2word[to_scalar(target_v[x])]) - 1
    #     rel_list[y].append(1)
    #     # print(to_scalar(max_index[x]) == to_scalar(target_v[x]))
    #     # print(type(to_scalar(max_index[x]) == to_scalar(target_v[x])))
    #     if to_scalar(max_index[x]) == to_scalar(target_v[x]):
    #         corrects += 1
    #         y = int(params.category_alphabet.id2word[to_scalar(target_v[x])]) - 1
    #         corrects_list[y].append(1)
    #     r = int(params.category_alphabet.id2word[to_scalar(max_index[x])]) - 1
    #     pre_list[r].append(1)
    # c_list = [len(ele) for ele in corrects_list]
    # r_list = [len(ele) for ele in rel_list]
    # p_list = [len(ele) for ele in pre_list]
    # # assert (torch.max(logit, 1)[1].view(target_v.size()).data == target_v.data).sum() == corrects
    #
    # recall = [float(x) / r_list[id] * 100.0 for id, x in enumerate(c_list)]
    # precision = [float(x) / p_list[id] * 100.0 for id, x in enumerate(c_list)]
    # f_score = []
    # for idx, p in enumerate(precision):
    #     if p + recall[idx] == 0:
    #         f_score.append(0.0)
    #     else:
    #         f_score.append(2 * p * recall[idx] / (p + recall[idx]))
    # for i in range(len(c_list)):
    #     print('category {}: precision: {:.4f}, recall: {}, fscore: {}% ({}/{}/{})'.format(i + 1, precision[i], recall[i], f_score[i], c_list[i], p_list[i], r_list[i]))
    #
    # micro_fscore = float(corrects) / size * 100.0
    # print('\nEvaluation - acc: {:.4f}%({}/{}) \n'.format(micro_fscore, corrects, size))
    # macro_fscore = (f_score[0]+f_score[1]+f_score[2]+f_score[3])/4
    # return micro_fscore, macro_fscore
    micro_fscore, macro_fscore = calc_fscore(logit, target_v, size, params)
    return micro_fscore, macro_fscore


def calc_fscore(logit, target_v, size, params):
    max_index = torch.max(logit, dim=1)[1].view(target_v.size())
    rel_list = [[], [], [], []]
    pre_list = [[], [], [], []]
    corrects_list = [[], [], [], []]

    # print(params.category_alphabet.id2word)     # ['2', '4', '3', '1']
    # print(params.category_alphabet.word2id)     # OrderedDict([('2', 0), ('4', 1), ('3', 2), ('1', 3)])
    corrects = 0
    for x in range(max_index.size(0)):
        target_id = int(params.category_alphabet.id2word[to_scalar(target_v[x])]) - 1
        # y = to_scalar(target_v[x])-1
        # y = int(params.category_alphabet.id2word[to_scalar(target_v[x])])-1
        rel_list[target_id].append(1)
        predict_label = int(params.category_alphabet.id2word[to_scalar(max_index[x])])
        predict_id = predict_label - 1
        if predict_id == target_id:
            corrects += 1
            # print(to_scalar(target_v[x]))
            # print(params.category_alphabet.id2word[to_scalar(target_v[x])])
            # y = int(params.category_alphabet.id2word[to_scalar(target_v[x])]) - 1
            corrects_list[target_id].append(1)
        pre_list[predict_id].append(1)
    c_list = [len(ele) for ele in corrects_list]
    r_list = [len(ele) for ele in rel_list]
    p_list = [len(ele) for ele in pre_list]
    # assert (torch.max(logit, 1)[1].view(target_v.size()).data == target_v.data).sum() == corrects

    # recall = [float(x) / r_list[id] * 100.0 for id, x in enumerate(c_list)]
    recall = []
    for id, x in enumerate(c_list):
        if r_list[id] != 0:
            temp = float(x)/r_list[id]
        else:
            temp = 0
        recall.append(temp)
    # precision = [float(x) / p_list[id] * 100.0 for id, x in enumerate(c_list)]
    precision = []
    for id, x in enumerate(c_list):
        if p_list[id] != 0:
            temp = float(x)/p_list[id]
        else:
            temp = 0
        precision.append(temp)
    f_score = []
    for idx, p in enumerate(precision):
        if p + recall[idx] == 0:
            f_score.append(0.0)
        else:
            f_score.append(2 * p * recall[idx] / (p + recall[idx]))
    for i in range(len(c_list)):
        print('category {}: precision: {:.4f}%, recall: {}%, fscore: {}% ({}/{}/{})'.format(i + 1, precision[i], recall[i], f_score[i], c_list[i], p_list[i], r_list[i]))

    micro_fscore = float(corrects) / size * 100.0
    macro_fscore = (f_score[0] + f_score[1] + f_score[2] + f_score[3]) / 4 * 100.0
    print('\nEvaluation - micro-fscore: {:.4f}%({}/{}), macro-fscore: {:.4f}% \n'.format(micro_fscore, corrects, size, macro_fscore))

    return micro_fscore, macro_fscore





























