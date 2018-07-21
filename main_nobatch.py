import argparse
import data.config as config
from data.vocab import Data
import model_joint.lstm as lstm
import model_joint.lstm_e as lstm_e
import model_joint.calc_loss as calc_loss

import model_att.context_att_b as context_att_b
import model_att.vanilla_att_b_new as vanilla_att_b_new
import model_att.vanilla_att_b_try as vanilla_att_b_try
import model_att.context_att_gate_b as context_att_gate_b
import model_att.bilstm as bilstm
import model_att.treelstm as treelstm
import model_att.DL4MT as DL4MT

import train_att
import train_tree
import train_tree_nobatch

import torch
import random
import numpy as np
import time
import data.vocab_base as vocab_base
import data.data_utils as data_utils


if __name__ == '__main__':
    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(666)
    torch.backends.cudnn.enabled = False

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config-file', default=r'C:\Users\song\Desktop\treelstm_word_nobatch\examples\config.cfg')
    argparser.add_argument('--use-cuda', default=False)
    argparser.add_argument('--static', default=False, help='fix the embedding')
    argparser.add_argument('--add-char', default=False, help='add char feature')
    argparser.add_argument('--metric', default='exact', help='choose from [exact, binary, proportional]')
    argparser.add_argument('--model', default='treelstm', help='choose from [treelstm, vanilla, context_att, context_att_gate]')

    # args = argparser.parse_known_args()
    args = argparser.parse_args()
    config = config.Configurable(args.config_file)

    data = Data()
    data.number_normalized = False
    data.static = args.static
    data.add_char = args.add_char
    data.use_cuda = args.use_cuda
    data.metric = args.metric

    test_time = time.time()
    train_insts, train_insts_index = data.get_instance(config.train_file, config.run_insts, config.shrink_feature_thresholds, char=args.add_char)
    print('test getting train_insts time: ', time.time()-test_time)
    # print(len(train_insts))
    if not args.static:
        data.fix_alphabet()
    dev_insts, dev_insts_index = data.get_instance(config.dev_file, config.run_insts, config.shrink_feature_thresholds, char=args.add_char)
    print('test getting dev_insts time: ', time.time() - test_time)

    data.fix_alphabet()
    test_insts, test_insts_index = data.get_instance(config.test_file, config.run_insts, config.shrink_feature_thresholds, char=args.add_char)
    print('test getting test_insts time: ', time.time() - test_time)

    # test_time = time.time()
    # write_word_path = r"C:\Users\song\Desktop\treelstm_word\examples\sent_train.txt"
    # label_path = r"C:\Users\song\Desktop\treelstm_word\examples\label_train.txt"
    # pos_path = r"C:\Users\song\Desktop\treelstm_word\examples\pos_train.txt"
    # parse_path = r"C:\Users\song\Desktop\treelstm_word\examples\parse_train.txt"
    # category_path = r"C:\Users\song\Desktop\treelstm_word\examples\category_train.txt"
    # train_insts, train_insts_index = data.get_instance(config.train_file, config.run_insts, config.shrink_feature_thresholds, write_word_path, label_path, pos_path, parse_path, category_path, char=args.add_char)
    # print('test getting train_insts time: ', time.time()-test_time)
    #
    # # if not args.static:
    # #     data.fix_alphabet()
    # write_word_path = r"C:\Users\song\Desktop\treelstm_word\examples\sent_test.txt"
    # label_path = r"C:\Users\song\Desktop\treelstm_word\examples\label_test.txt"
    # pos_path = r"C:\Users\song\Desktop\treelstm_word\examples\pos_test.txt"
    # parse_path = r"C:\Users\song\Desktop\treelstm_word\examples\parse_test.txt"
    # category_path = r"C:\Users\song\Desktop\treelstm_word\examples\category_test.txt"
    # dev_insts, dev_insts_index = data.get_instance(config.dev_file, config.run_insts, config.shrink_feature_thresholds, write_word_path, label_path, pos_path, parse_path, category_path, char=args.add_char)
    # print('test getting dev_insts time: ', time.time() - test_time)

    # # data.fix_alphabet()
    # test_insts, test_insts_index = data.get_instance(config.test_file, config.run_insts, config.shrink_feature_thresholds, write_word_path, label_path, pos_path, parse_path, category_path, char=args.add_char)
    # print('test getting test_insts time: ', time.time() - test_time)


    # vocab = vocab_base.Vocab('examples/vocab.txt')
    # data.vacab = vocab.labelToIdx
    # data.n_embed = len(vocab.labelToIdx)
    # train_dir = "examples/train/"
    # dev_dir = "examples/test/"
    # test_dir = "examples/test/"
    #
    # train_dataset = data_utils.DataUtils.build_deptree(train_dir + "sent.txt", train_dir + "label.txt", train_dir + "parse.txt", train_dir + "category.txt", train_dir + "pos.txt", vocab, config, data)
    # dev_dataset = data_utils.DataUtils.build_deptree(train_dir + "sent.txt", train_dir + "label.txt", train_dir + "parse.txt", train_dir + "category.txt", train_dir + "pos.txt", vocab, config, data)
    # test_dataset = data_utils.DataUtils.build_deptree(train_dir + "sent.txt", train_dir + "label.txt", train_dir + "parse.txt", train_dir + "category.txt", train_dir + "pos.txt", vocab, config, data)

    if config.pretrained_wordEmb_file != '':
        data.norm_word_emb = False
        data.build_word_pretrain_emb(config.pretrained_wordEmb_file, config.word_dims)
    if config.pretrained_charEmb_file != '':
        data.norm_char_emb = False
        data.build_char_pretrain_emb(config.pretrained_charEmb_file, config.char_dims)

    if args.model == 'context_att':
        # model_att = context_att.Context_att(config, data)
        model_att = context_att_b.Context_att(config, data)
    elif args.model == 'vanilla':
        # model_att = vanilla_att.Vanilla_att(config, data)
        # model_att = vanilla_att_b_new.Vanilla_att(config, data)
        # model_att = vanilla_att_b_try.Vanilla_att(config, data)
        # model_e = lstm_e.LSTM(config, data)
        # add_layer = calc_loss.Calc_loss(config, data)
        model_att = bilstm.LSTM_att(config, data)
    elif args.model == 'context_att_gate':
        model_att = context_att_gate_b.Context_att_gate(config, data)
    elif args.model == 'treelstm':
        # model_att = treelstm.BatchChildSumTreeLSTM(config, data)
        model_att = DL4MT.Encoder(config, data)

    # print('test building model time: ', time.time() - test_time)

    if data.use_cuda: model_att = model_att.cuda()

    # train_att.train_att(train_insts, train_insts_index, dev_insts, dev_insts_index, test_insts, test_insts_index, model_att, config, data)
    # train_tree.train(train_insts, train_insts_index, dev_insts, dev_insts_index, test_insts, test_insts_index, model_att, config, data)

    # train_tree.train(train_dataset, dev_dataset, test_dataset, model_att, config, data)
    train_tree_nobatch.train_att(train_insts, train_insts_index, dev_insts, dev_insts_index, test_insts, test_insts_index, model_att, config, data)

