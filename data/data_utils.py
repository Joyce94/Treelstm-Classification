import re
import random
from nltk.parse import stanford
import time
import sys
import torch
import torch.autograd as autograd
from tqdm import tqdm

random.seed(5)

class Instance:
    def __init__(self):
        self.sentence = ''
        self.label = '-1'

    def show(self):
        print(self.sentence, ' ', self.label)


class Code:
    def __init__(self):
        self.code_list = []
        self.label = []

    def show(self):
        print(self.code_list, ' ', self.label)


class Node:
    def __init__(self):
        self.word = ""
        self.word_index = ''
        self.parent_index = ''
        self.label = ''
        self.flag = False
        self.relation = ''
        self.category = ''
        self.pos = ''

class Tree:
    def __init__(self, value, label, level, word, word_idx, category, pos, config, params):
        self.value = value
        self.children = []
        self.state = (torch.zeros(1, config.lstm_hiddens), torch.zeros(1, config.lstm_hiddens))  # save c and h
        self.label = label
        self.category = category
        self.pos = pos
        self.level = level  # depth of tree
        self.step = 0
        self.forest_ix = 0
        self.loss = 0
        self.out = 0
        self.fc = torch.zeros(1, config.lstm_hiddens)
        self.word = word
        self.word_idx = word_idx
        self.mark = 0
        self.tree_idx = 0
        self.config = config
        self.params = params

    def add_child(self, child_tree):
        self.children.append(child_tree)


class Process:
    def __init__(self, path=None, sst=False, clean=False):
        self.result = []
        self.load_file(path, sst, clean)

    def load_file(self, path, sst, clean):
        with open(path, 'r') as f:
            for line in f:
                if clean:
                    if sst:
                        line = DataUtils.clean_str_sst(line)
                    else:
                        line = DataUtils.clean_str(line)
                info = line.split(' ', 1)
                assert len(info) == 2
                inst = Instance()
                inst.sentence = info[1].split()
                inst.label = info[0]
                self.result.append(inst)


class DataUtils:
    @staticmethod
    def create_voca(result):
        vocabulary = {}
        for r in result:
            for s in r.sentence:
                if s not in vocabulary.keys():
                    vocabulary[s] = len(vocabulary) + 1
                else:
                    pass
        vocabulary['-unknown-'] = len(vocabulary) + 1
        vocabulary['-padding-'] = 0
        return vocabulary

    @staticmethod
    def cross_validation(path, packet_nums, encoding='UTF-8', clean_switch=False):
        result = []
        packet_list = []
        with open(path, 'r', encoding=encoding) as fin:
            for line in fin:
                if clean_switch:
                    line = DataUtils.clean_str(line)
                info = line.split(' ', 1)
                inst = Instance()
                # print(line)
                assert len(info) == 2
                inst.sentence = info[1].split()
                inst.label = info[0]
                result.append(inst)
        random.shuffle(result)
        length = len(result)
        packet_len = length // packet_nums
        if length % packet_nums != 0:
            packet_len += 1
        packet = []
        for i, r in enumerate(result):
            if i % packet_len == 0 and i != 0:
                packet_list.append(packet)
                packet = [r]
            else:
                packet.append(r)
        if len(packet) > 0:
            packet_list.append(packet)
        return packet_list

    @staticmethod
    def read_data(path, clean_switch=False):
        result = []
        with open(path, 'r') as fin:
            for line in fin:
                info = line.split('|||')
                assert len(info) == 2
                if clean_switch:
                    info[0] = DataUtils.clean_str(info[0])
                    info[1] = DataUtils.clean_str(info[1])
                inst = Instance()
                inst.sentence = info[0].split()
                inst.label = info[1]
                result.append(inst)
        return result

    @staticmethod
    def clean_str(string):
        """
                Tokenization/string cleaning for all datasets except for SST.
                Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
            """
        # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    @staticmethod
    def clean_str_sst(string):
        """
            Tokenization/string cleaning for the SST dataset
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    @staticmethod
    def generate_sst_sen(path, result):
        print("w")
        with open(path, 'w') as fout:
            for r in result:
                fout.write(' '.join(r.sentence) + '\n')

    @staticmethod
    def encode(result, vocabulary):
        encodes = []
        for r in result:
            code = Code()
            for s in r.sentence:
                if s in vocabulary.keys():
                    code.code_list.append(vocabulary[s])
                else:
                    code.code_list.append(vocabulary['-unknown-'])
            if r.label == '0':
                code.label = 0
            elif r.label == '1':
                code.label = 1
            elif r.label == '2':
                code.label = 2
            elif r.label == '3':
                code.label = 3
            elif r.label == '4':
                code.label = 4
            else:
                raise RuntimeError("label index out of process ")
            if len(code.code_list) <=2:
                continue
            encodes.append(code)
        return encodes

    @staticmethod
    def reverse(string):
        return string[::-1]

    @staticmethod
    def build_tree_conll(file_name, result, vocab):
        trees = []
        i = 0
        with open(file_name, 'r') as fin:
            nodes = []
            for line in fin:
                if line != '\n':
                    # print(line)
                    lines = line.strip().split()
                    node = Node()
                    node.word = lines[1]
                    node.word_index = int(lines[0])
                    # node.parent_index = int(lines[5])
                    # node.relation = lines[6]
                    node.parent_index = int(lines[6])
                    node.relation = lines[7]
                    node.label = int(result[i].label)
                    nodes.append(node)
                else:
                    sentence = result[i].sentence
                    # print(sentence)
                    for node in nodes:
                        if node.relation == "root":
                            if sentence[node.word_index - 1] in vocab:
                                word_idx = vocab[sentence[node.word_index - 1]]
                            else:
                                word_idx = vocab['-unknown-']
                            tree = Tree(node.word_index, node.label, 0, sentence[node.word_index - 1], word_idx)
                            child_value = node.word_index
                            node.flag = True
                            DataUtils.add_tree(tree, child_value, nodes, 1, vocab)
                            trees.append(tree)
                    nodes = []
                    i += 1
        return trees

    # @staticmethod
    # def add_tree(tree, child_value, nodes, level, sentences, vocab):
    #     for node in nodes:
    #         if child_value == node.parent_index and node.flag is False:
    #             if sentences[node.word_index - 1] in vocab:
    #                 word_idx = vocab[sentences[node.word_index - 1]]
    #             else:
    #                 word_idx = vocab['-unknown-']
    #             child_tree = Tree(node.word_index, node.label, level, sentences[node.word_index - 1], word_idx)
    #             tree.add_child(child_tree)
    #             node_index = node.word_index
    #             # nodes.remove(node)
    #             node.flag = True
    #             DataUtils.add_tree(child_tree, node_index, nodes, level + 1, sentences, vocab)

    @staticmethod
    def add_tree(tree, child_value, nodes, level,  vocab, config, params):
        for node in nodes:
            if child_value == node.parent_index and node.flag is False:
                if node.word in vocab:
                    word_idx = vocab[node.word]
                else:
                    word_idx = vocab['-unknown-']
                child_tree = Tree(node.word_index, node.label, level, node.word, word_idx, node.category, node.pos, config, params)
                tree.add_child(child_tree)
                node_index = node.word_index
                node.flag = True
                DataUtils.add_tree(child_tree, node_index, nodes, level + 1, vocab, config, params)

    @staticmethod
    def build_deptree(sen_file, label_file, parent_file, category_file, pos_file, vocabulary, config, params):
        vocab = vocabulary.labelToIdx
        with open(sen_file, 'r', encoding='utf-8') as sentences, open(label_file, 'r', encoding='utf-8') as labels, open(parent_file, 'r', encoding='utf-8') as parents, open(category_file, 'r', encoding='utf-8') as categorys, open(pos_file, 'r', encoding='utf-8') as poss:
            sen = sentences.readlines()
            lab = labels.readlines()
            par = parents.readlines()
            cate = categorys.readlines()
            pos = poss.readlines()

            dep = zip(sen, lab, par, cate, pos)
            trees = []
            sentences = []
            for sen,lab,par,cate,pos in tqdm(dep):
                tree = DataUtils.read_tree(sen.strip(),lab.strip(),par.strip(),cate.strip(),pos.strip(),vocab,config,params)
                # if tree.label != 1:
                trees.append(tree)
                sentences.append(vocabulary.convertToIdx(sen.strip().split(),'<unk>'))
                # print(len(sentences[-1]))
            return trees,sentences

    @staticmethod
    def read_tree(sentences,labels,parents,category,poss,vocab,config,params):
        sens = sentences.split()
        labs = labels.split()
        pars = parents.split()
        cat = category.split()
        pos = poss.split()

        nodes = []
        tree = None
        for idx in range(len(sens)):
            node = Node()
            node.word = sens[idx]
            node.parent_index = int(pars[idx])
            # node.label = DataUtils.parse_dlabel_token(labs[idx],False) # binary set False,fine set True
            node.label = int(labs[idx])
            node.category = int(cat[idx])
            node.pos = int(pos[idx])
            node.word_index = idx+1
            nodes.append(node)
        for n in nodes:
            if n.parent_index == 0:
                if n.word in vocab:
                    word_idx = vocab[n.word]
                else:
                    word_idx = vocab['<unk>']    # -unknown-
                tree = Tree(n.word_index, n.label, 0, n.word, word_idx, n.category, n.pos, config, params)
                child_value = n.word_index
                n.flag = True
                DataUtils.add_tree(tree, child_value, nodes, 1, vocab, config, params)
        if tree is None: print(nodes)
        return tree

    @staticmethod
    def parse_dlabel_token(x,fine_grain=False):
        if x == '#':
            return None
        else:
            if fine_grain: # -2 -1 0 1 2 => 0 1 2 3 4
                return int(x)+2
            else: # # -2 -1 0 1 2 => 0 1 2
                tmp = int(x)
                if tmp < 0:
                    return 0
                elif tmp == 0:
                    return 1
                elif tmp >0 :
                    return 2

    @staticmethod
    def get_sentiment_dict(labels_file,dict_file):
        labels = []
        with open(labels_file,'r') as labelsfile:
            labelsfile.readline()
            for line in labelsfile:
                idx, rating = line.split('|')
                idx = int(idx)
                rating = float(rating)
                if rating <= 0.2:
                    label = -2
                elif rating <= 0.4:
                    label = -1
                elif rating > 0.8:
                    label = +2
                elif rating > 0.6:
                    label = +1
                else:
                    label = 0
                labels.append(label)

        d = {}
        with open(dict_file,'r') as dictionary:
            for line in dictionary:
                s, idx = line.split('|')
                d[s] = labels[int(idx)]
        return d