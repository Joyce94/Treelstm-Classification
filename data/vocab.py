from data.alphabet import Alphabet
from collections import Counter
import data.utils as utils
import math
import model_joint.evaluation as evaluation

class Data():
    def __init__(self):
        self.word_alphabet = Alphabet('word')
        self.category_alphabet = Alphabet('category', is_category=True)
        self.label_alphabet = Alphabet('label', is_label=True)
        self.char_alphabet = Alphabet('char')
        self.pos_alphabet = Alphabet('pos')
        self.parent_alphabet = Alphabet('parent')
        self.rel_alphabet = Alphabet('rel')

        self.number_normalized = True
        self.norm_word_emb = False
        self.norm_char_emb = False

        self.pretrain_word_embedding = None
        self.pretrain_char_embedding = None

        self.max_char_length = 0

        self.word_num = 0
        self.char_num = 0
        self.label_num = 0
        self.category_num = 0
        self.parent_num = 0
        self.pos_num = 0
        self.rel_num = 0


    def build_alphabet(self, word_counter, label_counter, category_counter, pos_counter, rel_counter, shrink_feature_threshold, char=False):
        for word, count in word_counter.most_common():
            if count > shrink_feature_threshold:
                self.word_alphabet.add(word, count)
        for label, count in label_counter.most_common():
            self.label_alphabet.add(label, count)
        for category, count in category_counter.most_common():
            self.category_alphabet.add(category, count)
        for pos, count in pos_counter.most_common():
            self.pos_alphabet.add(pos, count)
        # for parent, count in parent_counter.most_common():
        #     self.parent_alphabet.add(parent, count)
        # print(rel_counter)
        for rel, count in rel_counter.most_common():
            self.rel_alphabet.add(rel, count)


    def fix_alphabet(self):
        self.word_num = self.word_alphabet.close()
        self.category_num = self.category_alphabet.close()
        self.label_num = self.label_alphabet.close()
        # self.category_num = self.category_alphabet.close()
        self.parent_num = self.parent_alphabet.close()
        self.pos_num = self.pos_alphabet.close()
        self.rel_num = self.rel_alphabet.close()


    def get_instance(self, file, run_insts, shrink_feature_threshold, char=False, char_padding_symbol='<pad>'):
        words = []
        labels = []
        categorys = []
        poss = []
        parents = []
        rels = []
        insts = []
        word_counter = Counter()
        label_counter = Counter()
        category_counter = Counter()
        pos_counter = Counter()
        # parent_counter = Counter()
        rel_counter = Counter()

        count = 0
        with open(file, 'r', encoding='utf-8') as f:
            for id, line in enumerate(f.readlines()):
                if run_insts == count: break
                if len(line) > 2:
                    line = line.strip().split(' ')
                    if '' in line: line.remove('')
                    if len(line) != 6: print(id)
                    # print(line)
                    word = line[0]
                    if self.number_normalized: word = utils.normalize_word(word)
                    label = line[1]
                    category = line[2]
                    pos = line[3]
                    parent = line[-2]
                    if ',' in parent:
                        # print(parent)
                        parent = parent.split(',')[0]
                    # if parent == '':
                    #     print(id)
                    rel = line[-1]

                    words.append(word)
                    labels.append(label)
                    categorys.append(category)
                    poss.append(pos)
                    parents.append(parent)
                    rels.append(rel)

                    word_counter[word] += 1        #####
                    label_counter[label] += 1
                    category_counter[category] += 1
                    pos_counter[pos] += 1
                    # parent_counter[category] += 1
                    rel_counter[rel] += 1
                else:
                    # print(words)
                    # print(parents)
                    insts.append([words, labels, categorys, poss, parents, rels])
                    words = []
                    labels = []
                    categorys = []
                    poss = []
                    parents = []
                    rels = []
                    count += 1
        if not self.word_alphabet.fix_flag:
            self.build_alphabet(word_counter, label_counter, category_counter, pos_counter, rel_counter, shrink_feature_threshold, char)
        insts_index = []

        for inst in insts:
            words_index = [self.word_alphabet.get_index(w) for w in inst[0]]
            labels_index = [self.label_alphabet.get_index(l) for l in inst[1]]
            categorys_index = [self.category_alphabet.get_index(c) for c in inst[2]]
            poss_index = [self.pos_alphabet.get_index(p) for p in inst[3]]
            # parents_index = [self.parent_alphabet.get_index(p) for p in inst[-2]]
            parents_index = [int(p)-1 for p in inst[-2]]
            rels_index = [self.rel_alphabet.get_index(r) for r in inst[-1]]
            insts_index.append([words_index, labels_index, categorys_index, poss_index, parents_index, rels_index])
        return insts, insts_index


    def get_instance_tree(self, file, run_insts, shrink_feature_threshold, write_word_path, label_path, pos_path, parse_path, category_path, char=False, char_padding_symbol='<pad>'):
        words = []
        labels = []
        categorys = []
        poss = []
        parses = []
        insts = []
        word_counter = Counter()
        label_counter = Counter()
        category_counter = Counter()
        pos_counter = Counter()
        parse_counter = Counter()

        word_file = open(write_word_path, 'w', encoding='utf-8')
        label_file = open(label_path, 'w', encoding='utf-8')
        pos_file = open(pos_path, 'w', encoding='utf-8')
        parse_file = open(parse_path, 'w', encoding='utf-8')
        category_file = open(category_path, 'w', encoding='utf-8')

        count = 0
        with open(file, 'r', encoding='utf-8') as f:
            for id, line in enumerate(f.readlines()):
                if run_insts == count: break
                if len(line) > 2:
                    line = line.strip().split(' ')
                    # print(line)
                    word = line[0]
                    if self.number_normalized: word = utils.normalize_word(word)
                    label = line[1]
                    category = line[2]
                    pos = line[-2]
                    parse = line[-1]

                    words.append(word)
                    labels.append(label)
                    categorys.append(category)
                    parses.append(parse)
                    poss.append(pos)

                    word_counter[word] += 1        #####
                    label_counter[label] += 1
                    category_counter[category] += 1
                    pos_counter[pos] += 1
                    parse_counter[category] += 1
                else:
                    # print(words)
                    word_write = ' '.join(words)
                    word_file.write(word_write)
                    word_file.write('\n')
                    parse_write = ' '.join(parses)
                    parse_file.write(parse_write)
                    parse_file.write('\n')

                    insts.append([words, labels, categorys, poss, parses])
                    words = []
                    labels = []
                    categorys = []
                    poss = []
                    parses = []
                    count += 1
        if not self.word_alphabet.fix_flag:
            self.build_alphabet(word_counter, label_counter, category_counter, pos_counter, parse_counter, shrink_feature_threshold, char)
        insts_index = []

        path = r"C:\Users\song\Desktop\treelstm_word\examples\vocab-2.txt"
        file = open(path, 'w', encoding='utf-8')
        words = self.word_alphabet.id2word
        # print(len(words))       # 6799, 6310
        for id in range(len(words)):
            file.write(words[id])
            file.write('\n')
        file.close()

        for inst in insts:
            words_index = [self.word_alphabet.get_index(w) for w in inst[0]]
            labels_index = [str(self.label_alphabet.get_index(l)) for l in inst[1]]
            categorys_index = [str(self.category_alphabet.get_index(c)) for c in inst[2]]
            pos_index = [str(self.pos_alphabet.get_index(p)) for p in inst[-2]]
            parses_index = [self.parse_alphabet.get_index(p) for p in inst[-1]]
            insts_index.append([words_index, labels_index, categorys_index, pos_index, parses_index])

            label_write = ' '.join(labels_index)
            label_file.write(label_write)
            label_file.write('\n')
            pos_write = ' '.join(pos_index)
            pos_file.write(pos_write)
            pos_file.write('\n')
            category_write = ' '.join(categorys_index)
            category_file.write(category_write)
            category_file.write('\n')
        return insts, insts_index


    def build_word_pretrain_emb(self, emb_path, word_dims):
        self.pretrain_word_embedding = utils.load_pretrained_emb_avg(emb_path, self.word_alphabet.word2id, word_dims, self.norm_word_emb)

    def build_char_pretrain_emb(self, emb_path, char_dims):
        self.pretrain_char_embedding = utils.load_pretrained_emb_avg(emb_path, self.char_alphabet.word2id, char_dims, self.norm_char_emb)


    def generate_batch_buckets(self, batch_size, insts, char=False):
        batch_num = int(math.ceil(len(insts) / batch_size))
        buckets = [[[], [], [], [], [], []] for _ in range(batch_num)]
        labels_raw = [[] for _ in range(batch_num)]
        category_raw = [[] for _ in range(batch_num)]
        target_start = [[] for _ in range(batch_num)]
        target_end = [[] for _ in range(batch_num)]

        inst_save = []
        for id, inst in enumerate(insts):
            idx = id // batch_size
            if id == 0 or id % batch_size != 0:
                inst_save.append(inst)
            elif id % batch_size == 0:
                assert len(inst_save) == batch_size
                inst_sorted = utils.sorted_instances_index(inst_save)
                max_length = len(inst_sorted[0][0])
                for idy in range(batch_size):
                    cur_length = len(inst_sorted[idy][0])
                    buckets[idx-1][0].append(inst_sorted[idy][0] + [self.word_alphabet.word2id['<pad>']] * (max_length - cur_length))
                    buckets[idx-1][1].append(inst_sorted[idy][1] + [self.label_alphabet.word2id['<pad>']] * (max_length - cur_length))
                    buckets[idx-1][2].append(inst_sorted[idy][-3] + [self.pos_alphabet.word2id['<pad>']] * (max_length - cur_length))
                    buckets[idx-1][3].append(inst_sorted[idy][-2] + [self.parent_alphabet.word2id['<pad>']] * (max_length - cur_length))
                    buckets[idx-1][4].append(inst_sorted[idy][-1] + [self.rel_alphabet.word2id['<pad>']] * (max_length - cur_length))
                    buckets[idx-1][-1].append([1] * cur_length + [0] * (max_length - cur_length))
                    labels_raw[idx-1].append(inst_sorted[idy][1])

                    start, end = evaluation.extract_target(inst_sorted[idy][1], self.label_alphabet)
                    if start == []:
                        start = [0]
                        end = [0]
                    target_start[idx-1].append(start[0])
                    target_end[idx-1].append(end[0])
                    # target_start.extend(start)
                    # target_end.extend(end)
                    category_raw[idx-1].append(inst_sorted[idy][2][0])
                inst_save = []
                inst_save.append(inst)
        if inst_save != []:
            inst_sorted = utils.sorted_instances_index(inst_save)
            max_length = len(inst_sorted[0][0])
            for idy in range(len(inst_sorted)):
                cur_length = len(inst_sorted[idy][0])
                buckets[batch_num-1][0].append(inst_sorted[idy][0] + [self.word_alphabet.word2id['<pad>']] * (max_length - cur_length))
                buckets[batch_num-1][1].append(inst_sorted[idy][1] + [self.label_alphabet.word2id['<pad>']] * (max_length - cur_length))
                buckets[batch_num-1][2].append(inst_sorted[idy][-3] + [self.pos_alphabet.word2id['<pad>']] * (max_length - cur_length))
                buckets[batch_num-1][3].append(inst_sorted[idy][-2] + [self.parent_alphabet.word2id['<pad>']] * (max_length - cur_length))
                buckets[batch_num-1][4].append(inst_sorted[idy][-1] + [self.rel_alphabet.word2id['<pad>']] * (max_length - cur_length))
                buckets[batch_num-1][-1].append([1] * cur_length + [0] * (max_length - cur_length))
                labels_raw[batch_num-1].append(inst_sorted[idy][1])
                category_raw[batch_num-1].append(inst_sorted[idy][2][0])
                start, end = evaluation.extract_target(inst_sorted[idy][1], self.label_alphabet)
                if start == []:
                    start = [0]
                    end = [0]
                target_start[batch_num - 1].append(start[0])
                target_end[batch_num - 1].append(end[0])
        # print(buckets)
        # print(labels_raw)
        # print(category_raw)
        # print(target_start)
        # print(target_end)
        return buckets, labels_raw, category_raw, target_start, target_end

    def generate_batch_buckets_save(self, batch_size, insts, char=False):
        # insts_length = list(map(lambda t: len(t) + 1, inst[0] for inst in insts))
        # insts_length = list(len(inst[0]+1) for inst in insts)
        # if len(insts) % batch_size == 0:
        #     batch_num = len(insts) // batch_size
        # else:
        #     batch_num = len(insts) // batch_size + 1
        batch_num = int(math.ceil(len(insts) / batch_size))

        if char:
            buckets = [[[], [], [], []] for _ in range(batch_num)]
        else:
            buckets = [[[], [], []] for _ in range(batch_num)]
        max_length = 0
        for id, inst in enumerate(insts):
            idx = id // batch_size
            if id % batch_size == 0:
                max_length = len(inst[0]) + 1
            cur_length = len(inst[0])

            buckets[idx][0].append(inst[0] + [self.word_alphabet.word2id['<pad>']] * (max_length - cur_length))
            buckets[idx][1].append([self.label_alphabet.word2id['<start>']] + inst[-1] + [self.label_alphabet.word2id['<pad>']] * (max_length - cur_length - 1))
            if char:
                char_length = len(inst[1][0])
                buckets[idx][2].append((inst[1] + [[self.char_alphabet.word2id['<pad>']] * char_length] * (max_length - cur_length)))
            buckets[idx][-1].append([1] * (cur_length + 1) + [0] * (max_length - (cur_length + 1)))

        return buckets











