import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# from module import Init
# from module.Embeddings import Embeddings
# from module.CGRU import CGRUCell
from model_att.RNN import RNN
# from module.Utils import *
from model_att.treeGRU import *
from model_att.tree import *
# from data.Vocab import *

class Encoder(nn.Module):
    # def __init__(self, n_words, n_rels, input_size, hidden_size):
    def __init__(self, config, params):
        super(Encoder, self).__init__()
        # Use PAD
        # self.embedding = Embeddings(num_embeddings=n_words, embedding_dim=input_size, dropout=0.0, add_position_embedding=False, padding_idx=SRCVocab.PAD)
        #
        # self.rel_embedding = Embeddings(num_embeddings=n_rels, embedding_dim=input_size, dropout=0.0, add_position_embedding=False, padding_idx=SRCVocab.PAD)
        self.word_num = params.word_num
        self.label_num = params.label_num
        self.char_num = params.char_num
        self.category_num = params.category_num
        self.parent_num = params.parent_num
        self.rel_num = params.rel_num
        self.pos_num = params.pos_num

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
        self.dropout_lstm = nn.Dropout(p=config.dropout_lstm)

        self.lstm_layers = config.lstm_layers
        self.batch_size = config.train_batch_size

        self.embedding = nn.Embedding(self.word_num, self.word_dims)
        if params.pretrain_word_embedding is not None:
            # pretrain_weight = np.array(params.pretrain_word_embedding)
            # self.embedding.weight.data.copy_(torch.from_numpy(pretrain_weight))
            # pretrain_weight = np.array(params.pretrain_embed)
            pretrain_weight = torch.FloatTensor(params.pretrain_word_embedding)
            self.embedding.weight.data.copy_(pretrain_weight)

        self.label_embedding = nn.Embedding(self.label_num, self.word_dims)
        self.pos_embedding = nn.Embedding(self.pos_num, self.word_dims)
        self.rel_embedding = nn.Embedding(self.rel_num, self.word_dims)

        self.gru = RNN(type="gru", batch_first=True, input_size=self.word_dims*2, hidden_size=self.lstm_hiddens, bidirectional=True)

        self.transform = nn.Linear(in_features=2*self.lstm_hiddens, out_features=self.word_dims, bias=True)
        self.dt_tree = DTTreeGRU(2*self.word_dims, self.lstm_hiddens)
        self.td_tree = TDTreeGRU(2*self.word_dims, self.lstm_hiddens)

        self.fc = nn.Linear(self.lstm_hiddens*2, 100, bias=True)
        self.fc2 = nn.Linear(100, self.category_num, bias=True)

    def forward(self, xs, heads, rels, labels, poss, xlengths, target_start, target_end, x_mask):
        """ fea_v, parse_v, rel_v, batch_length.tolist(), start_v, end_v, label_v, mask_v
        :param x: Input sequence.
            with shape [batch_size, seq_len, input_size]
        """
        # x_mask = xs.data.eq(SRCVocab.PAD)
        word_emb = self.embedding(xs)
        label_emb = self.embedding(labels)
        # pos_emb = self.embedding(poss)
        # emb = torch.cat([word_emb, label_emb, pos_emb], 2)
        ##### emb: variable(batch_size, length, 3*word_dims)
        emb = torch.cat([word_emb, label_emb], 2)

        grus, _ = self.gru(emb, xlengths)
        grus = self.transform(grus)

        rel_emb = self.rel_embedding(rels)
        outputs = torch.cat([grus, rel_emb], 2)
        outputs = outputs.transpose(0, 1)

        max_length, batch_size, input_dim = outputs.size()

        trees = []
        indexes = np.zeros((max_length, batch_size), dtype=np.int32)
        for b, head in enumerate(heads):
            root, tree = creatTree(head)
            root.traverse()
            for step, index in enumerate(root.order):
                indexes[step, b] = index
            trees.append(tree)

        dt_outputs, dt_hidden_ts = self.dt_tree(outputs, indexes, trees, xlengths)
        td_outputs, td_hidden_ts = self.td_tree(outputs, indexes, trees, xlengths)

        ctx = torch.cat([dt_outputs, td_outputs], dim=2)
        # print(ctx)      # [torch.FloatTensor of size 1x35x400]
        ctx = ctx.transpose(1, 2)
        ctx = F.max_pool1d(ctx, ctx.size(2)).squeeze(2)

        ctx = self.fc(ctx)
        ctx = self.fc2(ctx)
        return ctx


# class Decoder(nn.Module):
#
#     def __init__(self,
#                  n_words,
#                  input_size,
#                  hidden_size,
#                  bridge_type="mlp",
#                  dropout_rate=0.0):
#
#         super(Decoder, self).__init__()
#
#         self.bridge_type = bridge_type
#         self.hidden_size = hidden_size
#         self.context_size = hidden_size * 2
#
#         self.embedding = Embeddings(num_embeddings=n_words,
#                                     embedding_dim=input_size,
#                                     dropout=0.0,
#                                     add_position_embedding=False,
#                                     padding_idx=TGTVocab.PAD)
#
#         self.cgru_cell = CGRUCell(input_size=input_size, hidden_size=hidden_size)
#
#         self.linear_input = nn.Linear(in_features=input_size, out_features=input_size)
#         self.linear_hidden = nn.Linear(in_features=hidden_size, out_features=input_size)
#         self.linear_ctx = nn.Linear(in_features=hidden_size * 2, out_features=input_size)
#
#         self.dropout = nn.Dropout(dropout_rate)
#
#         self._reset_parameters()
#
#         self._build_bridge()
#
#     def _reset_parameters(self):
#         Init.default_init(self.linear_input.weight.data)
#         Init.default_init(self.linear_hidden.weight.data)
#         Init.default_init(self.linear_ctx.weight.data)
#
#     def _build_bridge(self):
#         if self.bridge_type == "mlp":
#             self.linear_bridge = nn.Linear(in_features=self.context_size, out_features=self.hidden_size)
#             Init.default_init(self.linear_bridge.weight.data)
#         elif self.bridge_type == "zero":
#             pass
#         else:
#             raise ValueError("Unknown bridge type {0}".format(self.bridge_type))
#
#     def init_decoder(self, context, mask):
#
#         # Generate init hidden
#         if self.bridge_type == "mlp":
#             no_pad_mask = Variable((1.0 - mask.float()))
#             ctx_mean = (context * no_pad_mask.unsqueeze(2)).sum(1) / no_pad_mask.unsqueeze(2).sum(1)
#             dec_init = F.tanh(self.linear_bridge(ctx_mean))
#         elif self.bridge_type == "zero":
#             batch_size = context.size(0)
#             dec_init = Variable(context.data.new(batch_size, self.hidden_size).zero_())
#         else:
#             raise ValueError("Unknown bridge type {0}".format(self.bridge_type))
#
#         dec_cache = self.cgru_cell.compute_cache(context)
#
#         return dec_init, dec_cache
#
#
#     def forward(self, y, context, context_mask, hidden, one_step=False, cache=None):
#
#         emb = self.embedding(y) # [seq_len, batch_size, dim]
#
#         if one_step:
#             (out, attn), hidden = self.cgru_cell(emb, hidden, context, context_mask, cache)
#         else:
#             # emb: [seq_len, batch_size, dim]
#             out = []
#             attn = []
#
#             for emb_t in torch.split(emb, split_size=1, dim=0):
#
#                 (out_t, attn_t), hidden = self.cgru_cell(emb_t.squeeze(0), hidden, context, context_mask, cache)
#                 out += [out_t]
#                 attn += [attn_t]
#
#             out = torch.stack(out)
#             attn = torch.stack(attn)
#
#         logits = self.linear_input(emb) + self.linear_hidden(out) + self.linear_ctx(attn)
#
#         logits = F.tanh(logits)
#
#         logits = self.dropout(logits) # [seq_len, batch_size, dim]
#
#         return logits, hidden

# class Generator(nn.Module):
#
#     def __init__(self, n_words, hidden_size, shared_weight=None, padding_idx=-1):
#
#         super(Generator, self).__init__()
#
#         self.n_words = n_words
#         self.hidden_size = hidden_size
#         self.padding_idx = padding_idx
#
#         self.proj = nn.Linear(self.hidden_size, self.n_words, bias=False)
#         self.actn = nn.LogSoftmax(dim=-1)
#
#         if shared_weight is not None:
#             self.proj.weight = shared_weight
#         else:
#             self._reset_parameters()
#
#     def _reset_parameters(self):
#
#         Init.embedding_init(self.proj.weight)
#
#
#     def forward(self, input):
#         """
#         input == > Linear == > LogSoftmax
#         """
#         return self.actn(self.proj(input))

# class DL4MT(nn.Module):
#     def __init__(self, config, n_src_vocab, n_rel_vocab, n_tgt_vocab, use_gpu=True):
#
#         super().__init__()
#
#         self.config = config
#
#         self.encoder = Encoder(n_words=n_src_vocab, n_rels=n_rel_vocab, input_size=config.embed_size, \
#                                hidden_size=config.hidden_size)
#
#         # self.decoder = Decoder(n_words=n_tgt_vocab, input_size=config.embed_size, hidden_size=config.hidden_size,
#         #                        dropout_rate=config.dropout_hidden, bridge_type=config.bridge_type)
#
#         # if config.proj_share_weight is False:
#         #     generator = Generator(n_words=n_tgt_vocab, hidden_size=config.embed_size, padding_idx=TGTVocab.PAD)
#         # else:
#         #     generator = Generator(n_words=n_tgt_vocab, hidden_size=config.embed_size, padding_idx=TGTVocab.PAD,
#         #                           shared_weight=self.decoder.embedding.embeddings.weight
#         #                           )
#         # self.generator = generator
#         self.use_gpu = use_gpu
#
#         for p in self.parameters():
#             nn.init.uniform(p.data, -config.param_init, config.param_init)
#
#     def force_teaching(self, x, rel, head, y, lengths):
#
#         ctx, ctx_mask = self.encoder(x, rel, head, lengths)
#
#         dec_init, dec_cache = self.decoder.init_decoder(ctx, ctx_mask)
#
#         logits, _ = self.decoder(y,
#                                  context=ctx,
#                                  context_mask=ctx_mask,
#                                  one_step=False,
#                                  hidden=dec_init,
#                                  cache=dec_cache) # [tgt_len, batch_size, dim]
#
#         return logits.transpose(1, 0).contiguous() # Convert to batch-first mode.
#
#     def batch_beam_search(self, x, rel, head, lengths, beam_size=5, max_steps=150):
#
#         batch_size = x.size(0)
#
#         ctx, ctx_mask = self.encoder(x, rel, head, lengths)
#         dec_init, dec_cache = self.decoder.init_decoder(ctx, ctx_mask)
#
#         ctx = tile_batch(ctx, multiplier=beam_size, batch_dim=0)
#         dec_cache = tile_batch(dec_cache, multiplier=beam_size, batch_dim=0)
#         hiddens = tile_batch(dec_init, multiplier=beam_size, batch_dim=0).data
#         ctx_mask = tile_batch(ctx_mask, multiplier=beam_size, batch_dim=0)
#
#         beam_mask = ctx_mask.new(batch_size, beam_size).fill_(1).float()
#         dec_memory_len = ctx_mask.new(batch_size, beam_size).zero_().float()
#         beam_scores = ctx_mask.new(batch_size, beam_size).zero_().float()
#         final_word_indices = x.data.new(batch_size, beam_size, 1).fill_(TGTVocab.BOS)
#
#         for t in range(max_steps):
#             logits, hiddens = self.decoder(y=Variable(final_word_indices[:,:,-1].contiguous().view(batch_size * beam_size, ), volatile=True),
#                                            hidden=Variable(hiddens.view(batch_size * beam_size, -1), volatile=True),
#                                            context=ctx,
#                                            context_mask=ctx_mask,
#                                            one_step=True,
#                                            cache=dec_cache
#                                            )
#
#             hiddens = hiddens.view(batch_size, beam_size, -1).data
#
#             next_scores = - self.generator(logits)  # [B * Bm, N]
#             next_scores = next_scores.view(batch_size, beam_size, -1).data
#             next_scores = mask_scores(next_scores, beam_mask=beam_mask)
#
#             beam_scores = next_scores + beam_scores.unsqueeze(2) # [B, Bm, N] + [B, Bm, 1] ==> [B, Bm, N]
#
#             vocab_size = beam_scores.size(-1)
#             if t == 0:
#                 beam_scores = beam_scores[:,0,:].contiguous()
#
#             beam_scores = beam_scores.view(batch_size, -1)
#
#             # Get topK with beams
#             beam_scores, indices = torch.topk(beam_scores, k=beam_size, dim=-1, largest=False, sorted=False)
#             next_beam_ids = torch.div(indices, vocab_size)
#             next_word_ids = indices % vocab_size
#
#             # gather beam cache
#             dec_memory_len = tensor_gather_helper(gather_indices=next_beam_ids,
#                                                   gather_from=dec_memory_len,
#                                                   batch_size=batch_size,
#                                                   beam_size=beam_size,
#                                                   gather_shape=[-1],
#                                                   use_gpu=self.use_gpu)
#
#             hiddens = tensor_gather_helper(gather_indices=next_beam_ids,
#                                            gather_from=hiddens,
#                                            batch_size=batch_size,
#                                            beam_size=beam_size,
#                                            gather_shape=[batch_size * beam_size, -1],
#                                            use_gpu=self.use_gpu)
#
#             beam_mask = tensor_gather_helper(gather_indices=next_beam_ids,
#                                              gather_from=beam_mask,
#                                              batch_size=batch_size,
#                                              beam_size=beam_size,
#                                              gather_shape=[-1],
#                                              use_gpu=self.use_gpu)
#
#             final_word_indices = tensor_gather_helper(gather_indices=next_beam_ids,
#                                                       gather_from=final_word_indices,
#                                                       batch_size=batch_size,
#                                                       beam_size=beam_size,
#                                                       gather_shape=[batch_size * beam_size, -1],
#                                                       use_gpu=self.use_gpu)
#
#             # If next_word_ids is EOS, beam_mask_ should be 0.0
#             beam_mask_ = 1.0 - next_word_ids.eq(TGTVocab.EOS).float()
#             next_word_ids.masked_fill_((beam_mask_ + beam_mask).eq(0.0), TGTVocab.PAD) # If last step a EOS is already generated, we replace the last token as PAD
#             beam_mask = beam_mask * beam_mask_
#
#             # update beam
#             dec_memory_len += beam_mask
#
#             final_word_indices = torch.cat((final_word_indices, torch.unsqueeze(next_word_ids, 2)), dim=2)
#
#             if beam_mask.eq(0.0).all():
#                 # All the beam is finished (be zero
#                 break
#
#         scores = beam_scores / (dec_memory_len + 1e-2)
#
#         _, reranked_ids = torch.sort(scores, dim=-1, descending=False)
#
#         return tensor_gather_helper(gather_indices=reranked_ids,
#                                     gather_from=final_word_indices[:, :, 1:].contiguous(),
#                                     batch_size=batch_size,
#                                     beam_size=beam_size,
#                                     gather_shape=[batch_size * beam_size, -1],
#                                     use_gpu=self.use_gpu)
#
#     def forward(self, src_seq, src_rel, src_head, tgt_seq=None, mode="train", **kwargs):
#         if mode == "train":
#             assert tgt_seq is not None
#             tgt_seq = tgt_seq.transpose(1, 0).contiguous() # length first
#             return self.force_teaching(src_seq, src_rel, src_head, tgt_seq, **kwargs)
#
#         elif mode == "infer":
#             return self.batch_beam_search(x=src_seq, rel=src_rel, head=src_head, **kwargs)