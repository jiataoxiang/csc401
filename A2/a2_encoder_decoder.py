'''
This code is provided solely for the personal and private use of students
taking the CSC401H/2511H course at the University of Toronto. Copying for
purposes other than this use is expressly prohibited. All forms of
distribution of this code, including but not limited to public repositories on
GitHub, GitLab, Bitbucket, or any other online platform, whether as given or
with any changes, are expressly prohibited.

Authors: Sean Robertson, Jingcheng Niu, Zining Zhu, and Mohamed Abdall

All of the files in this directory and all subdirectories are:
Copyright (c) 2021 University of Toronto
'''

'''Concrete implementations of abstract base classes.

You don't need anything more than what's been imported here
'''

import torch

from a2_abcs import EncoderBase, DecoderBase, EncoderDecoderBase


# All docstrings are omitted in this file for simplicity. So please read
# a2_abcs.py carefully so that you can have a solid understanding of the
# structure of the assignment.

class Encoder(EncoderBase):

    def init_submodules(self):
        # Hints:
        # 1. You must initialize the following submodules:
        #   self.rnn, self.embedding
        # 2. You will need the following object attributes:
        #   self.source_vocab_size, self.word_embedding_size,
        #   self.pad_id, self.dropout, self.cell_type,
        #   self.hidden_state_size, self.num_hidden_layers.
        # 3. cell_type will be one of: ['lstm', 'gru', 'rnn']
        # 4. Relevant pytorch modules: torch.nn.{LSTM, GRU, RNN, Embedding}
        # assert False, "Fill me"

        self.embedding = torch.nn.Embedding(num_embeddings=self.source_vocab_size,
                                            embedding_dim=self.word_embedding_size,
                                            padding_idx=self.pad_id)
        self.rnn = eval("torch.nn." + self.cell_type.upper() + "(" +
                        "input_size=self.word_embedding_size, " +
                        "hidden_size=self.hidden_state_size, " +
                        "num_layers=self.num_hidden_layers, " +
                        "dropout=self.dropout, bidirectional=True" +
                        ")")

    def forward_pass(self, F, F_lens, h_pad=0.):
        # Recall:
        #   F is size (S, M)
        #   F_lens is of size (M,)
        #   h_pad is a float
        #
        # Hints:
        # 1. The structure of the encoder should be:
        #   input seq -> |embedding| -> embedded seq -> |rnn| -> seq hidden
        # 2. You will need to use the following methods:
        #   self.get_all_rnn_inputs, self.get_all_hidden_states
        # assert False, "Fill me"
        rnn_in = self.get_all_rnn_inputs(F)
        return self.get_all_hidden_states(rnn_in, F_lens, h_pad)

    def get_all_rnn_inputs(self, F):
        # Recall:
        #   F is size (S, M)
        #   x (output) is size (S, M, I)
        # assert False, "Fill me"
        x_out = self.embedding(F)
        return x_out

    def get_all_hidden_states(self, x, F_lens, h_pad):
        # Recall:
        #   x is of size (S, M, I)
        #   F_lens is of size (M,)
        #   h_pad is a float
        #   h (output) is of size (S, M, 2 * H)
        #
        # Hint:
        #   relevant pytorch modules:
        #   torch.nn.utils.rnn.{pad_packed,pack_padded}_sequence
        # assert False, "Fill me"
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, F_lens, enforce_sorted=False)
        output, h_n = self.rnn(packed)
        output, length = torch.nn.utils.rnn.pad_packed_sequence(output, padding_value=h_pad)
        return output


class DecoderWithoutAttention(DecoderBase):
    '''A recurrent decoder without attention'''

    def init_submodules(self):
        # Hints:
        # 1. You must initialize the following submodules:
        #   self.embedding, self.cell, self.ff
        # 2. You will need the following object attributes:
        #   self.target_vocab_size, self.word_embedding_size, self.pad_id
        #   self.hidden_state_size, self.cell_type.
        # 3. cell_type will be one of: ['lstm', 'gru', 'rnn']
        # 4. Relevant pytorch modules:
        #   torch.nn.{Embedding, Linear, LSTMCell, RNNCell, GRUCell}
        # assert False, "Fill me"
        self.embedding = torch.nn.Embedding(num_embeddings=self.target_vocab_size,
                                            embedding_dim=self.word_embedding_size,
                                            padding_idx=self.pad_id)
        modules = {
            'lstm': "LSTMCell",
            'gru': "GRUCell",
            'rnn': "RNNCell"
        }

        self.cell = eval("torch.nn." + modules[self.cell_type] + "(" +
                         "input_size=self.word_embedding_size, " +
                         "hidden_size=self.hidden_state_size, " +
                         ")")
        self.ff = torch.nn.Linear(self.hidden_state_size, self.target_vocab_size)

    def forward_pass(self, E_tm1, htilde_tm1, h, F_lens):
        # Recall:
        #   E_tm1 is of size (M,)
        #   htilde_tm1 is of size (M, 2 * H)
        #   h is of size (S, M, 2 * H)
        #   F_lens is of size (M,)
        #   logits_t (output) is of size (M, V)
        #   htilde_t (output) is of same size as htilde_tm1
        #
        # Hints:
        # 1. The structure of the encoder should be:
        #   encoded hidden -> |embedding| -> embedded hidden -> |rnn| ->
        #   decoded hidden -> |output layer| -> output logits
        # 2. You will need to use the following methods:
        #   self.get_current_rnn_input, self.get_current_hidden_state,
        #   self.get_current_logits
        # 3. You can assume that htilde_tm1 is not empty. I.e., the hidden state
        #   is either initialized, or t > 1.
        # 4. The output of an LSTM cell is a tuple (h, c), but a GRU cell or an
        #   RNN cell will only output h.
        # assert False, "Fill me"

        xtilde_t = self.get_current_rnn_input(E_tm1, htilde_tm1, h, F_lens).to(E_tm1.device)
        h_t = self.get_current_hidden_state(xtilde_t, htilde_tm1)
        if self.cell_type == 'lstm':
            logits = self.get_current_logits(h_t[0])
        else:
            logits = self.get_current_logits(h_t)
        return logits, h_t

    def get_first_hidden_state(self, h, F_lens):
        # Recall:
        #   h is of size (S, M, 2 * H)
        #   F_lens is of size (M,)
        #   htilde_tm1 (output) is of size (M, 2 * H)
        #
        # Hint:
        # 1. Ensure it is derived from encoder hidden state that has processed
        # the entire sequence in each direction. You will need to:
        # - Populate indices [0: self.hidden_state_size // 2] with the hidden
        #   states of the encoder's forward direction at the highest index in
        #   time *before padding*
        # - Populate indices [self.hidden_state_size//2:self.hidden_state_size]
        #   with the hidden states of the encoder's backward direction at time
        #   t=0
        # 2. Relevant pytorch function: torch.cat
        # assert False, "Fill me"
        forward = h[F_lens - 1, torch.arange(F_lens.size(0)), : self.hidden_state_size // 2]
        backward = h[0, :, self.hidden_state_size // 2: self.hidden_state_size]

        # S, N, H2 = h.shape
        # forward = h[F_lens - 1, torch.arange(N), : H2 // 2]
        # backward = h[0, :, H2 // 2:]

        return torch.cat((forward, backward), dim=1)

    def get_current_rnn_input(self, E_tm1, htilde_tm1, h, F_lens):
        # Recall:
        #   E_tm1 is of size (M,)
        #   htilde_tm1 is of size (M, 2 * H) or a tuple of two of those (LSTM)
        #   h is of size (S, M, 2 * H)
        #   F_lens is of size (M,)
        #   xtilde_t (output) is of size (M, Itilde)
        # assert False, "Fill me"
        return self.embedding(E_tm1)  # x_tilde_t

    def get_current_hidden_state(self, xtilde_t, htilde_tm1):
        # Recall:
        #   xtilde_t is of size (M, Itilde)
        #   htilde_tm1 is of size (M, 2 * H) or a tuple of two of those (LSTM)
        #   htilde_t (output) is of same size as htilde_tm1
        # assert False, "Fill me"
        htilde_t = self.cell(xtilde_t, htilde_tm1)
        return htilde_t

    def get_current_logits(self, htilde_t):
        # Recall:
        #   htilde_t is of size (M, 2 * H), even for LSTM (cell state discarded)
        #   logits_t (output) is of size (M, V)
        # assert False, "Fill me"
        return self.ff(htilde_t)  # logits


class DecoderWithAttention(DecoderWithoutAttention):
    '''A decoder, this time with attention

    Inherits from DecoderWithoutAttention to avoid repeated code.
    '''

    def init_submodules(self):
        # Hints:
        # 1. Same as the case without attention, you must initialize the
        #   following submodules: self.embedding, self.cell, self.ff
        # 2. You will need the following object attributes:
        #   self.target_vocab_size, self.word_embedding_size, self.pad_id
        #   self.hidden_state_size, self.cell_type.
        # 3. cell_type will be one of: ['lstm', 'gru', 'rnn']
        # 4. Relevant pytorch modules:
        #   torch.nn.{Embedding, Linear, LSTMCell, RNNCell, GRUCell}
        # 5. The implementation of this function should be different from
        #   DecoderWithoutAttention.init_submodules.
        # assert False, "Fill me"
        self.embedding = torch.nn.Embedding(num_embeddings=self.target_vocab_size,
                                            embedding_dim=self.word_embedding_size,
                                            padding_idx=self.pad_id)
        modules = {
            'lstm': "LSTMCell",
            'gru': "GRUCell",
            'rnn': "RNNCell"
        }

        self.cell = eval("torch.nn." + modules[self.cell_type] + "(" +
                         "input_size=self.word_embedding_size + self.hidden_state_size, " +
                         "hidden_size=self.hidden_state_size, " +
                         ")")
        self.ff = torch.nn.Linear(self.hidden_state_size, self.target_vocab_size)

    def get_first_hidden_state(self, h, F_lens):
        # Hint: For this time, the hidden states should be initialized to zeros.
        # assert False, "Fill me"
        return torch.zeros(h.shape[1:]).to(h.device)

    def get_current_rnn_input(self, E_tm1, htilde_tm1, h, F_lens):
        # Hint: Use attend() for c_t
        # assert False, "Fill me"
        embedding = self.embedding(E_tm1).to(h.device)
        c_tm1 = self.attend(htilde_tm1, h, F_lens)
        return torch.cat((embedding, c_tm1), dim=1)

    def attend(self, htilde_t, h, F_lens):
        '''The attention mechanism. Calculate the context vector c_t.

        Parameters
        ----------
        htilde_t : torch.FloatTensor or tuple
            Like `htilde_tm1` (either a float tensor or a pair of float
            tensors), but matching the current hidden state.
        h : torch.FloatTensor
            A float tensor of size ``(S, M, self.hidden_state_size)`` of
            hidden states of the encoder. ``h[s, m, i]`` is the
            ``i``-th index of the encoder RNN's last hidden state at time ``s``
            of the ``m``-th sequence in the batch. The states of the
            encoder have been right-padded such that ``h[F_lens[m]:, m]``
            should all be ignored.
        F_lens : torch.LongTensor
            An integer tensor of size ``(M,)`` corresponding to the lengths
            of the encoded source sentences.

        Returns
        -------
        c_t : torch.FloatTensor
            A float tensor of size ``(M, self.hidden_state_size)``. The
            context vectorc_t is the product of weights alpha_t and h.

        Hint: Use get_attention_weights() to calculate alpha_t.
        '''
        # assert False, "Fill me"

        alpha_t = self.get_attention_weights(htilde_t, h, F_lens)
        alpha_t = torch.repeat_interleave(alpha_t.unsqueeze(2), h.shape[2], dim=2)
        # print(alpha_t.shape, h.shape)
        return torch.sum(torch.mul(alpha_t, h), dim=0)

    def get_attention_weights(self, htilde_t, h, F_lens):
        # DO NOT MODIFY! Calculates attention weights, ensuring padded terms
        # in h have weight 0 and no gradient. You have to implement
        # get_energy_scores()
        # alpha_t (output) is of size (S, M)
        e_t = self.get_energy_scores(htilde_t, h)
        pad_mask = torch.arange(h.shape[0], device=h.device)
        pad_mask = pad_mask.unsqueeze(-1) >= F_lens  # (S, M)
        e_t = e_t.masked_fill(pad_mask, -float('inf'))
        return torch.nn.functional.softmax(e_t, 0)

    def get_energy_scores(self, htilde_t, h):
        # Recall:
        #   htilde_t is of size (M, 2 * H)
        #   h is of size (S, M, 2 * H)
        #   e_t (output) is of size (S, M)
        #
        # Hint:
        # Relevant pytorch function: torch.nn.functional.cosine_similarity
        # assert False, "Fill me"
        if self.cell_type == "lstm":
            return torch.nn.CosineSimilarity(dim=2)(h, htilde_t[0].expand_as(h)).to(h.device)
        return torch.nn.CosineSimilarity(dim=2)(h, htilde_t.expand_as(h)).to(h.device)


class DecoderWithMultiHeadAttention(DecoderWithAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.W is not None, 'initialize W!'
        assert self.Wtilde is not None, 'initialize Wtilde!'
        assert self.Q is not None, 'initialize Q!'

    def init_submodules(self):
        super().init_submodules()  # Do not change this line

        # Hints:
        # 1. The above line should ensure self.ff, self.embedding, self.cell are
        #    initialized
        # 2. You need to initialize the following submodules:
        #       self.W, self.Wtilde, self.Q
        # 3. You will need the following object attributes:
        #       self.hidden_state_size
        # 4. self.W, self.Wtilde, and self.Q should process all heads at once. They
        #    should not be lists!
        # 5. Relevant pytorch module: torch.nn.Linear (note: set bias=False!)
        # 6. You do *NOT* need self.heads at this point
        # assert False, "Fill me"
        self.W = torch.nn.Linear(self.hidden_state_size, self.hidden_state_size, bias=False)
        self.Wtilde = torch.nn.Linear(self.hidden_state_size, self.hidden_state_size, bias=False)
        self.Q = torch.nn.Linear(self.hidden_state_size, self.hidden_state_size, bias=False)

    def attend(self, htilde_t, h, F_lens):
        # Hints:
        # 1. You can use super().attend to call for the regular attention
        #   function.
        # 2. Relevant pytorch function:
        #   tensor().view, tensor().repeat_interleave
        # 3. You *WILL* need self.heads at this point
        # 4. Fun fact:
        #   tensor([1,2,3,4]).repeat(2) will output tensor([1,2,3,4,1,2,3,4]).
        #   tensor([1,2,3,4]).repeat_interleave(2) will output
        #   tensor([1,1,2,2,3,3,4,4]), just like numpy.repeat.
        # assert False, "Fill me"
        # print(h.shape, self.heads)
        # print(h.shape)
        h_multi = h.repeat_interleave(self.heads, dim=1)
        # print(h_multi.shape)
        h = self.W(h_multi)
        h = h[:, torch.arange(0, h.shape[1], self.heads), :]
        if self.cell_type == "lstm":
            htilde_t_val = self.Wtilde(htilde_t[0].repeat_interleave(self.heads, dim=0))
            htilde_t_val = htilde_t_val[torch.arange(0, htilde_t_val.shape[0], self.heads), :]
            htilde_t = (htilde_t_val, htilde_t[1])
            # print(htilde_t[0].shape)
        else:
            htilde_t = self.Wtilde(htilde_t.repeat_interleave(self.heads, dim=0))
            htilde_t = htilde_t[torch.arange(0, htilde_t.shape[0], self.heads), :]
            # print(htilde_t)
        # print(h.shape)
        c_t = super().attend(htilde_t, h, F_lens)
        return self.Q(c_t)


class EncoderDecoder(EncoderDecoderBase):

    def init_submodules(self, encoder_class, decoder_class):
        # Hints:
        # 1. You must initialize the following submodules:
        #   self.encoder, self.decoder
        # 2. encoder_class and decoder_class inherit from EncoderBase and
        #   DecoderBase, respectively.
        # 3. You will need the following object attributes:
        #   self.source_vocab_size, self.source_pad_id,
        #   self.word_embedding_size, self.encoder_num_hidden_layers,
        #   self.encoder_hidden_size, self.encoder_dropout, self.cell_type,
        #   self.target_vocab_size, self.target_eos, self.heads
        # 4. Recall that self.target_eos doubles as the decoder pad id since we
        #   never need an embedding for it
        # assert False, "Fill me"
        self.encoder = encoder_class(self.source_vocab_size,
                                     self.source_pad_id,
                                     self.word_embedding_size,
                                     self.encoder_num_hidden_layers,
                                     self.encoder_hidden_size,
                                     self.encoder_dropout,
                                     self.cell_type)
        self.decoder = decoder_class(self.target_vocab_size,
                                     self.target_eos,
                                     self.word_embedding_size,
                                     self.encoder_hidden_size * 2,
                                     self.cell_type,
                                     self.heads)

    def get_logits_for_teacher_forcing(self, h, F_lens, E):
        # Recall:
        #   h is of size (S, M, 2 * H)
        #   F_lens is of size (M,)
        #   E is of size (T, M)
        #   logits (output) is of size (T - 1, M, Vo)
        #
        # Hints:
        # 1. Relevant pytorch modules: torch.{zero_like, stack}
        # 2. Recall an LSTM's cell state is always initialized to zero.
        # 3. Note logits sequence dimension is one shorter than E (why?)
        # assert False, "Fill me"
        # T, M = E.shape
        # logits = torch.zeros(T - 1, M, self.target_vocab_size, device=h.device)
        #
        # htilde_tm1 = None
        # for t in range(T - 1):
        #     E_tml = E[t]
        #     logits_t, htilde_tm1 = self.decoder.forward(E_tml, htilde_tm1, h, F_lens)
        #     logits[t, :, :] = logits_t
        # return logits
        logits = []
        htilde_tm1 = None
        for t in range(E.size()[0] - 1):
            l, htilde_tm1 = self.decoder.forward(E[t], htilde_tm1, h, F_lens)
            logits.append(l)
        logits = torch.stack(logits[:], 0)
        return logits

    def update_beam(self, htilde_t, b_tm1_1, logpb_tm1, logpy_t):
        # perform the operations within the psuedo-code's loop in the
        # assignment.
        # You do not need to worry about which paths have finished, but DO NOT
        # re-normalize logpy_t.
        #
        # Recall
        #   htilde_t is of size (M, K, 2 * H) or a tuple of two of those (LSTM)
        #   logpb_tm1 is of size (M, K)
        #   b_tm1_1 is of size (t, M, K)
        #   b_t_0 (first output) is of size (M, K, 2 * H) or a tuple of two of
        #      those (LSTM)
        #   b_t_1 (second output) is of size (t + 1, M, K)
        #   logpb_t (third output) is of size (M, K)
        #
        # Hints:
        # 1. Relevant pytorch modules:
        #   torch.{flatten, topk, unsqueeze, expand_as, gather, cat}
        # 2. If you flatten a two-dimensional array of size z of (A, B),
        #   then the element z[a, b] maps to z'[a*B + b]
        # assert False, "Fill me"
        N, K, V = logpy_t.shape
        T = b_tm1_1.shape[0]
        extentions_t = (logpb_tm1.unsqueeze(-1) + logpy_t)

        logpb_t, v = extentions_t.flatten(start_dim=1).topk(K, dim=1)

        indices = v // self.target_vocab_size
        modulos = v % self.target_vocab_size

        b_t_1 = torch.zeros(T + 1, N, K, dtype=torch.long)
        for n in range(N):
            idx = indices[n, :]
            mod = modulos[n, :]
            i = 0
            for j in range(K):
                k, word_idx = idx[j], mod[j]
                b_t_1[:T, n, i] = b_tm1_1[:, n, k]
                b_t_1[T, n, i] = word_idx
                i += 1

        if self.decoder.cell_type == "lstm":
            hid = htilde_t[0]
            cell = htilde_t[1]

            expanded_indices_hid = torch.unsqueeze(indices, dim=2).expand_as(hid)
            expanded_indices_cell = torch.unsqueeze(indices, dim=2).expand_as(cell)
            b_t_0 = (
                torch.gather(hid, dim=1, index=expanded_indices_hid),
                torch.gather(cell, dim=1, index=expanded_indices_cell)
            )
            return b_t_0, b_t_1.to(htilde_t[0].device), logpb_t
        else:
            expanded_indices = torch.unsqueeze(indices, dim=2).expand_as(htilde_t)
            b_t_0 = torch.gather(htilde_t, dim=1, index=expanded_indices)

        return b_t_0, b_t_1.to(htilde_t.device), logpb_t
