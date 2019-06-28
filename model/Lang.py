import unicodedata
import string
import re
import random
import time
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def add_database(language, filename):
    sentences = open(filename).read().split('\n')
    for sentence in sentences:
        language.index_words(sentence)
def read_langs(file_name1, file_name2):
    print("Reading lines...")

    input_sentences = open(file_name1).read().split('\n')
    output_sentences = open(file_name2).read().split('\n')
    pairs = [[input_sentences[i], output_sentences[i]]for i in range(len(input_sentences))]
    print('len_pairs ', len(pairs))
    print(pairs[2])
    language = Lang('Language')
    return language, pairs

MAX_LENGTH = 20

def filter_pair(p):
    # not use !!!!!!!!!!!
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


def prepare_data(file_name1, file_name2):
    language, pairs = read_langs(file_name1, file_name2)
    print("Read %s sentence pairs" % len(pairs))


    print("Indexing words...")
    print('len = ', len(pairs))
    for pair in pairs:
        language.index_words(pair[0])
        language.index_words(pair[1])

    return language, pairs




class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, USE_CUDA = True):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.USE_CUDA = USE_CUDA

    def forward(self, word_inputs, hidden):
        # Note: we run this all at once (over the whole input sequence)
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if self.USE_CUDA: hidden = hidden.cuda()
        return hidden


class Attn(nn.Module):
    def __init__(self, method, hidden_size, max_length=MAX_LENGTH, USE_CUDA=True):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size
        self.USE_CUDA = USE_CUDA
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)
        #print('Attention ', hidden.size())
        #print('encoder_output ', encoder_outputs.size())
        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(seq_len))  # B x 1 x S

        if self.USE_CUDA: attn_energies = attn_energies.cuda()

        # Calculate energies for each encoder output
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])
        #print('energy.size ', attn_energies.size())
        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return F.softmax(attn_energies).unsqueeze(0).unsqueeze(0)

    def score(self, hidden, encoder_output):
        hidden = hidden.squeeze(0)
        encoder_output = encoder_output.squeeze(0)
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.other.dot(energy)
            return energy


class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout_p=0.1, USE_CUDA = True):
        super(AttnDecoderRNN, self).__init__()

        # Keep parameters for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 2, output_size)
        self.USE_CUDA = USE_CUDA

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size, USE_CUDA = self.USE_CUDA)

    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, 1, -1)  # S=1 x B x N
        #print('word_input', word_input)
        #print('word_embedded.size', word_embedded.size())
        # Combine embedded input word and last context, run through RNN
        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
        rnn_output, hidden = self.gru(rnn_input, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)
        #print('attn_weights', attn_weights.size())
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N
        #print('context ', context.size())
        # Final output layer (next word prediction) using the RNN hidden state and context vector
        rnn_output = rnn_output.squeeze(0)  # S=1 x B x N -> B x N
        context = context.squeeze(1)  # B x S=1 x N -> B x N
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)))
        #print('output ', output.size())

        # Return final output, hidden state, and attention weights (for visualization)
        return output, context, hidden, attn_weights



def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

