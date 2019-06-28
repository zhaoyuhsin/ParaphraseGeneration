import random
import time

import torch
from torch.autograd import Variable
from model.CopyNet import CopyNet
from model.Lang import prepare_data, time_since, add_database, Lang
from model.util import generate

attn_model = 'general'
hidden_size = 500
n_layers = 2
dropout_p = 0.05
USE_CUDA = False

n_epochs = 50000
plot_every = 200
print_every = 100
sava_every = 5000

SOS_token = 0
EOS_token = 1

language, pairs = prepare_data('data/train_src.txt', 'data/train_tgt.txt')
model = CopyNet(language.n_words, language.n_words, hidden_size, attn_model, n_layers, USE_CUDA = USE_CUDA)
model.load_state_dict(torch.load('save_model/model_100.pt'))

result = model.predict(language, 'how do i upload pictures to quora questions')
result = generate(language, result)
print(result)