import random
import time

import torch
from torch.autograd import Variable
from model.CopyNet import CopyNet
from model.Lang import prepare_data, time_since



def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def variable_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    var = Variable(torch.LongTensor(indexes).view(-1, 1))
#     print('var =', var)
    if USE_CUDA: var = var.cuda()
    return var

def variables_from_pair(pair):
    input_variable = variable_from_sentence(input_lang, pair[0])
    target_variable = variable_from_sentence(output_lang, pair[1])
    return (input_variable, target_variable)


attn_model = 'general'
hidden_size = 500
n_layers = 2
dropout_p = 0.05
USE_CUDA = False

n_epochs = 50000
plot_every = 200
print_every = 100
sava_every = 10000

SOS_token = 0
EOS_token = 1

input_lang, output_lang, pairs = prepare_data('eng', 'fra', True)
model = CopyNet(input_lang.n_words, output_lang.n_words, hidden_size, attn_model, n_layers, USE_CUDA = USE_CUDA)
if USE_CUDA:
    model = model.cuda()
# Keep track of time elapsed and running averages
start = time.time()
plot_losses = []
print_loss_total = 0 # Reset every print_every
plot_loss_total = 0 # Reset every plot_every
# Begin!
for epoch in range(1, n_epochs + 1):
    # Get training data for this cycle
    print(epoch)
    training_pair = variables_from_pair(random.choice(pairs))
    input_variable = training_pair[0]
    target_variable = training_pair[1]

    # Run the train function
    loss = model.train(input_variable, target_variable)

    # Keep track of loss
    print_loss_total += loss
    plot_loss_total += loss

    if epoch == 0: continue

    if epoch % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print_summary = '%s (%d %d%%) %.4f' % (
        time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
        print(print_summary)

    if epoch % plot_every == 0:
        plot_loss_avg = plot_loss_total / plot_every
        plot_losses.append(plot_loss_avg)
        plot_loss_total = 0