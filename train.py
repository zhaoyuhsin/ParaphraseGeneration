import random
import time

import torch
from torch.autograd import Variable
from model.CopyNet import CopyNet
from model.Lang import prepare_data, time_since, add_database
from model.util import variables_from_pair





attn_model = 'general'
hidden_size = 500
n_layers = 2
dropout_p = 0.05
USE_CUDA = False

n_epochs = 50000
plot_every = 200
print_every = 100
sava_every = 100

SOS_token = 0
EOS_token = 1
language, pairs = prepare_data('data/train_src.txt', 'data/train_tgt.txt')
#add_database(language, 'data/valid_src.txt')
#add_database(language, 'data/valid_tgt.txt')
#add_database(language, 'data/test_src.txt')
#add_database(language, 'data/test_tgt.txt')
print(language.n_words)

print('--------------------------------------------------------------------')



model = CopyNet(language.n_words, language.n_words, hidden_size, attn_model, n_layers, USE_CUDA = USE_CUDA)
#model.load_state_dict(torch.load('save_model/model_300.pt'))
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
    training_pair = variables_from_pair(language, random.choice(pairs), USE_CUDA)
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
    if epoch % sava_every == 0:
        save_path = 'save_model/model_' + str(epoch) + '.pt'
        torch.save(model.state_dict(), save_path)
