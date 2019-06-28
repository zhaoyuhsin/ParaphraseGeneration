import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from model.Lang import EncoderRNN, AttnDecoderRNN
from model.util import variable_from_sentence
import random
class CopyNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, attn_model, n_layers = 1, dropout_p = 0.1, learning_rate = 0.0001, USE_CUDA = False):
        super(CopyNet, self).__init__()
        self.encoder = EncoderRNN(input_size, hidden_size, n_layers, USE_CUDA=USE_CUDA)
        self.decoder = AttnDecoderRNN(attn_model, hidden_size, output_size, n_layers, dropout_p=dropout_p, USE_CUDA=USE_CUDA)
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate)
        self.criterion = nn.NLLLoss()
        self.USE_CUDA = USE_CUDA
        self.SOS_token = 0
        self.EOS_token = 1
        self.teacher_forcing_ratio = 0.5
        self.clip = 5.0
    def forward(self):
        return 0
    def train(self, input_variable, target_variable):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss = 0  # Added onto for each word

        # Get size of input and target sentences
        input_length = input_variable.size()[0]
        target_length = target_variable.size()[0]

        # Run words through encoder
        encoder_hidden = self.encoder.init_hidden()
        encoder_outputs, encoder_hidden = self.encoder(input_variable, encoder_hidden)

        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([[self.SOS_token]]))
        decoder_context = Variable(torch.zeros(1, self.decoder.hidden_size))
        decoder_hidden = encoder_hidden  # Use last hidden state from encoder to start decoder
        if self.USE_CUDA:
            decoder_input = decoder_input.cuda()
            decoder_context = decoder_context.cuda()

        # Choose whether to use teacher forcing
        use_teacher_forcing = random.random() < self.teacher_forcing_ratio
        if use_teacher_forcing:

            # Teacher forcing: Use the ground-truth target as the next input
            for di in range(target_length):
                decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(decoder_input,
                                                                                             decoder_context,
                                                                                             decoder_hidden,
                                                                                             encoder_outputs)
                loss += self.criterion(decoder_output, target_variable[di])
                decoder_input = target_variable[di]  # Next target is next input

        else:
            # Without teacher forcing: use network's own prediction as the next input
            for di in range(target_length):
                decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(decoder_input,
                                                                                             decoder_context,
                                                                                             decoder_hidden,
                                                                                             encoder_outputs)
                loss += self.criterion(decoder_output, target_variable[di])

                # Get most likely word index (highest value) from output
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]

                decoder_input = Variable(torch.LongTensor([[ni]]))  # Chosen word is next input
                if self.USE_CUDA: decoder_input = decoder_input.cuda()

                # Stop at end of sentence (not necessary when using known targets)
                if ni == self.EOS_token: break

        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.encoder.parameters(), self.clip)
        torch.nn.utils.clip_grad_norm(self.decoder.parameters(), self.clip)
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        return loss.data / target_length
    def predict(self, language, sentence):
        input_variable = variable_from_sentence(language, sentence, self.USE_CUDA)
        encoder_hidden = self.encoder.init_hidden()
        encoder_outputs, encooder_hidden = self.encoder(input_variable, encoder_hidden)
        decoder_input = Variable(torch.LongTensor([[self.SOS_token]]))
        decoder_context = Variable(torch.zeros(1, self.decoder.hidden_size))
        decoder_hidden = encoder_hidden
        if self.USE_CUDA:
            decoder_input = decoder_input.cuda()
            decoder_context = decoder_context.cuda()
        di = 0
        result = []
        while(True):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            result.append(ni)
            decoder_input = Variable(torch.LongTensor([[ni]]))
            if self.USE_CUDA: decoder_input = decoder_input.cuda()
            if ni == self.EOS_token: break
        return result