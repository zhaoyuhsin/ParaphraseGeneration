import torch
from torch.autograd import Variable
SOS_token = 0
EOS_token = 1
def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def variable_from_sentence(lang, sentence, USE_CUDA):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    var = Variable(torch.LongTensor(indexes).view(-1, 1))
#     print('var =', var)
    if USE_CUDA: var = var.cuda()
    return var

def variables_from_pair(language, pair, USE_CUDA):
    input_variable = variable_from_sentence(language, pair[0], USE_CUDA)
    target_variable = variable_from_sentence(language, pair[1], USE_CUDA)
    return (input_variable, target_variable)

def generate(language, words):
    ans = ''
    for word in words:
        index = word.item()
        if (index == 0 or index == 1): break
        ans += language.index2word[index] + ' '
    return ans