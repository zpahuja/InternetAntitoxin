import torch.nn as nn
import torch
#import torch.nn.utils.rnn.pack_padded_sequence as pack
#import torch.nn.utils.rnn.pad_packed_sequence as unpack

from torch.autograd import Variable
#from gensim.models import KeyedVectors
import numpy as np

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, nlabel, ntoken, embeds, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        # pretrained_weight is a numpy matrix of shape (num_embeddings, embedding_dim)
        # self.encoder.weight.data.copy_(torch.from_numpy(pretrained_weight))
        # self.encoder.weight.requires_grad = false
        # > parameters = filter(lambda p: p.requires_grad, net.parameters()) on optimizer to tell it to not grad encoder
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout, bidirectional=False)
        self.decoder = nn.Linear(nhid, nlabel)
        self.sigmoid = nn.Sigmoid()

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights(tuple(embeds),ntoken,ninp)

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

        
    def init_weights(self,embeds,ntoken,ninp):
        initrange = 0.1

        k = len(embeds) # the first k indices are pretrained. the rest are unknown
        if k is not 0:
            first = np.array(embeds)
            second = np.random.uniform(-initrange,initrange,size=(ntoken-k,ninp))
            self.encoder.weight.data.copy_(torch.from_numpy(np.concatenate((first,second),axis=0)))
        else:
            self.encoder.weight.data.uniform_(-initrange, initrange)
        #self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, lengths):
        # Sort the input and lengths as the descending order
        #emb = self.drop(self.encoder(input))
        #emb = self.encoder(input)
        lengths, perm_index = lengths.sort(0, descending=True)
        input = input[:,perm_index]
        #print(input)
        emb = self.encoder(input)
        #print(emb)
        #print(emb)
        #print(lengths, torch.typename(lengths.data), lengths[0], lengths[-1]<=0)
        packed_input = nn.utils.rnn.pack_padded_sequence(emb, lengths.data.cpu().numpy())

        
        output, hidden = self.rnn(packed_input, hidden)

        last = hidden[0][-1]
        decoded = self.decoder(last)
        decoded = self.sigmoid(decoded)
        #print(decoded)
        #print(decoded)
        #output, _ = nn.utils.rnn.pad_packed_sequence(output)

        #output = self.drop(output)
        #decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        #decoded = decoded.view(output.size(0),output.size(1), decoded.size(1))

        #print(hidden)

        #decoded = torch.sum(decoded, 0) #NOTE: edit this to be a weighted sum by making an extra linear layer
        #print(decoded)
        #decoded = self.decoder(output)
        #return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden, perm_index
        return decoded, hidden, perm_index

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
