import torch.nn as nn
import torch
import torch.nn.functional as F
#import torch.nn.utils.rnn.pack_padded_sequence as pack
#import torch.nn.utils.rnn.pad_packed_sequence as unpack

from torch.autograd import Variable
#from gensim.models import KeyedVectors
import numpy as np

class RNNModelWord(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, nlabel, ntoken, embeds, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModelWord, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        
        #gru
        self.rnn = nn.GRU(ninp, nhid, nlayers, dropout=dropout, bidirectional=False)
        
        #attention parameters
        #weights for single-layer MLP
        self.Ww = nn.Parameter(torch.Tensor(nhid, nhid)) #weight
        self.bw = nn.Parameter(torch.Tensor(nhid, 1)) #bias
        self.uw = nn.Parameter(torch.Tensor(nhid, 1)) #word level context vector

        self.softmax = nn.Softmax(dim=0)


        #self.decoder = nn.Linear(nhid, nlabel)
        #self.sigmoid = nn.Sigmoid()

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        #if tie_weights:
            # if nhid != ninp:
            #     raise ValueError('When using the tied flag, nhid must be equal to emsize')
            # self.decoder.weight = self.encoder.weight

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

        self.Ww.data.uniform_(-initrange, initrange)
        self.uw.data.uniform_(-initrange, initrange) #as it said to in the paper

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        output_affine = self.batch_matmul_bias(output, self.Ww, self.bw, nonlinearity='tanh')
        word_attn = self.batch_matmul(output_affine, self.uw) #uit*uw
        word_attn_norm = self.softmax(word_attn) #softmax over all exp(uit*uw), used as importance weight
        word_attn_vectors = self.attention_mul(output, word_attn_norm)
        return word_attn_vectors, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

    #Functions for attention: copied from https://github.com/EdGENetworks/attention-networks-for-classification
    def batch_matmul_bias(self,seq, weight, bias, nonlinearity=''):
        s = None
        bias_dim = bias.size()
        for i in range(seq.size(0)):
            _s = torch.mm(seq[i], weight) 
            _s_bias = _s + bias.expand(bias_dim[0], _s.size()[0]).transpose(0,1)
            if(nonlinearity=='tanh'):
                _s_bias = torch.tanh(_s_bias)
            _s_bias = _s_bias.unsqueeze(0)
            if(s is None):
                s = _s_bias
            else:
                s = torch.cat((s,_s_bias),0)
        return s.squeeze()
    
    def batch_matmul(self, seq, weight, nonlinearity=''):
        s = None
        for i in range(seq.size(0)):
            _s = torch.mm(seq[i], weight)
            if(nonlinearity=='tanh'):
                _s = torch.tanh(_s)
            _s = _s.unsqueeze(0)
            if(s is None):
                s = _s
            else:
                s = torch.cat((s,_s),0)
        return s.squeeze()

    def attention_mul(self, rnn_outputs, att_weights):
        attn_vectors = None
        for i in range(rnn_outputs.size(0)):
            h_i = rnn_outputs[i]
            a_i = att_weights[i].unsqueeze(1).expand_as(h_i)
            h_i = a_i * h_i
            h_i = h_i.unsqueeze(0)
            if(attn_vectors is None):
                attn_vectors = h_i
            else:
                attn_vectors = torch.cat((attn_vectors,h_i),0)
        return torch.sum(attn_vectors, 0).unsqueeze(0)

#I think the batch size should usually be one? So we get one document.
class RNNModelSentence(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    #def __init__(self, rnn_type, nlabel, ntoken, embeds, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
    #ninp == rnnmodelword's hidden size
    def __init__(self, rnn_type, nlabel, ninp, nhid, nlayers, dropout=0.5):
        super(RNNModelSentence, self).__init__()
        self.drop = nn.Dropout(dropout)
        #self.encoder = nn.Embedding(ntoken, ninp)
        
        #gru
        self.rnn = nn.GRU(ninp, nhid, nlayers, dropout=dropout, bidirectional=False)
        
        #attention parameters
        self.us = nn.Parameter(torch.Tensor(nhid, 1)) #word level context vector
        self.linear_attention = nn.Linear(nhid, nhid)

        self.linear_out = nn.Linear(nhid, nlabel) #goes from attention to label probability vector

        self.softmax = nn.Softmax(dim=0)
        self.softmax2 = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers


        
    def init_weights(self):
        initrange = 0.1
        self.us.data.uniform_(-initrange, initrange) #as it said to in the paper


    def forward(self, input, hidden):
        output, hidden = self.rnn(self.drop(input), hidden)
        output = self.drop(output.unsqueeze(1))
        output_affine = torch.tanh(self.linear_attention(output))
        sent_attn = torch.matmul(output_affine, self.us)
        sent_attn_norm = self.softmax(sent_attn)
        sent_attn_vector = self.weighted_sum_attn(output, sent_attn_norm)

        feature_to_classification = self.linear_out(sent_attn_vector)
        #print("SOFTMAX:",self.softmax2(feature_to_classification),"LOGSOFTMAX:",self.logsoftmax(feature_to_classification))
        return self.logsoftmax(feature_to_classification), hidden, sent_attn_norm

    #hopefully should be 1 bsz? always?
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

    #8x200 and 8
    def weighted_sum_attn(self,output, weights):
        attn_vector = None
        for i in range(len(weights)):
            h_i = output[i]
            a_i = weights[i]
            if attn_vector is None:
                attn_vector = a_i*h_i
            else:
                torch.cat((attn_vector,a_i*h_i),0)
        return torch.sum(attn_vector, 0)