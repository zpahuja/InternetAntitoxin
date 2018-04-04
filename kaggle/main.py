# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

import data
import model

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--labels', type=int, default=6,
                    help='num of labels')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
parser.add_argument('--lang', type=str, default='en',
			help='which lang')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

args.data = './data/'+args.lang
corpus = data.Corpus(args.data, args.lang)
#corpus = data.Corpus(args.data, None)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

# def batchify(data, bsz):
#     # Work out how cleanly we can divide the dataset into bsz parts.
#     nbatch = data.size(0) // bsz
#     # Trim off any extra elements that wouldn't cleanly fit (remainders).
#     data = data.narrow(0, 0, nbatch * bsz)
#     # Evenly divide the data across the bsz batches.
#     data = data.view(bsz, -1).t().contiguous()
#     if args.cuda:
#         data = data.cuda()
#     return data

eval_batch_size = 10
#train_data = batchify(corpus.train, args.batch_size)
#val_data = batchify(corpus.valid, eval_batch_size)
#test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
nlabels = args.labels
embeds = corpus.dictionary.pretrain_vec
model = model.RNNModel(args.model, nlabels, ntokens, embeds, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
if args.cuda:
    model.cuda()

#criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, labels, lens, load, bsz, evaluation=False):
    #seq_len = min(args.bptt, len(source) - 1 - i)
    #data = Variable(source[i:i+seq_len], volatile=evaluation)
    #return data, target
    #MAX LEN IS PER LOAD

    #source is a list of longtensor
    #labels is a list of longtensor
    # comments = []
    # for ii in range(load,load+bsz):
    #     comments.append(source[ii])
    # maxlen = max(comments, key=size)
    if load+bsz > len(source):
        end = len(source)
        #data_tensor = torch.LongTensor(maxlen, end-load).zero_()
        #label_tensor = torch.LongTensor(6, end-load).zero_()
    else:
        end = load+bsz
        
    maxlen = max(lens[load:end])

    
    data_tensor = torch.LongTensor(maxlen, bsz).zero_()
    label_tensor = torch.FloatTensor(6, bsz).zero_()
    
    count = 0
    for ii in range(load,end): #only use these sentences
        source[ii]
        data_tensor[:source[ii].size(0),count] = source[ii]
        #print(labels[ii])
        label_tensor[:,count] = labels[ii]
        count+=1

    data = Variable(data_tensor)
    label = Variable(label_tensor)
    lens = Variable(torch.from_numpy(np.array(lens[load:end])))

    return data, label, lens

def evaluate(data_source, test=False):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    nlabels = args.labels
    hidden = model.init_hidden(eval_batch_size)
    #for i in range(0, data_source.size(0) - 1, args.bptt):
    #for batch, load in enumerate(range(0,eval_batch_size,len(corpus.validid))):
    for batch, load in enumerate(range(0,len(corpus.validid),eval_batch_size)):
        if test:
            if load+eval_batch_size > len(corpus.testid):
                continue
            data, targets, lens = get_batch(corpus.testid, corpus.testlab, corpus.testlen, load, eval_batch_size, evaluation=True)
        else:
            if load+eval_batch_size > len(corpus.validid):
                continue
            data, targets, lens = get_batch(corpus.validid, corpus.validlab, corpus.validlen, load, eval_batch_size, evaluation=True)
        output, hidden, perm_index = model.forward(data, hidden, lens)
        targets = targets[:,perm_index]
        output_flat = output.view(-1, nlabels)
        #print(output_flat, torch.t(targets))
        total_loss += criterion(output_flat, torch.t(targets)).data
        #print(total_loss[0])
        hidden = repackage_hidden(hidden)
    if test:
        return total_loss[0]
    else:
        return total_loss[0]


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    nlabels = args.labels
    hidden = model.init_hidden(args.batch_size)

    #for batch, load in enumerate(range(0,args.batch_size,len(corpus.trainid))):
    for batch, load in enumerate(range(0,len(corpus.trainid),args.batch_size)):
        #print("TRAIN CORPUS LEN",len(corpus.trainid))
        if load+args.batch_size > len(corpus.trainid):
            continue
        data, targets, lens = get_batch(corpus.trainid, corpus.trainlab, corpus.trainlen, load, args.batch_size)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        #print(data)
        output, hidden, perm_index = model.forward(data, hidden, lens)
        targets = targets[:,perm_index]
        #print(output.view(-1, nlabels))
        #print(torch.t(targets))
        loss = criterion(output.view(-1, nlabels), torch.t(targets))
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(corpus.trainid) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    x = range(1,args.epochs+1)
    y = [] # valid loss
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(corpus.validid, test=False)
        y.append(val_loss)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(x,y)
    plt.title(args.lang + 'baseline')
    fig.savefig(args.lang + 'baseline.png')   # save the figure to file
    plt.close(fig)    # close the figure
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = evaluate(corpus.testid, test=True)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
