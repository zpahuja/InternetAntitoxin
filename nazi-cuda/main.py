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
parser.add_argument('--labels', type=int, default=2,
                    help='num of labels')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nhidsent', type=int, default=200,
                    help='number of hidden unit per layer in sentence model')
parser.add_argument('--nlayers', type=int, default=1,
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
parser.add_argument('--dropout', type=float, default=0.0,
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
                    help='path to save the final word model')
parser.add_argument('--savesent', type=str, default='modelsent.pt',
                    help='path to save the final sent model')
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

def batchify_user(data, bsz, start, end):
    if end==start:
        return False
    if end<start:
        return False
    #print(start,end)
    user_data = data[start:end]
    nbatch = user_data.size(0) // bsz
    #print(nbatch, bsz, start, end)
    if nbatch == 0:
        return False
    user_data = user_data.narrow(0,0,nbatch*bsz)
    user_data = user_data.view(bsz,-1).t().contiguous()
    if args.cuda:
        user_data = user_data.cuda()
    return user_data

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
word_attn_model = model.RNNModelWord(args.model, nlabels, ntokens, embeds, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
sent_attn_model = model.RNNModelSentence(args.model, nlabels, args.nhid, args.nhidsent, args.nlayers, args.dropout)
if args.cuda:
    word_attn_model.cuda()
    sent_attn_model.cuda()

#criterion = nn.CrossEntropyLoss()
#criterion = nn.BCELoss()
#criterion = nn.NLLLoss(Variable(torch.FloatTensor([.1,.9])))
criterion = nn.NLLLoss()

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

#if batch size = 20, return 20 sentences.
#source = list of tweets of a user
#lens = list of lens of tweets of a user
'''
def get_batch(source, lens, load, bsz, evaluation=False):
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
    
    count = 0
    for ii in range(load,end): #only use these sentences
        source[ii]
        data_tensor[:source[ii].size(0),count] = source[ii]
        #print(labels[ii])
        count+=1

    data = Variable(data_tensor)
    #if end is not load+bsz:
    #    lens = Variable(torch.from_numpy(np.append(np.array(lens[load:end]),np.zeros(load+bsz-len(source)))))
    #else:
    lens = Variable(torch.from_numpy(np.array(lens[load:end])))

    #print(data, "batch")

    return data, lens
'''
#pass in result of batchify_user
def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i*args.bptt)
    #print(len(source), i*args.bptt, i*args.bptt+seq_len)
    if i*args.bptt - i*args.bptt+seq_len < 3:
        return False
    data = Variable(source[i*args.bptt:i*args.bptt+seq_len], volatile=evaluation)
    return data

def evaluate(data_source, data_labels, data_idx, test=False):
    # Turn on evaluation mode which disables dropout.
    word_attn_model.eval()
    sent_attn_model.eval()
    total_loss = 0
    nlabels = args.labels
    

    correct = 0
    total = 0
    for ii in range(len(data_labels)): #loop through users
        total+=1

        label = data_labels[ii:ii+1] #label of user   

        user_idx = data_idx[ii] #where user starts
        
        if ii+1 == len(data_labels):
            user_end_idx = len(data_source)
        else:
            user_end_idx = corpus.trainidx[ii+1]
        
        user_id = batchify_user(data_source, eval_batch_size, user_idx, user_end_idx)


        hidden = word_attn_model.init_hidden(eval_batch_size)
        hidden_sent = sent_attn_model.init_hidden(1)
        sentences = None
        #for load in range(0,len(user_id)):
        if user_id is not False:
            for load in range(0,len(user_id)//args.bptt+1):
                #print("iter eval",load)
                #if len(train_user_id)-load < 3: #not enough data remaining
                    #continue
                data = get_batch(user_id, load, True)
                if data is False:
                    continue
                #for batch, load in enumerate(range(0,len(user_id)-1,args.bptt)):
                #if len(user_id)-load < 3: #not enough data remaining
                    #continue
                #data = get_batch(user_id, load, evaluation=True)
                
                output, hidden = word_attn_model.forward(data, hidden)
                hidden = repackage_hidden(hidden)
                if sentences is None:
                    sentences = output.transpose(0,1)
                else:
                    sentences = torch.cat((sentences,output.transpose(0,1)),0)


        if sentences is None:
            continue
        y_pred, hidden_sent, _ = sent_attn_model(sentences, hidden_sent)
        hidden_sent = repackage_hidden(hidden_sent)

        output_flat = y_pred.view(-1, nlabels)
        #print(output_flat, "output of sent attn model")
        #print(Variable(label), "actual labels")

        total_loss += criterion(output_flat, Variable(label)).data
        if int(label[0])==0:#gold is 0
            if float(output_flat[0,0])>float(output_flat[0,1]):
                correct +=1
            else:
                correct +=0
        else:#gold is 1
            if float(output_flat[0,0])>float(output_flat[0,1]):
                correct+=0
            else:
                correct+=1
        hidden = repackage_hidden(hidden)
    
    if test:
        return total_loss[0], correct/total
    else:
        return total_loss[0], correct/total


def train():
    # Turn on training mode which enables dropout.
    word_attn_model.train()
    sent_attn_model.train()
    total_loss = 0
    start_time = time.time()
    nlabels = args.labels
    

    #for batch, load in enumerate(range(0,args.batch_size,len(corpus.trainid))):
    #num_sentences = len(corpus.trainid)//args.batch_size
    total = 0
    correct = 0
    for ii in range(len(corpus.trainlab)): #loop through users
        hidden = word_attn_model.init_hidden(args.batch_size)
        hidden_sent = sent_attn_model.init_hidden(1)
        train_user_idx = corpus.trainidx[ii] #where user starts
        
        if ii+1 == len(corpus.trainlab):
            train_user_end_idx = len(corpus.trainid)
        else:
            train_user_end_idx = corpus.trainidx[ii+1]
        #print("Training a user: words",train_user_idx,"-",train_user_end_idx)
        train_user_id = batchify_user(corpus.trainid, args.batch_size, train_user_idx, train_user_end_idx)
        train_user_lab = corpus.trainlab[ii:ii+1] #label of user
        sentences = None
        #for batch, load in enumerate(range(0,len(train_user_id)-1,args.bptt)):
        if train_user_id is not False:
            for load in range(0,len(train_user_id)//args.bptt+1):
                #TODO do the last seq -_-
                #print("iter",load)
                #if len(train_user_id)-load < 3: #not enough data remaining
                    #continue
                data = get_batch(train_user_id, load, False)
                if data is False:
                    continue
                # Starting each batch, we detach the hidden state from how it was previously produced.
                # If we didn't, the model would try backpropagating all the way to start of the dataset.
                hidden = repackage_hidden(hidden)
                word_attn_model.zero_grad()
                output, hidden = word_attn_model.forward(data, hidden)
                if sentences is None:
                    sentences = output.transpose(0,1)
                else:
                    sentences = torch.cat((sentences,output.transpose(0,1)),0)

        if sentences is None: #means user didn't have enough data/had none. TODO hope this is right to skip.
            continue
        hidden_sent = repackage_hidden(hidden_sent)
        y_pred, hidden_sent, _ = sent_attn_model(sentences, hidden_sent)

        #print("Y_PRED:",y_pred.view(-1,nlabels),"LAB:",train_user_lab)
        loss = criterion(y_pred.view(-1, nlabels), Variable(train_user_lab))
        loss.backward()

        label = train_user_lab
        output_flat = y_pred.view(-1,nlabels)
        if int(label[0])==0:#gold is 0
            if float(output_flat[0,0])>float(output_flat[0,1]):
                correct +=1
                print("Correct, not nazi (0)")
                print("OMG!!!!!!!!!!")
            else:
                print("Incorrect, not nazi (0)")
                correct +=0
        else:#gold is 1
            if float(output_flat[0,0])>float(output_flat[0,1]):
                print("Incorrect, is nazi (1)")
                print("OMG!!!!!!!!!!!!!!!!")
                correct+=0
            else:
                print("Correct, is nazi (1)")
                correct+=1
        total+=1
        print("training accuracy so far:",correct/total)
        #print("training loss:",loss[0])

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(word_attn_model.parameters(), args.clip)
        for p in word_attn_model.parameters():
            p.data.add_(-lr, p.grad.data)
        torch.nn.utils.clip_grad_norm(sent_attn_model.parameters(), args.clip)
        for p in sent_attn_model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.data

        '''
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | user {:3d}'.format(
                epoch, batch, len(train_user_id) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), ii))
            total_loss = 0
            start_time = time.time()
        '''


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
        val_loss, accuracy = evaluate(corpus.validid, corpus.validlab, corpus.valididx, test=False)
        y.append(val_loss)
        print('-' * 96)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} '
                '| accuracy {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, accuracy))
        print('-' * 96)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(word_attn_model, f) #TODO: obviously fix this to extend to sent
            with open(args.savesent, 'wb') as f:
                torch.save(sent_attn_model, f)
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
    print('-' * 96)
    print('Exiting from training early')

# Load the best saved model.
#TODO FIX THIS TO WORK?
with open(args.save, 'rb') as f:
    word_attn_model = torch.load(f)
with open(args.savesent, 'rb') as f:
    sent_attn_model = torch.load(f)

# Run on test data.
test_loss, accuracy = evaluate(corpus.testid, corpus.testlab, corpus.testidx, test=True)
print('=' * 89)
print('| End of training | test loss {:5.2f} | accuracy {:8.2f}'.format(
    test_loss, accuracy))
print('=' * 89)
