import os
import torch
import numpy as np
import csv

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.pretrain_vec = [] # should match index order of words in dict.

    def add_word(self, word, vec=None):
        if vec is None:
            if word not in self.word2idx:
                self.idx2word.append(word)
                self.word2idx[word] = len(self.idx2word) - 1
        else:
            if word not in self.word2idx:
                self.pretrain_vec.append(vec)
                self.idx2word.append(word)
                self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, language):
        self.dictionary = Dictionary()
        if language is not None:
            self.pretrained = self.add_pretrained(os.path.join(path, 'wiki.' + language + '.vec'))
        self.trainid, self.trainlab, self.trainidx = self.tokenize_by_user(os.path.join(path, 'train.csv'),True)
        self.validid, self.validlab, self.valididx = self.tokenize_by_user(os.path.join(path, 'valid.csv'),False)
        self.testid, self.testlab, self.testidx = self.tokenize_by_user(os.path.join(path, 'test.csv'),False)

    def add_pretrained(self, path):
        assert os.path.exists(path)

        # Add words with pretrained vectors to the dictionary
        # might be weird because no eos was added?
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split()
                if len(words) == 2: #first line
                    continue
                word = words[0]
                vec = words[1:]
                if len(vec) != 300:
                    continue #this skips the space embedding
                #vec = np.array(list(map(float, vec)))
                vec = list(map(float,vec))
                tokens += 1
                
                self.dictionary.add_word(word, vec)
    def tokenize_by_user(self, path, header):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            reader = csv.reader(f)
            tokens = 0
            prev = None
            if header:
                first = True
            else:
                first = False
            tweet_count = 0
            user_idx = -1
            for row in reader:
                if first:
                    first = False
                    continue
                if len(row) is not 6:
                    continue
                
                tweet = row[0]
                label = row[1]
                if not label.isdigit():
                    continue
                extra = row[2:5] #bio, tweet pic, profile pic, user id
                if row[2] != prev: #new user
                    prev = row[2]
                    tweet_count = 0
                    user_idx += 1

                words = tweet.split()
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            idxs = torch.LongTensor(user_idx+1)
            labels = torch.LongTensor(user_idx+1)
            #print(user_idx+1)
            token = 0
            prev = None

            reader = csv.reader(f)
            if header:
                first = True
            else:
                first = False
            user_idx = -1
            for row in reader:
                if first:
                    first = False
                    continue
                if len(row) is not 6:
                    continue
                
                tweet = row[0]
                label = row[1]
                if not label.isdigit():
                    continue
                extra = row[2:5] #bio, tweet pic, profile pic, user id
                if row[2] != prev:
                    user_idx += 1
                    prev = row[2]
                    labels[user_idx] = int(label)
                    #print(token, "NEW USER")
                    idxs[user_idx] = token
                

                words = tweet.split()
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token+=1
                

        return ids, labels, idxs
