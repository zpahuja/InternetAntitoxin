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
        self.trainid, self.trainlab, self.trainlen = self.tokenize(os.path.join(path, 'train.csv'),True)
        self.validid, self.validlab, self.validlen = self.tokenize(os.path.join(path, 'valid.csv'),False)
        self.testid, self.testlab, self.testlen = self.tokenize(os.path.join(path, 'test.csv'),False)

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
    def tokenize(self, path, header):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            lens = []
            reader = csv.reader(f)
            tokens = 0
            rows = 0
            if header:
                first = True
            else:
                first = False
            for row in reader:
                if first:
                    first = False
                    continue
                if len(row) is not 8:
                    continue
                id = row[0]
                text = row[1]
                labels = row[2:8]
                words = text.split()
                lens.append(len(words))
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids_list = []
            labels_list = []
            #ids = torch.LongTensor(tokens)
            token = 0

            reader = csv.reader(f)
            if header:
                first = True
            else:
                first = False
            i=0
            for row in reader:
                if first:
                    first = False
                    continue
                if len(row) is not 8:
                    continue
                ids = torch.LongTensor(lens[i])
                labels = torch.from_numpy(np.array(row[2:8],dtype=float))
                id = row[0]
                text = row[1]
                words = text.split()
                tokens += len(words)
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
                i+=1
                token = 0
                ids_list.append(ids)
                labels_list.append(labels)

        return ids_list, labels_list, lens
