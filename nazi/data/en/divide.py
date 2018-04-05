import numpy as np
import os
import csv

assert os.path.exists('dataset.csv')

csvfile = open('dataset.csv','r').readlines()
#n = len(csvfile)
n=10000
c = 100000
start = 0+c
train_end = int(.6*n)+c
valid_end = int(.2*n) + train_end+c
test_end = int(.2*n) + valid_end+c
open(str('train') + '.csv', 'w+').writelines(csvfile[start:train_end])
open(str('valid') + '.csv', 'w+').writelines(csvfile[train_end:valid_end])
open(str('test') + '.csv', 'w+').writelines(csvfile[valid_end:test_end])