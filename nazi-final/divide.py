import numpy as np
import os
import csv

assert os.path.exists('dataset.csv')

csvfile = open('dataset.csv','r').readlines()
#n = len(csvfile)
n = 160000
#n=3000
#n=100000
c = 0
start = 0
train_end = int(.7*n)
#valid_end = int(.2*n) + train_end
#test_end = int(.2*n) + valid_end

test_end = int(.3*n) + train_end
open(str('train') + '.csv', 'w+').writelines(csvfile[start:train_end])
#open(str('valid') + '.csv', 'w+').writelines(csvfile[train_end:valid_end])
#open(str('test') + '.csv', 'w+').writelines(csvfile[valid_end:test_end])
open(str('test') + '.csv', 'w+').writelines(csvfile[train_end:test_end])