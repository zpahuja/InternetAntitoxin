# Note

Please add 'wiki.en.vec' to this directory. It can be downloaded from https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec .

## Files


abbreviations.txt - Abbreviations that Nazis use. They stand for phrases, groups, or symbols.

german.txt - German words that American Nazis use. Unused.

groups.txt - Groups of Nazis (gangs, bands, clubs). Unused.

phrases.txt - Phrases Nazis say according to ADL. Unused except for ((( and ))).


baseline.ipynb - Hierachical document model baseline.

bigru.ipynb - biGRU.

bigru-attn.ipynb - biGRU with attention.

lstm.ipynb - biLSTM.

lstm-attn.ipynb - biLSTM with attention.

adl.ipynb - Baseline, but uses abbreviations and echo as discrete feature concatenated to word embeddings.

adl-user.ipynb - Like above, but adds the features to the user representation if they use ADL-defined words. Also has code in place to easily find image vectors, but it's unused because it gave poor results.

dataset.csv - Full data.

train,test.csv - Data cut in two.


divide.py - You can run it to redivide train/test or artifically shrink the data used. n is data, .7* and .3* can be changed to find your split.

img_to_vec.py - From https://github.com/christiansafka/img2vec.git. If you want image vectors.

