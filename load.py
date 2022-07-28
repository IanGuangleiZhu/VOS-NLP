# load 20 news dataset
import numpy as np
import torch.nn as nn
import torch
from sklearn.datasets import fetch_20newsgroups

categories = ['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']

newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True, 
                                      categories=categories,)

print (newsgroups_train.target_names)
print (len(newsgroups_train.data))
print("    ***************  ")
print((newsgroups_train.data[1]))
print((categories[newsgroups_train.target[1]]))