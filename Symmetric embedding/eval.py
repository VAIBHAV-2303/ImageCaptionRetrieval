import numpy as np
import torch
from torch import nn
from utils import get_hot
from model import ImgEncoder, SentenceEncoder
import pickle

# parameters
dim_image = 4096
dim = 1024
dim_word = 300
path = 'data/coco/'

# Loading the dataset
print('Loading the test dataset')
with open('data/coco/Test/image_features.pkl', 'rb') as f:
	ims = pickle.load(f)
with open('data/coco/Test/captions.pkl', 'rb') as f:
	caps = pickle.load(f)
print('Images shape:', ims.shape, 'Length of captions:', len(caps))

# Loading dictionary
with open('worddict.pkl', 'rb') as f:
	worddict = pickle.load(f)
print('Loaded dictionary')

# Loading trained models
ImgEncoder = ImgEncoder(dim_image, dim).cuda()
ImgEncoder.load_state_dict(torch.load('ImgEncoder.pt'))
SentenceEncoder = SentenceEncoder(len(worddict)+2, dim_word, dim).cuda()
SentenceEncoder.load_state_dict(torch.load('SentenceEncoder.pt'))
cos = nn.CosineSimilarity()
print('Models loaded')

# Data preproc
r = []
encoded_ims = ImgEncoder(ims)
for i in range(len(caps)):
	hot = get_hot(caps[i], worddict)
	encoded_cap = SentenceEncoder(hot).repeat(ims.shape[0], 1)
	S = cos(encoded_cap, encoded_ims)
	ranks = S.argsort().cpu().numpy()[::-1]
	r.append(np.where(ranks==i//5)[0][0] + 1)
	print('Rank:', str(r[-1])+'/'+str(ranks.shape[0]))
print(np.mean(np.array(r)))