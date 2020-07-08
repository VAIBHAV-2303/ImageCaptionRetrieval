import torch
import matplotlib.pyplot as plt
import json
from model import ImgEncoder, SentenceEncoder
from utils import get_hot, triplet_loss_cap, triplet_loss_img, build_dictionary
from PIL import Image
import random
import numpy as np
import pickle
import sys

# Parameters
dim_image = 4096
dim = 1024
dim_word = 300

# Loading dictionary
with open('worddict.pkl', 'rb') as f:
	worddict = pickle.load(f)
print('Loaded dictionary')

# Loading metadata
with open('./data/sis/val.story-in-sequence.json') as json_file:
	metadata = json.load(json_file)

# Loading image features
with open('data/val/image_features.pkl', 'rb') as f:
	ims = pickle.load(f)
	ims = ims/torch.norm(ims, dim=1, p=2).reshape(-1, 1)

# Loading models
ImgEncoder = ImgEncoder(dim_image, dim).cuda()
ImgEncoder.load_state_dict(torch.load('ImgEncoder.pt'))
SentenceEncoder = SentenceEncoder(len(worddict)+2, dim_word, dim).cuda()
SentenceEncoder.load_state_dict(torch.load('SentenceEncoder.pt'))
print('Models loaded')

# Encoding images
encoded_imgs = ImgEncoder(ims)
print(encoded_imgs.shape, len(metadata['annotations']))
ind = int(sys.argv[1])
cos = torch.nn.CosineSimilarity()

for i in range(5):
	hot = get_hot(metadata['annotations'][5*ind + i][0]['text'], worddict)
	encoded_cap = SentenceEncoder(hot).repeat(encoded_imgs.shape[0], 1)
	S = cos(encoded_cap, encoded_imgs)
	pred = Image.open('/ssd_scratch/cvit/vaibhav.garg/val/' + metadata['annotations'][S.argsort().cpu().numpy()[::-1][0]][0]['photo_flickr_id'] + '.jpg')
	true = Image.open('/ssd_scratch/cvit/vaibhav.garg/val/' + metadata['annotations'][5*ind + i][0]['photo_flickr_id'] + '.jpg')

	# plotting
	print('Caption:', metadata['annotations'][5*ind + i][0]['text'])
	plt.subplot(5, 2, 2*i+1)
	plt.imshow(np.asarray(pred))
	plt.subplot(5, 2, 2*i+2)
	plt.imshow(np.asarray(true))

plt.savefig(str(ind), dpi=500)