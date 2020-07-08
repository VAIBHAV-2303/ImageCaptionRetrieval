import torch
import torchvision.models.vgg as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import json
from PIL import Image
from PIL import ImageFile
import numpy as np
import pickle
import sys

# Function definitions
def get10Crop(I, flippedI):
	ret = torch.zeros(10, 3, 224, 224)
	
	# Normal crops
	ret[0] = I[0, :, :224, :224]
	ret[1] = I[0, :, :224, -224:]
	ret[2] = I[0, :, -224:, :224]
	ret[3] = I[0, :, -224:, -224:]
	ret[4] = I[0, :, I.shape[2]//2 - 112:I.shape[2]//2 + 112, I.shape[3]//2 - 112:I.shape[3]//2 + 112]

	# Flipped crops
	ret[5] = flippedI[0, :, :224, :224]
	ret[6] = flippedI[0, :, :224, -224:]
	ret[7] = flippedI[0, :, -224:, :224]
	ret[8] = flippedI[0, :, -224:, -224:]
	ret[9] = flippedI[0, :, flippedI.shape[2]//2 - 112:flippedI.shape[2]//2 + 112, flippedI.shape[3]//2 - 112:flippedI.shape[3]//2 + 112]

	return ret

# Parameters
ImageFile.LOAD_TRUNCATED_IMAGES = True
dim_image = 4096
dim = 1024
dim_word = 300
transform_pipeline = transforms.Compose([transforms.Resize((224, 224)), \
											transforms.ToTensor(), \
											transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Model
vgg19 = models.vgg19(pretrained=True).cuda()
print('Models loaded')

# Loading metadata
with open('./data/sis/train.story-in-sequence.json') as json_file:
	metadata = json.load(json_file)
data_size = len(metadata['annotations'])

caps = []
for i in range(data_size):
	print('Image Number:', i)

	img_id = metadata['annotations'][i][0]['photo_flickr_id']
	try:
		I = Image.open('/ssd_scratch/cvit/vaibhav.garg/Train/train/' + img_id + '.jpg')
	except:
		continue

	# Converting to average COCO size
	if I.size[0] > 640:
		s = 640/I.size[0]
		I = I.resize((int(I.size[0]*s), int(I.size[1]*s)))
	if I.size[1] > 640:
		s = 640/I.size[1]
		I = I.resize((int(I.size[0]*s), int(I.size[1]*s)))
	
	# Converting to RGB
	if I.mode != 'RGB':
		I = I.convert('RGB')

	# Resizing to get proper crops
	if I.size[0] < 224:
		I = I.resize((224, I.size[1]))
	if I.size[1] < 224:
		I = I.resize((I.size[0], 224))

	flippedI = I.transpose(Image.FLIP_LEFT_RIGHT)

	I = transform_pipeline(I).unsqueeze(0)
	flippedI = transform_pipeline(flippedI).unsqueeze(0)

	Icrops = get10Crop(I, flippedI)
	features = vgg19.features(torch.FloatTensor(Icrops).cuda()).data
	features = vgg19.avgpool(features).data.reshape(10, 512*7*7)
	features = vgg19.classifier[:-1](features).data
	features = features.mean(dim=0).reshape(1, -1)

	try:
		img_features = torch.cat((img_features, features), 0)
	except:
		img_features = features

	# Captions
	caps.append(metadata['annotations'][i][0]['text'])

# Storage in pickle files
print('Image features shape:', img_features.shape)
with open('data/train/image_features.pkl', 'wb') as f:
	pickle.dump(img_features, f)

print('Captions length:', len(caps))
with open('data/train/captions.pkl', 'wb') as f:
	pickle.dump(caps, f)