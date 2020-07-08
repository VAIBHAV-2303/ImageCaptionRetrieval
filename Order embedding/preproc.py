import torch
import numpy as np
import torchvision.models.vgg as models
import torchvision.transforms as transforms
from PIL import Image
import json
import pickle

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
dim_image = 4096
transform_pipeline = transforms.Compose([transforms.ToTensor(), 
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
mode = 'Val'

print('Processing begins')
with open('data/coco/' + mode + '/annotations/' + mode + '.json') as json_file:
	metadata = json.load(json_file)

caps = []
imgs = torch.zeros(len(metadata['images']), dim_image).cuda()
vgg19 = models.vgg19(pretrained=True).cuda()	
for i in range(len(metadata['images'])):

	print('Processing image number: ' + str(i) + '/' + str(len(metadata['images'])))

	# Reading image and flipping
	img_path = 'data/coco/' + mode + '/images/' + metadata['images'][i]['file_name']
	I = Image.open(img_path)
	
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
	features = features.mean(dim=0)
	
	imgs[i] = features
	
	# Captions
	img_id = metadata['images'][i]['id']
	cnt = 0
	for j in range(len(metadata['annotations'])):
		if metadata['annotations'][j]['image_id'] == img_id:
			cnt += 1
			caps.append(metadata['annotations'][j]['caption'])
			if cnt == 5:
				break

print('Images shape:', imgs.shape, 'Length of captions:', len(caps))

# Saving
with open('data/coco/' + mode + '/image_features.pkl', 'wb') as f:
	pickle.dump(imgs, f)
with open('data/coco/' + mode + '/captions.pkl', 'wb') as f:
	pickle.dump(caps, f)
print('Saved')