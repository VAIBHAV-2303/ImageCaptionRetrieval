import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Image Encoder
class ImgEncoder(nn.Module):

	def __init__(self, EMBEDDING_SIZE, COMMON_SIZE):
		super(ImgEncoder, self).__init__()
		self.linear = nn.Linear(EMBEDDING_SIZE, COMMON_SIZE)

	def forward(self, x):
		return self.linear(x)

class SentenceEncoder(nn.Module):

	def __init__(self, VOCAB_SIZE, WORD_EMBEDDING_SIZE, COMMON_SIZE):
		super(SentenceEncoder, self).__init__()
		self.embed = nn.Linear(VOCAB_SIZE, WORD_EMBEDDING_SIZE)
		self.encoder = nn.GRU(WORD_EMBEDDING_SIZE, COMMON_SIZE)

	def forward(self, x):
		x = self.embed(x)
		o, h = self.encoder(x.reshape(x.shape[0], 1, x.shape[1]))
		return h.reshape(1, -1)