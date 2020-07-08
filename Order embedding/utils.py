import torch
import numpy as np

def build_dictionary(text):
    """
    Build a dictionary (mapping of tokens to indices)
    text: list of sentences (pre-tokenized)
    """
    wordcount = {}
    for cc in text:
        words = cc.split()
        for w in words:
            if w not in wordcount:
                wordcount[w] = 0
            wordcount[w] += 1
    words = list(wordcount.keys())
    freqs = list(wordcount.values())
    sorted_idx = np.argsort(freqs)[::-1]

    worddict = {}
    for idx, sidx in enumerate(sorted_idx):
        worddict[words[sidx]] = idx+2  # 0: <eos>, 1: <unk>

    return worddict

def get_hot(cap, worddict):
	x = np.zeros((len(cap.split())+1, len(worddict)+2))

	r = 0
	for w in cap.split():
		if w in worddict:
			x[r, worddict[w]] = 1
		else:
			# Unknown word/character
			x[r, 1] = 1
		r += 1
	# EOS
	x[r, 0] = 1

	return torch.from_numpy(x).float().cuda()

def Score(caps, imgs):
	z = torch.zeros(caps.shape).cuda()
	return -torch.sum(torch.max(z, caps-imgs)**2, dim=1)

def triplet_loss_img(anchor, positive, negative, margin):
	ps = Score(positive, anchor)
	pn = Score(negative, anchor)
	z = torch.zeros(ps.shape).cuda()
	return torch.sum(torch.max(z, margin - ps + pn))

def triplet_loss_cap(anchor, positive, negative, margin):
	ps = Score(anchor, positive)
	pn = Score(anchor, negative)
	z = torch.zeros(ps.shape).cuda()
	return torch.sum(torch.max(z, margin - ps + pn))
	