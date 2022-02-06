import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import *
from dataio import *
from torch.utils.data import DataLoader

def train(opts):
	model = FlowNet()
	if opts.cuda:
		model.cuda()
	model.apply(weights_init)
	objects = GetObjets(opts.train_data_path,opts.x,opts.y,opts.z,opts.data_type)
	train_loader = DataLoader(dataset=objects,batch_size=opts.batch_size,shuffle=True)
	optimizer = optim.Adam(model.parameters(), lr=opts.lr)
	loss = 0.0
	bce = nn.BCELoss()
	for itera in range(1,opts.epochs+1):
		for batch_idx,data in enumerate(train_loader):
			if opts.cuda:
				data = data.cuda()
			optimizer.zero_grad()
			result = model(data)
			loss = bce(result,data)
			loss.backward()
			optimizer.step()
		if itera%opts.checkpoint == 0:
			torch.save(model.state_dict(),opts.model_path+str(itera)+'.pth')



def inf(opts):
	model = FlowNet()
	model.load_state_dict(torch.load(opt.model_path+str(opts.epochs)+'.pth'))
	if opts.cuda:
		model.cuda()
	objects = GetObjets(opts.infer_data_path,opts.x,opts.y,opts.z,opts.data_type)
	train_loader = DataLoader(dataset=objects,batch_size=opts.batch_size,shuffle=False)
	model.eval()
	idx = 1
	with torch.no_grad():
		for batch_idx,data in enumerate(train_loader):
			if opts.cuda:
				data = data.cuda()
			features = model.encoder(data)
			features = features.detach().cpu().numpy()
			for k in range(0,len(features)):
				features[k].tofile(opts.infer_data_path+'{:04d}'.format(idx)+'.dat',format='<f')
				idx += 1


