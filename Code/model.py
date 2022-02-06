import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)

class FlowNet(nn.Module):  ### for 5cp data set with dimension 51 by 51 by 51
	def __init__(self):
		super(FlowNet,self).__init__()
		self.conv1 = nn.Conv3d(1,512,3)
		self.conv2 = nn.Conv3d(512,256,3)
		self.conv3 = nn.Conv3d(256,128,3)
		self.conv4 = nn.Conv3d(128,64,3)
		self.conv5 = nn.Conv3d(64,1,3)
		self.fc1 = nn.Linear(41*41*41,1024)
		self.fc = nn.Linear(1024,1024)
		self.fc2 = nn.Linear(1024,61*61*61)
		self.deconv1 = nn.Conv3d(1,512,3)
		self.deconv2 = nn.Conv3d(512,256,3)
		self.deconv3 = nn.Conv3d(256,128,3)
		self.deconv4 = nn.Conv3d(128,64,3)
		self.deconv5 = nn.Conv3d(64,1,3)
		self.batch1 = nn.BatchNorm3d(512)
		self.batch2 = nn.BatchNorm3d(256)
		self.batch3 = nn.BatchNorm3d(128)
		self.batch4 = nn.BatchNorm3d(64)

	def forward(self,x):
		return self.decoder(self.encoder(x))

	def encoder(self,x):
		x = self.conv1(x)
		x = self.batch1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = self.batch2(x)
		x = F.relu(x)
		x = self.conv3(x)
		x = self.batch3(x)
		x = F.relu(x)
		x = self.conv4(x)
		x = self.batch4(x)
		x = F.relu(x)
		x = F.relu(self.conv5(x))
		x = x.view(1,41*41*41)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc(x))
		return x


	def decoder(self,f):
		x = F.relu(self.fc2(f))
		x = x.view((1,1,61,61,61))
		x = self.deconv1(x)
		x = self.batch1(x)
		x = F.relu(x)
		x = self.deconv2(x)
		x = self.batch2(x)
		x = F.relu(x)
		x = self.deconv3(x)
		x = self.batch3(x)
		x = F.relu(x)
		x = self.deconv4(x)
		x = self.batch4(x)
		x = F.relu(x)
		x = F.sigmoid(self.deconv5(x))
		return x

