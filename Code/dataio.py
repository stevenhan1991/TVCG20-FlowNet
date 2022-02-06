import numpy as np
import torch

def GetObjects(folder,x,y,z,data_type):
	dat = []
	for r,d,f in os.walk(folder):
		for file in f:
			if data_type in file:
				dat.append(os.path.join(r,file))
	dat.sort()
	objects = np.zeros((len(dat),1,x,y,z))
	for i in range(0,len(dat)):
		d = np.fromfile(dat[i],dtye='<f')
		d = d.reshape(z,y,x).transpose()
		objects[i] = d

	objects = torch.FlaotTensor(objects)
	return objects