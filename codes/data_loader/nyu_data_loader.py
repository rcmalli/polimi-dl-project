import numpy as np
import h5py


class NYUDataLoader:
	def __init__(self, config):
		self.config = config
		print("Loading the data")
		# Large .mat file can be loaded only by h5py
		f = h5py.File(self.config.data_path)
		# Load input images
		self.input = np.array(list(f['images']))

		numofimages = self.input.shape[0]

		# Reshaping for network
		self.input = np.reshape(self.input,[self.input.shape[0],self.input.shape[2],self.input.shape[3],self.input.shape[1]])		
		# Load output images
		self.y = np.array(list(f['depths']))
		# Reshaping for network		
		self.y = np.reshape(self.y,[self.y.shape[0],self.y.shape[1],self.y.shape[2],1])		

		# Index list of unused images by batch
		self.batch_list = np.array(range(numofimages))

	def next_batch(self, batch_size):
		# This batch method chooses k data point from n without replacement
		# If not enough input left in the "unused" list, refill
		if(len(self.batch_list) <batch_size):
			self.batch_list = np.array(range(len(self.input)))
		# Choose k point from the list
		idx = (np.random.choice(self.batch_list, batch_size,replace=False))
		self.batch_list = np.setdiff1d(self.batch_list,idx)
		# Return iterative object
		yield self.input[idx], self.y[idx]

