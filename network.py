"""
	This file contains a neural network module for us to
	define our actor and critic networks in PPO.
"""

import torch
from torch import nn
import numpy as np


class FeedForwardNN(nn.Module):
	"""
		A standard in_dim-64-64-out_dim Feed Forward Neural Network.
	"""
	def __init__(self, in_dim, out_dim, steps=1, activation=None):
		"""
			Initialize the network and set up the layers.

			Parameters:
				in_dim - input dimensions as an int
				out_dim - output dimensions as an int

			Return:
				None
		"""
		super(FeedForwardNN, self).__init__()

		self.activation = activation

		self.network = nn.Sequential(
			nn.Linear(in_dim, 64),
			nn.ReLU(),
			nn.Linear(64, 64),
			nn.ReLU(),
			nn.Linear(64, 64),
			nn.ReLU(),
			nn.Linear(64, out_dim),
		)
	def forward(self, obs):
		"""
			Runs a forward pass on the neural network.

			Parameters:
				obs - observation to pass as input

			Return:
				output - the output of our forward pass
		"""
		# Convert observation to tensor if it's a numpy array
		if isinstance(obs, np.ndarray):
			obs = torch.tensor(obs, dtype=torch.float)

		output = self.network(obs)

		if self.activation is not None:
			output = self.activation(output)

		return output


class StateEncoderHead(nn.Module):
	def __init__(self, in_dim, out_dim):
		super().__init__()

		self.network = nn.Sequential(
			nn.Linear(in_dim, 32),
			nn.ReLU(),
			nn.Linear(32, out_dim),
			nn.ReLU(),
		)

	def forward(self, obs):
		if isinstance(obs, np.ndarray):
			obs = torch.tensor(obs, dtype=torch.float)

		return self.network(obs)


class FeedForwardMultiStepNN(nn.Module):
	"""
		A standard in_dim-64-64-out_dim Feed Forward Neural Network.
	"""
	def __init__(self, obs_dim, acts_dim, out_dim, steps, state_encoder=None, state_head_size=16, activation=None):
		"""
			Initialize the network and set up the layers.

			Parameters:
				in_dim - input dimensions as an int
				out_dim - output dimensions as an int
				steps - amount of sequential steps fed into network

			Return:
				None
		"""
		super(FeedForwardMultiStepNN, self).__init__()

		if state_encoder is not None:
			self.stateEncoderHead = state_encoder
			self.stateEncoderHead.requires_grad_(False)

		self.stateEncoderHead = StateEncoderHead(obs_dim, state_head_size)
		self.activation = activation

		self.network = nn.Sequential(
			nn.Linear(state_head_size * steps + acts_dim, 64),
			nn.ReLU(),
			nn.Linear(64, 64),
			nn.ReLU(),
			nn.Linear(64, out_dim)
		)

	def forward(self, obs, acts=None):
		"""
			Runs a forward pass on the neural network.

			Parameters:
				obs - observation to pass as input, shape: (B, T, C), where T - sequential steps, C - in_dim

			Return:
				output - the output of our forward pass
		"""
		# Convert observation to tensor if it's a numpy array
		if isinstance(obs, np.ndarray):
			obs = torch.tensor(obs, dtype=torch.float)

		if isinstance(acts, np.ndarray):
			acts = torch.tensor(acts, dtype=torch.float)

		has_batch_dim = True

		shape = obs.shape
		if len(shape) == 2:
			has_batch_dim = False
			obs = obs.unsqueeze(0)
			shape = obs.shape

		if acts is not None and acts.shape[0] != shape[0]:
			raise ValueError

		obs = obs.view(-1, shape[2])
		obs = self.stateEncoderHead(obs)
		obs_encoded = obs.view(shape[0], -1)

		if acts is not None:
			obs_encoded = torch.cat([obs_encoded, acts], dim=1)

		output = self.network(obs_encoded)

		if self.activation:
			output = self.activation(output)

		return output if has_batch_dim else output[0]
