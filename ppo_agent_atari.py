import torch
import torch.nn as nn
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from replay_buffer.gae_replay_buffer import GaeSampleMemory
from base_agent import PPOBaseAgent
from models.atari_model import AtariNet
import gym
from gym.wrappers import GrayScaleObservation
from gym.wrappers import ResizeObservation
from gym.wrappers import FrameStack
import custom_env
import random

class AtariPPOAgent(PPOBaseAgent):
	
	def __init__(self, config):
		super(AtariPPOAgent, self).__init__(config)
		### TODO ###
		# initialize env
		# self.env = ???
		

		### TODO ###
		# initialize test_env
		# self.test_env = ???

		env = gym.make(config["env_id"], render_mode='rgb_array')
		# env = gym.make(config["env_id"], render_mode='human')
		env = custom_env.ReacherRewardWrapper(env)
		env = ResizeObservation(env, 84)
		env = GrayScaleObservation(env)
		env = FrameStack(env, 4)
		self.env = env

		test_env = gym.make(config["env_id"], render_mode='rgb_array')
		# test_env.metadata['render_fps']=30

		# test_env = gym.make(config["env_id"], render_mode='human')
		test_env = gym.wrappers.RecordVideo(test_env, 'video')

		def seed_everything(self, seed):
			random.seed(seed)
			np.random.seed(seed)
			os.environ['PYTHONHASHSEED'] = str(seed)
			torch.manual_seed(seed)
			torch.cuda.manual_seed(seed)
			torch.backends.cudnn.deterministic = True
			self.test_env.seed(seed)

		

		test_env = ResizeObservation(test_env, 84)
		test_env = GrayScaleObservation(test_env)
		test_env = FrameStack(test_env, 4)
		self.test_env = test_env

		seed_everything(self, 20)

		self.net = AtariNet(self.env.action_space.n)
		self.net.to(self.device)
		self.lr = config["learning_rate"]
		self.update_count = config["update_ppo_epoch"]
		self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)
		
	def decide_agent_actions(self, observation, eval=False):
		### TODO ###
		# add batch dimension in observation
		# get action, value, logp from net
		
		state = torch.tensor(np.array(observation)).view(1, 4, 84, 84).to(self.device)
		#print("state:", state)

		if eval:
			with torch.no_grad():
				action, value, logp, _ = self.net(state, eval=True)
		else:
			action, value, logp, _ = self.net(state)
		
		return action.detach().cpu(), value.detach().cpu(), logp.detach().cpu()


	def update(self):
		loss_counter = 0.0001
		total_surrogate_loss = 0
		total_v_loss = 0
		total_entropy = 0
		total_loss = 0

		batches = self.gae_replay_buffer.extract_batch(self.discount_factor_gamma, self.discount_factor_lambda)
		sample_count = len(batches["action"])
		batch_index = np.random.permutation(sample_count)
		
		observation_batch = {}
		for key in batches["observation"]:
			observation_batch[key] = batches["observation"][key][batch_index]
		action_batch = batches["action"][batch_index]
		return_batch = batches["return"][batch_index]
		adv_batch = batches["adv"][batch_index]
		v_batch = batches["value"][batch_index]
		logp_pi_batch = batches["logp_pi"][batch_index]

		for _ in range(self.update_count):
			# tmp = 0
			for start in range(0, sample_count, self.batch_size):
				ob_train_batch = {}
				for key in observation_batch:
					ob_train_batch[key] = observation_batch[key][start:start + self.batch_size]
				ac_train_batch = action_batch[start:start + self.batch_size]
				return_train_batch = return_batch[start:start + self.batch_size]
				adv_train_batch = adv_batch[start:start + self.batch_size]
				v_train_batch = v_batch[start:start + self.batch_size]
				logp_pi_train_batch = logp_pi_batch[start:start + self.batch_size]

				v_train_batch = torch.from_numpy(v_train_batch)
				v_train_batch = v_train_batch.to(self.device, dtype=torch.float32)

				ob_train_batch = torch.from_numpy(ob_train_batch["observation_2d"])
				ob_train_batch = ob_train_batch.to(self.device, dtype=torch.float32)
				ac_train_batch = torch.from_numpy(ac_train_batch)
				ac_train_batch = ac_train_batch.to(self.device, dtype=torch.long)
				adv_train_batch = torch.from_numpy(adv_train_batch)
				adv_train_batch = adv_train_batch.to(self.device, dtype=torch.float32)
				logp_pi_train_batch = torch.from_numpy(logp_pi_train_batch)
				logp_pi_train_batch = logp_pi_train_batch.to(self.device, dtype=torch.float32)
				return_train_batch = torch.from_numpy(return_train_batch)
				return_train_batch = return_train_batch.to(self.device, dtype=torch.float32)

				### TODO ###
				# calculate loss and update network
				# ???, ???, ???, ??? = self.net(...)
				action, critic_value, curr_log_pi, distribution = self.net(ob_train_batch)

				# calculate policy loss
				# ratio = ???
				# 128, 1
				# print(ac_train_batch.shape)

				curr = distribution.log_prob(ac_train_batch.squeeze(dim=1))
				# print('curr:', curr.shape)
				# print(distribution)
				# curr = []
				# for a in ac_train_batch:
				# 	for b in a :
				# 		curr.append(distribution.log_prob(b))
				# print(len(curr[0]),'AAAA', len(curr[1]))
				
				# print('curr shape:',curr.shape)
				# print(curr)
				# print('logp_pi_train_batch.shape:', logp_pi_train_batch.shape)
				# print(logp_pi_train_batch)
				
				ratio = torch.exp(curr - logp_pi_train_batch.squeeze(dim=1))
				# if tmp == 0 and start == 0:
				# 	print(ratio)
				# 	tmp+=1
				# print("ratio shape:", ratio.shape)
				with torch.no_grad():
					clip_ratio = ratio.clamp(min=1-self.clip_epsilon, max=1+self.clip_epsilon)
					
				
				surrogate_loss = torch.max(-ratio * adv_train_batch, -clip_ratio* adv_train_batch).mean()
				

				# calculate value loss
				value_criterion = nn.MSELoss()
				
				v_loss = value_criterion(critic_value, return_train_batch).mean()
				
				

				# calculate total loss
				entropy = distribution.entropy().mean()
				
				loss = surrogate_loss + self.value_coefficient * v_loss - self.entropy_coefficient * entropy
				# update network
				self.optim.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(self.net.parameters(), self.max_gradient_norm)
				self.optim.step()

				total_surrogate_loss += surrogate_loss.item()
				total_v_loss += v_loss.item()
				total_entropy += entropy.item()
				total_loss += loss.item()
				loss_counter += 1

		self.writer.add_scalar('PPO/Loss', total_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Surrogate Loss', total_surrogate_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Value Loss', total_v_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Entropy', total_entropy / loss_counter, self.total_time_step)
		print(f"Loss: {total_loss / loss_counter}\
			\tSurrogate Loss: {total_surrogate_loss / loss_counter}\
			\tValue Loss: {total_v_loss / loss_counter}\
			\tEntropy: {total_entropy / loss_counter}\
			")
	

	

