import tensorflow as tf
import numpy as np
import random
import time
import pickle
from collections import deque
from game import Game
from model import PolicyValueNet
from random_player import RandomPlayer
from state import State
from mcts import MCTS

class Agent(object):
	def __init__(self):
		self.width = 6
		self.height = 6
		self.max_state_size = 3
		self.win_contition = 4
		self.batch_size = 128
		self.max_game_count = 300000
		self.max_data_size = 1280
		self.max_training_loop_count = 1
		self.learning_rate = 0.001
		self.simulation_count = 400
		self.c_puct = 0.96

	def train(self, start_step=0):
		model = PolicyValueNet(self.height, self.width, self.max_state_size,
				model_path=None, train_mode=True)
		state =  State(self.height, self.width, self.max_state_size, self.win_contition)
		game = Game(state)
		player = MCTS(model.policy_value, self.simulation_count)
		data_queue = deque(maxlen=self.max_data_size)
		prev_loss = 10

		game_count = 0
		training_step = 0
		new_data_count = 0
		while game_count < self.max_game_count:
			winner, game_states, action_probs, values = \
				game.play(player, player)

			if winner == 2:
				continue
			"""
			print("")
			print(state.get_game_state())
			if winner == 1:
				print("Black Win")
			else:
				print("White Win")
			"""


			augmented_states, augmented_actions, augmented_values = \
				self.augmenting_data(game_states, action_probs, values)
			play_data = list(zip(augmented_states, augmented_actions,
				augmented_values))[:]
			data_queue.extend(play_data)
			loss = 10
			new_data_count += len(augmented_states)

			if len(data_queue) == self.max_data_size and new_data_count > 128:
				loss, value_mse, policy_entropy = 0.0, 0.0, 0.0
				mini_batch = random.sample(data_queue, self.batch_size)
				states_batch = [d[0] for d in mini_batch]
				actions_batch = [d[1] for d in mini_batch]
				values_batch = [d[2] for d in mini_batch]
				for i in range(self.max_training_loop_count):
					loss, value_mse, policy_entropy = \
						model.train(states_batch, actions_batch, values_batch,
							training_step, self.learning_rate)
					training_step += 1
				new_data_count = 0
				#data_queue.clear()
				if prev_loss > loss:
					prev_loss = loss
					print("game_count %d, training_step %d" %(game_count, training_step))
					print("loss %.5f, value %.5f, entropy %.5f" %(loss,value_mse,policy_entropy))
					model.save_model("data/best_model", None)

			"""
			if (training_step+1) == 100:
				self.learning_rate *= 0.1

			if (training_step+1) == 300:
				self.learning_rate *= 0.1
			"""

			game_count += 1

	def make_init_data_queue(self):
		model = PolicyValueNet(self.height, self.width, self.max_state_size,
				model_path=None, train_mode=False)
		state =  State(self.height, self.width, self.max_state_size, self.win_contition)
		game = Game(state)
		player = MCTS(model.policy_value, self.simulation_count)
		data_queue = deque(maxlen=self.max_data_size)
		while len(data_queue) < self.max_data_size:
			winner, game_states, action_probs, values = \
				game.play(player, player)

			if winner == 2:
				continue

			augmented_states, augmented_actions, augmented_values = \
				self.augmenting_data(game_states, action_probs, values)
			play_data = list(zip(augmented_states, augmented_actions,
				augmented_values))[:]
			data_queue.extend(play_data)

		with open('data/init_data_queue.pickle', 'wb') as f:
    		pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def load_init_data_queue(self, target):
    	with open('data/init_data_queue.pickle', 'rb') as f:
    		target = pickle.load(f)

	def augmenting_data(self, states, action_probs, values):
		augmented_states = []
		augmented_actions = []
		augmented_values = []

		for i in range(0, len(states)):
			state = states[i]
			action_prob = action_probs[i].reshape((self.height,
				self.width))
			value = values[i]

			for j in [1, 2, 3, 4]:
				rotate_state = []
				for k in state:
					rotate_state.append(np.rot90(k, j))
				rotate_action = np.rot90(action_prob, j)
				augmented_states.append(rotate_state)
				augmented_actions.append(rotate_action.reshape(
					self.height * self.width))
				augmented_values.append(value)

			flip_state = np.fliplr(state)
			flip_action = np.fliplr(action_prob)
			augmented_states.append(flip_state)
			augmented_actions.append(flip_action.reshape(
				self.height * self.width))
			augmented_values.append(value)

		return augmented_states, augmented_actions, augmented_values