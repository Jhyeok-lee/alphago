import tensorflow as tf
import numpy as np
import random
import time
from collections import deque
from game import Game
from model import PolicyValueNet
from random_player import RandomPlayer
from state import State
from mcts import MCTS

class Agent(object):
	def __init__(self):
		self.width = 8
		self.height = 8
		self.max_state_size = 3
		self.win_contition = 4
		self.batch_size = 512
		self.max_game_count = 300000
		self.max_training_loop_count = 20
		self.max_data_size = 10000
		self.learning_rate = 0.01
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

			if len(data) == self.max_data_size:
				loss, value_mse, policy_entropy = 0, 0, 0
				for i in range(self.max_training_loop_count):
					mini_batch = random.sample(data_queue, self.batch_size)
					states_batch = [d[0] for d in mini_batch]
					actions_batch = [d[1] for d in mini_batch]
					values_batch = [d[2] for d in mini_batch]
					loss, value_mse, policy_entropy = \
						model.train(states_batch, actions_batch, values_batch,
							training_step, self.learning_rate)
					training_step += 1
				print("loss : ", loss)
				print("value : ", value_mse)
				print("entropy : ", policy_entropy)
				model.save_model("data/model", training_step+1)
				if prev_loss > loss:
					model.save_model("data/best_model", None)

			if (training_step+1) == 400:
				self.learning_rate /= 10.0

			if (training_step+1) == 2000:
				self.learning_rate /= 10.0

			game_count += 1

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