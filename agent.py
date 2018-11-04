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
		self.batch_size = 256
		self.max_game_count = 150000
		self.max_data_size = 5120
		self.max_training_loop_count = 5
		self.learning_rate = 0.001
		self.simulation_count = 400
		self.value_head_weight = 1.0
		self.policy_entropy_weight = 1.0
		self.c_puct = 5
		self.vailidation_test_count = 100

	def train(self, start_step=0):
		model = PolicyValueNet(self.height, self.width, self.max_state_size,
				self.value_head_weight, self.policy_entropy_weight,
				model_path=None, train_mode=True)
		state =  State(self.height, self.width, self.max_state_size, self.win_contition)
		game = Game(state)
		current_player = MCTS(model.policy_value, self.simulation_count,
			c_puct=self.c_puct)
		best_player = MCTS(model.best_policy_value, self.simulation_count,
			c_puct=self.c_puct)
		data_queue = deque(maxlen=self.max_data_size)
		prev_loss = 10
		prev_entropy = 10
		prev_value = 10

		game_count = 0
		training_step = 0
		new_data_count = 0

		while True:
			game_count += 1
			
			# collecting data
			if len(data_queue) == 0:
				print("start collecting data")
			winner, game_states, action_probs, values = None, None, None, None
			if game_count%2 == 0:
				winner, game_states, action_probs, values = \
					game.play(best_player, best_player)
			else:
				winner, game_states, action_probs, values = \
					game.play(best_player, best_player)

			if winner == 2:
				continue

			augmented_states, augmented_actions, augmented_values = \
				self.augmenting_data(game_states, action_probs, values)
			play_data = list(zip(augmented_states, augmented_actions,
				augmented_values))[:]
			data_queue.extend(play_data)

			if len(data_queue) < self.max_data_size:
				continue

			# training
			print("%d games : end collecting data, start training" %(game_count))
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
			data_queue.clear()
			print("end training : step %d, loss %f, value_mse %f, policy_entropy %f"
				%(training_step, loss, value_mse, policy_entropy))
			model.save_model("data/model", training_step)

			# vailidation test
			print("start validation test")
			current_win = 0
			best_win = 0
			for i in range(self.vailidation_test_count):
				if i%2 == 0:
					winner, game_states, action_probs, values = \
						game.play(current_player, best_player)
					if winner == 1:
						best_win += 1
					else:
						current_win += 1
				else:
					winner, game_states, action_probs, values = \
						game.play(best_player, current_player)
					if winner == 0:
						best_win += 1
					else:
						current_win += 1
			print("end validation test : current_win %d, best_win %d"
				%(current_win, best_win))
			if best_win > current_win:
				print("new best model")
				model.update_best_model()
				model.save_model("data/best_model")

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