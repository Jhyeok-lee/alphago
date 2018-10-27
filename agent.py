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
		self.episode_num = 3000
		self.width = 7
		self.height = 7
		self.max_state_size = 4
		self.batch_size = 512
		self.learning_rate = 0.001
		self.simulation_count = 400
		self.c_puct = 0.96

	def train(self, start_step=0):
		if start_step != 0:
			model_path = 'data/model-' + str(start_step-1)
		else:
			model_path = None
		model = PolicyValueNet(self.height, self.width, self.max_state_size,
				model_path=model_path, train_mode=True)
		state =  State(self.height, self.width, self.max_state_size)
		game = Game(state)
		player1 = MCTS(model.policy_value, self.simulation_count)
		player2 = MCTS(model.policy_value, self.simulation_count)
		data = deque(maxlen=10000)
		prev_loss = 10
		prev_value_mse = 10
		prev_policy_entropy = 10

		episode = 0
		if start_step > 0:
			episode = start_step - 1
		while episode < self.episode_num + start_step:
			black, white = None, None
			if episode % 2 == 0:
				black = player1
				white = player2
			else:
				black = player2
				white = player1

			winner, game_states, action_probs, values = \
				game.play(black, white)

			if winner == 2:
				continue
			print("")
			print(state.get_game_state())
			if winner == 1:
				print("Black Win")
			else:
				print("White Win")

			augmented_states, augmented_actions, augmented_values = \
				self.augmenting_data(game_states, action_probs, values)
			play_data = list(zip(augmented_states, augmented_actions,
				augmented_values))[:]
			data.extend(play_data)
			loss = 10
			value_mse = 10
			policy_entropy = 10

			if (episode+1) % 5 == 0:
				mini_batch = random.sample(data, self.batch_size)
				states_batch = [d[0] for d in mini_batch]
				actions_batch = [d[1] for d in mini_batch]
				values_batch = [d[2] for d in mini_batch]
				loss, value_mse, policy_entropy = \
					model.train(states_batch, actions_batch, values_batch, episode+1,
						self.learning_rate)
				print("loss : ", loss)
				print("value : ", value_mse)
				print("entropy : ", policy_entropy)

			if (episode+1) % 100 == 0:
				model_path = "data/model"
				model.save_model(model_path, episode+1)
				if prev_loss > loss and \
					prev_policy_entropy > policy_entropy and \
					prev_value_mse > value_mse:
					prev_loss = loss
					prev_policy_entropy = policy_entropy
					prev_value_mse = value_mse
					model.save_model("best/model", None)

			episode += 1

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