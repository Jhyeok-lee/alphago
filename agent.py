import tensorflow as tf
import numpy as np
import random
import time
from collections import deque
from game import Game
from model import PolicyValueNet
from random_player import RandomPlayer

class Agent(object):
	def __init__(self):
		self.episode_num = 5000
		self.width = 11
		self.height = 11
		self.max_state_size = 7
		self.batch_size = 512

	def train():
		model = PolicyValueNet(self.height, self.width, self.max_state_size)
		game = Game(self.height, self.width, self.max_state_size)
		player1 = RandomPlayer()
		player2 = RandomPlayer()
		data = deque(maxlen=10000)

		for episode in range(self.episode_num):
			winner, game_states, action_probs, values = \
				game.play(player1, player2)

			if winner == 2:
				continue

			augmented_states, augmented_actions, augmented_values = \
				augmenting_data(game_states, action_probs, values)
			play_data = list(zip(augmented_states, augmented_actions,
				augmented_values))[:]
			data.extend(play_data)

			if (episode+1) % 25 == 0:
				mini_batch = random.sample(data, self.batch_size)
				states_batch = [d[0] for d in mini_batch]
				actions_batch = [d[1] for d in mini_batch]
				values_batch = [d[2] for d in mini_batch]
				model.train(states_batch, actions_batch, values_batch)

			if (episode+1) % 100 == 0:
				model_path = "model/" + str(episode+1) + ".model"
				model.save_model(model_path, episode+1)

	def augmenting_data(states, action_probs, values):
		augmented_states = []
		augmented_actions = []
		augmented_values = []

		for i in range(0, len(states)):
			state = states[i]
			action_prob = action_probs[i]
			value = values[i]

			for j in [1, 2, 3, 4]:
				rotate_state = []
				for k in state:
					rotate_state.append(np.rot90(k, j))
				rotate_action = np.rot90(action_prob, j)
				augmented_states.append(rotate_state)
				augmented_actions.append(rotate_action)
				augmented_values.append(value)

			flip_state = np.fliplr(state)
			flip_action = np.fliplr(action_prob)
			augmented_states.append(flip_state)
			augmented_actions.append(flip_action)
			augmented_values.append(value)

		return augmented_states, augmented_actions, augmented_values