import numpy as np
import random

class RandomPlayer:

	def __init__(self, height, width, max_state_size=7):
		self.height = height
		self.width = width
		self.max_state_size = max_state_size

	def get_action(self, game_state):
		player = game_state[0][0][0]
		white_last_state = game_state[self.max_state_size]
		black_last_state = game_state[self.max_state_size*2]
		last_state = white_last_state + black_last_state

		action_probs = np.zeros((self.height, self.width))
		r = random.randrange(0, self.height)
		c = random.randrange(0, self.width)
		while last_state[r][c] != 0:
			r = random.randrange(0, self.height)
			c = random.randrange(0, self.width)

		action = r * self.height + self.width
		return action, action_probs