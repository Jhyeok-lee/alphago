import numpy as np

class State(object):

	def __init__(self, height, width, max_state_size):
		self.width = width
		self.height = height
		self.max_state_size = max_state_size
		self.black_states = []
		self.white_states = []
		self.counts = height * width
		self.reset()

	def reset(self):
		self.player = 1
		self.black_states.clear()
		self.white_states.clear()
		self.counts = self.height * self.width
		self.available_actions = list(range(self.height * self.height))
		for i in range(self.max_state_size):
			zero_state_black = np.zeros((self.height, self.width))
			zero_state_white = np.zeros((self.height, self.width))
			self.black_states.append(zero_state_black)
			self.white_states.append(zero_state_white)

	def get_game_state(self):
		p = np.zeros((self.height, self.width))
		p[self.black_states[len(self.black_states)-1] == 1] = 1
		p[self.white_states[len(self.white_states)-1] == 1] = 2
		return p

	def get_available_actions(self):
		return self.available_actions

	def get_current_player(self):
		return self.player

	def get_current_state(self):
		ret = []
		if self.player == 1:
			ret.append(np.ones((self.height, self.width)))
		else:
			ret.append(np.zeros((self.height, self.width)))

		ret.extend(self.white_states[(len(self.white_states) -
			self.max_state_size):])
		ret.extend(self.black_states[(len(self.black_states) -
			self.max_state_size):])

		return ret

	def do_action(self, action):
		r = int(action / self.height)
		c = action % self.width
		state = None
		if self.player == 1:
			state = np.array(self.black_states[len(self.black_states)-1])
		else:
			state = np.array(self.white_states[len(self.white_states)-1])
		state[r][c] = 1
		self.available_actions.remove(r * self.height + c)
		self.counts -= 1
		if self.player == 1:
			self.black_states.append(state)
			self.black_states.pop(0)
		else:
			self.white_states.append(state)
			self.white_states.pop(0)

		win = self.is_win(state, r, c)
		if win == 1:
			return self.player
		elif win == 2:
			if self.player == 0:
				return 1
			else:
				return 0

		if self.counts == 0:
			return 2

		self.player ^= 1

		return -1

	def is_win(self, state, r, c):
		lo_r = r
		hi_r = r
		while lo_r > -1 and state[lo_r][c] == 1:
			lo_r -= 1
		lo_r += 1
		while hi_r < self.height and state[hi_r][c] == 1:
			hi_r += 1
		hi_r -= 1
		counts = hi_r - lo_r + 1
		if counts == 5:
			return 1
		elif counts > 5:
			return 2


		lo_c = c
		hi_c = c
		while lo_c > -1 and state[r][lo_c] == 1:
			lo_c -= 1
		lo_c += 1
		while hi_c < self.width and state[r][hi_c] == 1:
			hi_c += 1
		hi_c -= 1
		counts = hi_c - lo_c + 1
		if counts == 5:
			return 1
		elif counts > 5:
			return 2

		lo_r = r
		hi_r = r
		lo_c = c
		hi_c = c
		while lo_r > -1 and lo_c > -1 and state[lo_r][lo_c] == 1:
			lo_r -= 1
			lo_c -= 1
		lo_r += 1
		lo_c += 1
		while hi_r < self.height and hi_c < self.width and \
				state[hi_r][hi_c] == 1:
			hi_r += 1
			hi_c += 1
		hi_r -= 1
		hi_c -= 1
		counts = hi_r - lo_r + 1
		if counts == 5:
			return 1
		elif counts > 5:
			return 2

		lo_r = r
		hi_r = r
		lo_c = c
		hi_c = c
		while lo_r > -1 and lo_c < self.width and state[lo_r][lo_c] == 1:
			lo_r -= 1
			lo_c += 1
		lo_r += 1
		lo_c -= 1
		while hi_r < self.height and hi_c > -1 and \
				state[hi_r][hi_c] == 1:
			hi_r += 1
			hi_c -= 1
		hi_r -= 1
		hi_c += 1
		counts = hi_r - lo_r + 1
		if counts == 5:
			return 1
		elif counts > 5:
			return 2

		return 0