import numpy as np

class Game:
	def __init__(self, height, width, max_state_size=7):
		self.width = width
		self.height = height
		self.max_state_size = max_state_size
		self.black_states = []
		self.white_states = []
		self.counts = height * width
		self.reset()

	def reset(self):
		self.black_states.clear()
		self.white_states.clear()
		self.counts = self.height * self.width
		for i in range(self.max_state_size):
			zero_state_black = np.zeros((self.height, self.width))
			zero_state_white = np.zeros((self.height, self.width))
			black_states.append(zero_state_black)
			white_states.append(zero_state_white)

	def play(self, black_player, white_player):
		self.reset()
		player = 1
		players = [white_player, black_player]
		game_states = []
		action_probs = []
		current_players = []
		winner = -1
		while True:
			current_state = self.get_current_state(player)
			action, probs = players[player].get_action(
				current_state)

			game_states.append(current_state)
			action_probs.append(probs)
			current_players.append(player)

			winner = self.do_action(player, action)
			if winner == -1:
				player ^= 1
			else:
				break

		if winner == 2:
			return [], [], []

		values = np.zeros(len(current_players))
		values[np.array(current_players) == winner] = 1.0
		values[np.array(current_players) != winner] = -1.0

		return game_states, action_probs, values

	def get_current_state(self, player):
		ret = []
		if player == 1:
			ret.append(np.ones((height, width)))
		else:
			ret.append(np.zeros((height, width)))

		ret.extend(white_states[(len(white_states) - self.max_state_size):])
		ret.extend(black_states[(len(black_states) - self.max_state_size):])

		return ret

	def do_action(self, player, action):
		r = int(action / self.height)
		c = action % self.width
		state = None
		if player == 1:
			state = np.array(self.black_states[len(self.black_states)-1])
		else:
			state = np.array(self.white_states[len(self.white_states)-1])
		state[r][c] = 1
		self.counts -= 1
		if player == 1:
			self.black_states.append(state)
		else:
			self.white_states.append(state)

		if self.is_win(state, r, c):
			return player

		if self.counts == 0:
			return 2

		return -1

	def is_win(self, state, r, c):
		lo_r = r
		hi_r = r
		while state[lo_r][c] == 1 and lo_r > -1:
			lo_r -= 1
		lo_r += 1
		while state[hi_r][c] == 1 and hi_r < self.height:
			hi_r += 1
		hi_r -= 1
		if hi_r - lo_r + 1 == 5:
			return True

		lo_c = c
		hi_c = c
		while state[r][lo_c] == 1 and lo_c > -1:
			lo_c -= 1
		lo_c += 1
		while state[r][hi_c] == 1 and hi_c < self.width:
			hi_c += 1
		hi_c -= 1

		if hi_c - lo_c + 1 == 5:
			return True

		lo_r = r
		hi_r = r
		lo_c = c
		hi_c = c
		while state[lo_r][lo_c] == 1 and lo_r > -1 and lo_c > -1:
			lo_r -= 1
			lo_c -= 1
		lo_r += 1
		lo_c += 1
		while state[hi_r][hi_c] == 1 and hi_r < self.height \
				and hi_c < self.width:
			hi_r += 1
			h-_c += 1
		hi_r -= 1
		hi_c -= 1

		if hi_r - lo_r + 1 == 5:
			return True

		lo_r = r
		hi_r = r
		lo_c = c
		hi_c = c
		while state[lo_r][lo_c] == 1 and lo_r > -1 and lo_c < self.width:
			lo_r -= 1
			lo_c += 1
		lo_r += 1
		lo_c -= 1
		while state[hi_r][hi_c] == 1 and hi_r < self.height \
				and hi_c > -1:
			hi_r += 1
			hi_c -= 1
		hi_r -= 1
		hi_c += 1

		if hi_r - lo_r + 1 == 5:
			return True

		return False