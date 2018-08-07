import numpy as np

class DDoGo(object):
	def __init__(self, height, width, brain):
		self.height = height
		self.width = width
		self.brain = brain
		self.state = None
		self.available = None

	def get_action(self, player, state, available):
		self.state = np.array(state)
		self.available = np.array(available)
		self.max_depth = 3
		self.max_branch_count = 5
		if player == 2:
			self.state = self.change_state(self.state)
			player = 1

		action_value, ret_value = self.minimax_search(self.state, \
			self.available, 100, 0, 0, 0)

		actions = [d[0] for d in action_value]
		values = [d[1] for d in action_value]

		values = self.softmax(values)
		action_probs = np.zeros(self.height * self.width)
		for i in range(len(actions)):
			action_probs[actions[i]] = values[i]
		move = np.argmax(action_probs)
		print(self.state)
		print(action_probs)

		return move, action_probs

	def minimax_search(self, state, available, n_count, x, y, depth):
		action_probs, value = self.brain.policy_value(state)
		action_probs *= np.reshape(available, self.height * self.width)
		possible_actions = action_probs.argsort()[::-1][:n_count]
		available_count = np.count_nonzero(available)
		winner = self.checkFinish(state, x, y)
		action_value = []
		if winner == 2:
			winner = -1
		if available_count == 0 or winner != 0:
			return action_value, winner
		if depth > self.max_depth:
			return action_value, value

		values = []
		branch_count = 0
		for action in possible_actions:
			if branch_count == self.max_branch_count:
				break
			branch_count += 1
			next_state = np.array(state)
			next_available = np.array(available)
			x, y = self.action_to_coor(action)

			next_state[x][y] = 1
			next_available[x][y] = 0
			next_state = self.change_state(next_state)
			p, v = self.minimax_search(next_state, next_available,
				n_count, x, y, depth+1)
			values.append(value - v)
			action_value.append([action, value - v])

		ret_value = max(values)
		return action_value, ret_value * (0.99 ** (self.max_depth - depth))

	def change_state(self, state):
		ret_state = np.array(state)
		for i in range(self.height):
			for j in range(self.width):
				if ret_state[i][j] == 1:
					ret_state[i][j] = 2
				elif ret_state[i][j] == 2:
					ret_state[i][j] = 1
		return ret_state

	def softmax(self, x):
		probs = np.exp(x - np.max(x))
		probs /= np.sum(probs)
		return probs

	def action_to_coor(self, action):
		x = int(action / self.height)
		y = action % self.height
		return x, y

	def coor_to_action(self, x, y):
		return x * self.height + y

	def checkFinish(self, state, x, y):
		tstate = list(state)
		player = tstate[x][y]
		if player == 0:
			return 0
		# up-down
		ylo = y
		yhi = y
		while ylo >= 0 and (tstate[x][ylo] == player or
			tstate[x][ylo] == 3):
			ylo -= 1
		ylo += 1
		while yhi < self.width and (tstate[x][yhi] == player or
			tstate[x][yhi] == 3):
			yhi += 1
		yhi -= 1
		if yhi - ylo + 1 == 6:
			return player
		if yhi - ylo + 1 > 6:
			if player == 1:
				return 2
			return 1

		# left-right
		xlo = x
		xhi = x
		while xlo >= 0 and (tstate[xlo][y] == player or
			tstate[xlo][y] == 3):
			xlo -= 1
		xlo += 1
		while xhi < self.height and (tstate[xhi][y] == player or
			tstate[xhi][y] == 3):
			xhi += 1
		xhi -= 1
		if xhi - xlo + 1 == 6:
			return player
		if xhi - xlo + 1 > 6:
			if player == 1:
				return 2
			return 1

		# diagnal
		xlo = x
		ylo = y
		xhi = x
		yhi = y
		while xlo >= 0 and ylo >= 0 and (tstate[xlo][ylo] == player or
			tstate[xlo][ylo] == 3):
			xlo -= 1
			ylo -= 1
		xlo += 1
		ylo += 1
		while xhi < self.height and yhi < self.width and \
			(tstate[xhi][yhi] == player or tstate[xhi][yhi] == 3):
			xhi += 1
			yhi += 1
		xhi -= 1
		yhi -= 1
		if xhi - xlo + 1 == 6:
			return player
		if xhi - xlo + 1 > 6:
			if player == 1:
				return 2
			return 1

		xlo = x
		ylo = y
		xhi = x
		yhi = y
		while xlo >= 0 and ylo < self.width and (tstate[xlo][ylo] == player or
			tstate[xlo][ylo] == 3):
			xlo -= 1
			ylo += 1
		xlo += 1
		ylo -= 1
		while xhi < self.height and yhi >= 0 and (tstate[xhi][yhi] == player or
			tstate[xhi][yhi] == 3):
			xhi += 1
			yhi -= 1
		xhi -= 1
		yhi += 1
		if xhi - xlo + 1 == 6:
			return player
		if xhi - xlo + 1 > 6:
			if player == 1:
				return 2
			return 1

		return 0