import numpy as np

class DDoGo(object):
	def __init__(self, height, width, brain):
		self.height = height
		self.width = width
		self.brain = brain
		self.state = None
		self.available = None

	def get_action(self, player, state, available):
		self.state = state
		self.available = available
		self.max_depth = 5
		self.max_branch_count = 5
		score = self.cal_score()
		action_probs, value = self.minimax_search(player, state, available,
			100, 0, 0, 0)
		action_probs *= np.reshape(self.available, self.height * self.width)
		return np.argmax(action_probs)

	def minimax_search(self, player, state, available, n_count, x, y, depth):
		action_probs, value = self.brain.policy_value(state)
		action_probs *= np.reshape(available, self.height * self.width)
		possible_actions = action_probs.argsort()[::-1][:n_count]
		value_by_action = np.zeros(self.height * self.width)
		ret_value = 0.0

		available_count = np.count_nonzero(available)
		winner = self.checkFinish(state, x, y)
		if winner == 2:
			winner = -1
		if available_count == 0 or winner != 0:
			return value_by_action, winner
		if depth >= self.max_depth:
			return value_by_action, value

		branch_count = 0
		for action in possible_actions:
			if action_probs[action] == 0:
				continue
			if branch_count == self.max_branch_count:
				break
			branch_count += 1
			next_state = np.array(state)
			next_available = np.array(available)
			x, y = self.action_to_coor(action)
			if state[x][y] != 0:
				continue
			next_state[x][y] = player
			next_available[x][y] = 0
			if player == 1:
				next_player = 2
			else:
				next_player = 1
			p, v = self.minimax_search(next_player, next_state, next_available,
				n_count, x, y, depth+1)
			value_by_action[action] = value + v

		if player == 1:
			ret_value = np.amax(value_by_action)
		else:
			ret_value = np.amin(value_by_action)

		ret_action_probs = self.softmax(value_by_action)
		return ret_action_probs, ret_value

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

	def cal_score(self):
		score = np.zeros(self.height * self.width).reshape(self.height,
			self.width)
		row_score = np.zeros(self.height)
		col_score = np.zeros(self.width)
		diag1_score = np.zeros(self.height + self.width - 1)
		diag2_score = np.zeros(self.height + self.width - 1)

		for i in range(0, self.height):
			row = 0
			col = 0
			for j in range(0, self.width):
				if self.state[i][j] != 0:
					row += 1
				if self.state[j][i] != 0:
					col += 1
			row_score[i] = row
			col_score[i] = col

		for i in range(0, self.height + self.width - 1):
			a = i
			b = 0
			while a >= self.height:
				a -= 1
				b += 1
			diag1 = 0
			while b < self.width:
				if self.state[a][b] != 0:
					diag1 += 1
				a -= 1
				b += 1
			diag1_score[i] = diag1

		for i in range(-self.width + 1, self.width):
			a = i
			b = 0
			while a < 0:
				a += 1
				b += 1
			diag2 = 0
			while a < self.height and b < self.width:
				if self.state[a][b] != 0:
					diag2 += 1
				a += 1
				b += 1
			diag2_score[i + self.width-1] = diag2

		for i in range(0, self.height):
			for j in range(0, self.width):
				score[i][j] += row_score[i] + col_score[j]
				score[i][j] += diag1_score[i+j]
				score[i][j] += diag2_score[i-j+self.width-1]

		return score

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