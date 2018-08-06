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
		score = self.cal_score()
		action_probs = self.brain.policy_value(self.state)
		action_probs *= self.available
		action_probs *= score
		action_probs = action_probs.reshape(self.height * self.width)
		possible_action = self.check_better_action(100, action_probs)
		return possible_action

	def check_better_action(self, n_count, action_probs):
		possible_actions = action_probs.argsort()[::-1][:n_count]
		best = 0
		for a in possible_actions:
			x = int(a / self.height)
			y = a % self.height
			self.state[x][y] = 1
			if self.checkFinish(1, x, y) != 2:
				best = a
				break
			self.state[x][y] = 0

		return best


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

	def checkFinish(self, player, x, y):
		# up-down
		ylo = y
		yhi = y
		while ylo >= 0 and (self.state[x][ylo] == player or
			self.state[x][ylo] == 3):
			ylo -= 1
		ylo += 1
		while yhi < self.width and (self.state[x][yhi] == player or
			self.state[x][yhi] == 3):
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
		while xlo >= 0 and (self.state[xlo][y] == player or
			self.state[xlo][y] == 3):
			xlo -= 1
		xlo += 1
		while xhi < self.height and (self.state[xhi][y] == player or
			self.state[xhi][y] == 3):
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
		while xlo >= 0 and ylo >= 0 and (self.state[xlo][ylo] == player or
			self.state[xlo][ylo] == 3):
			xlo -= 1
			ylo -= 1
		xlo += 1
		ylo += 1
		while xhi < self.height and yhi < self.width and (self.state[xhi][yhi] == player or
			self.state[xhi][yhi] == 3):
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
		while xlo >= 0 and ylo < self.width and (self.state[xlo][ylo] == player or
			self.state[xlo][ylo] == 3):
			xlo -= 1
			ylo += 1
		xlo += 1
		ylo -= 1
		while xhi < self.height and yhi >= 0 and (self.state[xhi][yhi] == player or
			self.state[xhi][yhi] == 3):
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