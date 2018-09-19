import numpy as np

class DDoGo(object):
	def __init__(self, height, width, brain):
		self.height = height
		self.width = width
		self.brain = brain
		self.state = None
		self.available = None
		self.count = 5
		self.a_score_board = []
		self.b_score_board = []

	def get_action(self, player, state, available):
		self.a_score_board = []
		self.b_score_board = []
		self.state = np.array(state)
		if player == 2:
			state = self.change_state(self.state)
			player = 1

		self.count_a_score()
		self.count_b_score()
		
		cands = []
		for i in range(self.height):
			for j in range(self.width):
				for k in range(4):
					if self.b_score_board[k][i][j] >= 4:
						cands.append([i, j])
		if len(cands) > 0:
			return self.choice(cands)

		for i in range(self.height):
			for j in range(self.width):
				for k in range(4):
					if self.a_score_board[k][i][j] >= 4:
						cands.append([i, j])
		if len(cands) > 0:
			return self.choice(cands)

		
		for i in range(self.height):
			for j in range(self.width):
				for k in range(4):
					if self.a_score_board[k][i][j] >= 3:
						cands.append([i, j])
		if len(cands) > 0:
			return self.choice(cands)
		

		for i in range(self.height):
			for j in range(self.width):
				for k in range(4):
					if self.a_score_board[k][i][j] >= 2:
						cands.append([i, j])
		if len(cands) > 0:
			return self.choice(cands)

		
		for i in range(self.height):
			for j in range(self.width):
				for k in range(4):
					if self.a_score_board[k][i][j] >= 1:
						cands.append([i, j])
		if len(cands) > 0:
			return self.choice(cands)
		

		for i in range(self.height):
			for j in range(self.width):
				if self.state[i][j] == 0:
					action = i * self.height + j

		return action

	def choice(self, cands):

		max_score = -10000
		best_action = None
		for a in cands:
			x = a[0]
			y = a[1]

			self.state[x][y] = 1
			self.count_a_score()
			self.count_b_score()
			score = np.sum(self.a_score_board) - np.sum(self.b_score_board)
			if score > max_score:
				max_score = score
				best_action = a
			self.state[x][y] = 0

		return a[0] * self.height + a[1]


	"""
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

		return move, action_probs
	"""

	def count_a_score(self):
		temp = np.zeros(self.height * self.width, dtype=int).reshape(
			self.height, self.width)
		for i in range(0, self.height):
			for j in range(0, self.width):
				temp[i][j] += self.count_row_score(self.state, i, j, 1)
		self.a_score_board.append(temp)
		temp = np.zeros(self.height * self.width, dtype=int).reshape(
			self.height, self.width)
		for i in range(0, self.height):
			for j in range(0, self.width):
				temp[i][j] += self.count_col_score(self.state, i, j, 1)
		self.a_score_board.append(temp)
		temp = np.zeros(self.height * self.width, dtype=int).reshape(
			self.height, self.width)
		for i in range(0, self.height):
			for j in range(0, self.width):
				temp[i][j] += self.count_bslash_score(self.state, i, j, 1)
		self.a_score_board.append(temp)
		temp = np.zeros(self.height * self.width, dtype=int).reshape(
			self.height, self.width)
		for i in range(0, self.height):
			for j in range(0, self.width):
				temp[i][j] += self.count_slash_score(self.state, i, j, 1)
		self.a_score_board.append(temp)

	def count_b_score(self):
		temp = np.zeros(self.height * self.width, dtype=int).reshape(
			self.height, self.width)
		for i in range(0, self.height):
			for j in range(0, self.width):
				temp[i][j] += self.count_row_score(self.state, i, j, 2)
		self.b_score_board.append(temp)
		temp = np.zeros(self.height * self.width, dtype=int).reshape(
			self.height, self.width)
		for i in range(0, self.height):
			for j in range(0, self.width):
				temp[i][j] += self.count_col_score(self.state, i, j, 2)
		self.b_score_board.append(temp)
		temp = np.zeros(self.height * self.width, dtype=int).reshape(
			self.height, self.width)
		for i in range(0, self.height):
			for j in range(0, self.width):
				temp[i][j] += self.count_bslash_score(self.state, i, j, 2)
		self.b_score_board.append(temp)
		temp = np.zeros(self.height * self.width, dtype=int).reshape(
			self.height, self.width)
		for i in range(0, self.height):
			for j in range(0, self.width):
				temp[i][j] += self.count_slash_score(self.state, i, j, 2)
		self.b_score_board.append(temp)

	def count_row_score(self, state, x, y, player):
		if state[x][y] != 0:
			return 0

		ret = 0
		lo = x-1
		count = 0
		while lo >= 0 and (state[lo][y] == player or state[lo][y] == 3) \
			and count < self.count:
			lo -= 1
			count += 1
		lo += 1
		ret += x - lo

		hi = x+1
		count = 0
		while hi < self.height and (state[hi][y] == player or state[hi][y] == 3) \
			and count < self.count:
			hi += 1
			count += 1
		hi -= 1
		ret += hi - x
		return ret

	def count_col_score(self, state, x, y, player):
		if state[x][y] != 0:
			return 0

		ret = 0
		lo = y-1
		count = 0
		while lo >= 0 and (state[x][lo] == player or state[x][lo] == 3) \
			and count < self.count:
			lo -= 1
			count += 1
		lo += 1
		ret += y - lo

		hi = y+1
		count = 0
		while hi < self.height and (state[x][hi] == player or state[x][hi] == 3) \
			and count < self.count:
			hi += 1
			count += 1
		hi -= 1
		ret += hi - y
		return ret

	def count_bslash_score(self, state, x, y, player):
		if state[x][y] != 0:
			return 0

		ret = 0
		lo_x = x-1
		lo_y = y-1
		count = 0
		while lo_x >= 0 and lo_y >= 0 and (state[lo_x][lo_y] == player or 
			state[lo_x][lo_y] == 3) and count < self.count:
			lo_x -= 1
			lo_y -= 1
			count += 1
		lo_x += 1
		lo_y += 1
		ret += x - lo_x

		hi_x = x+1
		hi_y = y+1
		count = 0
		while hi_x < self.height and hi_y < self.height and (state[hi_x][hi_y] ==
			player or state[hi_x][hi_y] == 3) and count < self.count:
			hi_x += 1
			hi_y += 1
			count += 1
		hi_x -= 1
		hi_y -= 1
		ret += hi_x - x
		return ret

	def count_slash_score(self, state, x, y, player):
		if state[x][y] != 0:
			return 0

		ret = 0
		lo_x = x+1
		lo_y = y-1
		count = 0
		while lo_x < self.height and lo_y >= 0 and (state[lo_x][lo_y] == player or 
			state[lo_x][lo_y] == 3) and count < self.count:
			lo_x += 1
			lo_y -= 1
			count += 1
		lo_x -= 1
		lo_y += 1
		ret += y - lo_y

		hi_x = x-1
		hi_y = y+1
		count = 0
		while hi_x >= 0 and hi_y < self.height and (state[hi_x][hi_y] == player
			or state[hi_x][hi_y] == 3) and count < self.count:
			hi_x -= 1
			hi_y += 1
			count += 1
		hi_x += 1
		hi_y -= 1
		ret += x - hi_x
		return ret

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
		ret_state = state
		ret_state[ret_state == 2] = 4
		ret_state[ret_state == 1] = 2
		ret_state[ret_state == 4] = 1
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
		if yhi - ylo + 1 == self.count:
			return player
		if yhi - ylo + 1 > self.count:
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
		if xhi - xlo + 1 == self.count:
			return player
		if xhi - xlo + 1 > self.count:
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
		if xhi - xlo + 1 == self.count:
			return player
		if xhi - xlo + 1 > self.count:
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
		if xhi - xlo + 1 == self.count:
			return player
		if xhi - xlo + 1 > self.count:
			if player == 1:
				return 2
			return 1

		return 0