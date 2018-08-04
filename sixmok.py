import random
import numpy as np

class Sixmok:
	def __init__(self, height, width, brain):
		self.width = width
		self.height = height
		self.brain = brain
		self.reset()

	def reset(self):
		self.player = random.randrange(1, 3)
		self.state = np.zeros(self.width * self.height, dtype=int).reshape(self.width, 
								self.height)
		self.available = self.state + 1
		self.randomBlockingCnt = random.randrange(0, 6) * 2
		self.remain = self.width * self.height - self.randomBlockingCnt
		self.blocking = []
		self.gibo = []

		for i in range(0, self.randomBlockingCnt):
			x = random.randrange(0, self.height)
			y = random.randrange(0, self.width)
			if self.state[x][y] == 0:
				self.state[x][y] = 3
				self.available[x][y] = 0
				self.blocking.append([x,y])
			else:
				i -= 1

	def runSelfPlay(self):
		turns = 0
		winner = 0
		states, actions, current_players = [], [], []
		while winner == 0:
			turns += 1
			action_probs = self.brain.policy_value(self.state)[0][0]
			action_probs = np.array(action_probs).reshape(self.width, self.height)
			action_probs = action_probs * self.available
			action = np.argmax(action_probs)
			if self.player == 2:
				action = self.lastAdjPolicy(self.player)

			x = int(action / self.width)
			y = action % self.height

			states.append(self.state)
			actions.append(action)
			current_players.append(self.player)

			self.state[x][y] = self.player
			self.available[x][y] = 0
			self.remain -= 1
			winner = self.checkFinish(self.player, x, y)
			if self.player == 1:
				self.player = 2
			else:
				self.player = 1

		winners = np.zeros(len(current_players))
		if winner == 1 or winner == 2:
			winners[np.array(current_players) == winner] = 1.0
			winners[np.array(current_players) != winner] = -1.0
			return winner, turns, states, actions, winners

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

	def checkBoard(self, x, y):
		if x < 0 or x > self.height-1:
			return False
		if y < 0 or y > self.width-1:
			return False
		if self.state[x][y] != 0:
			return False
		return True

	def randomPolicy(self, player):
		if self.remain <= 0:
			return 0

		x = random.randrange(0, self.height)
		y = random.randrange(0, self.width)
		while self.state[x][y] != 0:
			x = random.randrange(0, self.height)
			y = random.randrange(0, self.width)

		return x * self.height + y

	def adjacentPolicy(self, player):
		if self.randomBlockingCnt == 0:
			return self.randomPolicy(player)
		i = player-1
		adj = [[-1,-1], [-1,0], [-1,1], [0,-1], \
				[0,1], [1,-1], [1,0], [1,1]]
		candidate = []
		tempState = self.state
		while i < len(self.gibo):
			for k in adj:
				x = self.gibo[i][0] + k[0]
				y = self.gibo[i][1] + k[1]
				if self.checkBoard(x,y) == False:
					continue

				tempState[x][y] = player
				winner = self.checkFinish(player, x, y)
				if winner == 0 or winner == player:
					candidate.append([x,y])
				tempState[x][y] = 0
			i += 2

		for i in self.blocking:
			for j in adj:
				x = i[0] + j[0]
				y = i[1] + j[1]

				if self.checkBoard(x,y) == False:
					continue
				
				tempState[x][y] = player
				winner = self.checkFinish(player, x, y)
				if winner == 0 or winner == player:
					candidate.append([x,y])
				tempState[x][y] = 0

		if len(candidate) == 0:
			return self.randomPolicy(player)

		n = random.randrange(0, len(candidate))
		return candidate[n][0] * self.height + candidate[n][1]

	def lastAdjPolicy(self, player):
		if len(self.gibo) == 0:
			return self.adjacentPolicy(player)

		tempState = self.state
		candidate = []
		adj = [[-1,-1], [-1,0], [-1,1], [0,-1], \
				[0,1], [1,-1], [1,0], [1,1]]
		for i in adj:
			x = self.gibo[len(self.gibo)-2][0] + i[0]
			y = self.gibo[len(self.gibo)-2][1] + i[1]
			if self.checkBoard(x,y) == False:
				continue

			tempState[x][y] = player
			winner = self.checkFinish(player, x, y)
			if winner == 0 or winner == player:
				candidate.append([x,y])
			tempState[x][y] = 0

		if len(candidate) == 0:
			return self.adjacentPolicy(player)

		n = random.randrange(0, len(candidate))
		return candidate[n][0] * self.height + candidate[n][1]

	def step(self, player, action):
		x = int(action / self.height)
		y = action % self.height

		if self.state[x][y] != 0:
			return self.state, -1, 2, action

		self.remain -= 1
		self.state[x][y] = player
		self.gibo.append([x,y])
		reward = 0
		winner = self.checkFinish(player, x, y)
		if winner == 0:
			reward = 0
		elif winner == 1:
			reward = 1
			return self.state, reward, winner, action
		elif winner == 2:
			reward = -1
			return self.state, reward, winner, action
		elif self.remain == 0:
			winner = 3
			return self.state, reward, winner, action

		if player == 1:
			player = 2
		else:
			player = 1

		action = self.lastAdjPolicy(2)
		#action = self.brain.get_action(self.state)
		x = int(action / self.height)
		y = action % self.height

		if self.state[x][y] != 0:
			return self.state, 1, 1, action
		self.remain -= 1
		self.state[x][y] = player
		self.gibo.append([x,y])
		reward = 0
		winner = self.checkFinish(player, x, y)
		if winner == 0:
			reward = 0
		elif winner == 1:
			reward = 1
			return self.state, reward, winner, action
		elif winner == 2:
			reward = -1
			return self.state, reward, winner, action
		elif self.remain == 0:
			winner = 3
			return self.state, reward, winner, action

		return self.state, reward, winner, action

	def randomPlay(self):
		player = self.first
		while True:
			self.turn += 1
			if player == 1:
				r = random.randrange(0, 10)
				if r < 9:
					a = self.lastAdjPolicy(player)
					x = int(a / self.height)
					y = a % self.height
				else:
					a = self.adjacentPolicy(player)
					x = int(a / self.height)
					y = a % self.height
			else:
				r = random.randrange(0, 10)
				if r < 9:
					a = self.lastAdjPolicy(player)
					x = int(a / self.height)
					y = a % self.height
				else:
					a = self.adjacentPolicy(player)
					x = int(a / self.height)
					y = a % self.height
			self.remain -= 1
			self.state[x][y] = player
			self.gibo.append([x,y])
			result = self.checkFinish(player, x, y)
			if self.remain == 0:
				print("Draw")
				break;
			if result == -1:
				break;
			if result == 1:
				self.win = 1
				print("Player 1 win")
				return 1
			if result == 2:
				self.win = 2
				print("Player 2 win")
				return 2
			
			if player == 1:
				player = 2
			else:
				player = 1

"""
one = 0
two = 0
for i in range(0, 100):
	s = Sixmok(10, 10, 3)
	winner = 0
	while winner == 0:
		a = s.lastAdjPolicy(1)
		b, c, winner = s.step(1, a)
	s.printBoard()
	if winner == 1:
		one += 1
	elif winner == 2:
		two += 1
print("1 : ", one)
print("2 : ", two)
"""