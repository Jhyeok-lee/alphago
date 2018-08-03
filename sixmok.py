import random
import numpy as np

class Sixmok:
	def __init__(self, height, width, two_policy):
		self.width = width
		self.height = height
		self.two_policy = two_policy
		self.reset(1)

	def reset(self, player):
		self.turn = 0
		self.state = []
		self.randomBlockingCnt = random.randrange(0, 6) * 2
		self.remain = self.width * self.height - self.randomBlockingCnt
		self.blocking = []
		self.first = player
		self.gibo = []
		self.win = 0

		for i in range(0, self.height):
			self.state.append([0] * self.width)

		for i in range(0, self.randomBlockingCnt):
			x = random.randrange(0, self.height)
			y = random.randrange(0, self.width)
			if self.state[x][y] == 0:
				self.state[x][y] = 3
				self.blocking.append([x,y])
			else:
				i -= 1

		return self.state

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
		if self.remain < 0:
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

		if self.two_policy == 1:
			action = self.randomPolicy(2)
		elif self.two_policy == 2:
			action = self.adjacentPolicy(2)
		elif self.two_policy == 3:
			action = self.lastAdjPolicy(2)

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

	def printBoard(self):
		for row in self.state:
			print(row)

	def retTurn(self):
		return self.turn

	def retWin(self):
		return self.win

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