import random

class Sixmok:
	def __init__(self, first):
		self.turn = 0
		self.board = []
		self.randomBlockingCnt = random.randrange(1, 6) * 2
		self.remain = 19 * 19 - self.randomBlockingCnt
		self.blocking = []
		self.first = first
		self.player1 = 0
		self.player2 = 0
		self.gibo = []
		self.win = 0

		if first == 1:
			player1 = 1
			player2 = 2
		elif first == 2:
			player1 = 2
			player2 = 1

		for i in range(0, 19):
			self.board.append([0] * 19)

		for i in range(0, self.randomBlockingCnt):
			x = random.randrange(0, 19)
			y = random.randrange(0, 19)
			if self.board[x][y] == 0:
				self.board[x][y] = 3
				self.blocking.append([x,y])
			else:
				i -= 1

	def checkFinish(self, player, x, y):
		# up-down
		ylo = y
		yhi = y
		while ylo >= 0 and (self.board[x][ylo] == player or
			self.board[x][ylo] == 3):
			ylo -= 1
		ylo += 1
		while yhi < 19 and (self.board[x][yhi] == player or
			self.board[x][yhi] == 3):
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
		while xlo >= 0 and (self.board[xlo][y] == player or
			self.board[xlo][y] == 3):
			xlo -= 1
		xlo += 1
		while xhi < 19 and (self.board[xhi][y] == player or
			self.board[xhi][y] == 3):
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
		while xlo >= 0 and ylo >= 0 and (self.board[xlo][ylo] == player or
			self.board[xlo][ylo] == 3):
			xlo -= 1
			ylo -= 1
		xlo += 1
		ylo += 1
		while xhi < 19 and yhi < 19 and (self.board[xhi][yhi] == player or
			self.board[xhi][yhi] == 3):
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
		while xlo >= 0 and ylo < 19 and (self.board[xlo][ylo] == player or
			self.board[xlo][ylo] == 3):
			xlo -= 1
			ylo += 1
		xlo += 1
		ylo -= 1
		while xhi < 19 and yhi >= 0 and (self.board[xhi][yhi] == player or
			self.board[xhi][yhi] == 3):
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

		return 3

	def checkBoard(self, x, y):
		if x < 0 or x > 18:
			return False
		if y < 0 or y > 18:
			return False
		if self.board[x][y] != 0:
			return False
		return True

	def randomPolicy(self, player):
		x = -1
		y = -1
		while self.board[x][y] != 0:
			x = random.randrange(0, 19)
			y = random.randrange(0, 19)
		##print("turn ", self.turn)
		##print("player " + str(player) + " : ", [x,y])
		return x, y

	def adjacentPolicy(self, player):
		i = player-1
		adj = [[-1,-1], [-1,0], [-1,1], [0,-1], \
				[0,1], [1,-1], [1,0], [1,1]]
		candidate = []
		while i < len(self.gibo):
			for k in adj:
				x = self.gibo[i][0] + k[0]
				y = self.gibo[i][1] + k[1]

				if self.checkBoard(x,y):
					candidate.append([x,y])
			i += 2

		for i in self.blocking:
			for j in adj:
				x = i[0] + j[0]
				y = i[1] + j[1]

				if self.checkBoard(x,y):
					candidate.append([x,y])

		n = random.randrange(0, len(candidate))
		return candidate[n][0], candidate[n][1]

	def randomPlay(self):
		player = self.first
		while True:
			self.turn += 1
			if player == 1:
				x, y = self.adjacentPolicy(player)
			else:
				x, y = self.randomPolicy(player)
			self.remain -= 1
			self.board[x][y] = player
			self.gibo.append([x,y])
			result = self.checkFinish(player, x, y)
			##self.printBoard()
			if self.remain == 0:
				print("Draw")
				break;
			if result == -1:
				break;
			if result == 1:
				print("player 1 win")
				self.win = 1
				break
			if result == 2:
				print("player 2 win")
				self.win = 2
				break
			
			if player == 1:
				player = 2
			else:
				player = 1

	def printBoard(self):
		for row in self.board:
			print(row)

	def retTurn(self):
		return self.turn

	def retWin(self):
		return self.win

r = 0
one = 0
two = 0
for i in range(0, 100):
	s = Sixmok(1)
	s.randomPlay()
	r += s.retTurn()
	if s.retWin() == 1:
		one += 1
	else:
		two += 1
print(r/100.0)
print("player 1 : ", one)
print("player 2 : ", two)