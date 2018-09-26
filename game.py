import numpy as np

class Game:
	def __init__(self, state):
		self.state = state

	def play(self, black_player, white_player):
		self.state.reset()
		player = 1
		players = [white_player, black_player]
		game_states = []
		action_probs = []
		current_players = []
		winner = -1
		while True:
			current_state = self.state.get_current_state(player)
			action, probs = players[player].get_action(
				current_state)

			game_states.append(current_state)
			action_probs.append(probs)
			current_players.append(player)

			winner = self.state.do_action(player, action)
			if winner == -1:
				player ^= 1
			else:
				break

		if winner == 2:
			print("Draw")
			return 2, [], [], []

		if winner == 0:
			print("White Win")
		else:
			print("Black Win")

		values = np.zeros(len(current_players))
		values[np.array(current_players) == winner] = 1.0
		values[np.array(current_players) != winner] = -1.0

		return winner, game_states, action_probs, values