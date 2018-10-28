import numpy as np
import time

class Game:
	def __init__(self, state):
		self.state = state

	def play(self, black_player, white_player, show=False):
		self.state.reset()
		players = [white_player, black_player]
		game_states = []
		action_probs = []
		current_players = []
		time_per_turn = []
		game_time = time.time()
		winner = -1
		while True:
			player = self.state.get_current_player()
			current_state = self.state.get_current_state()
			start_time = time.time()
			action, probs = players[player].get_action(
				self.state)
			time_per_turn.append(time.time() - start_time)

			game_states.append(current_state)
			action_probs.append(probs)
			current_players.append(player)

			winner = self.state.do_action(action)
			if winner != -1:
				break
			if show:
				print("")
				print(self.state.get_game_state())
				print("Turn per secs : ", np.mean(time_per_turn))

		values = np.zeros(len(current_players))
		values[np.array(current_players) == winner] = 1.0
		values[np.array(current_players) != winner] = -1.0

		return winner, game_states, action_probs, values