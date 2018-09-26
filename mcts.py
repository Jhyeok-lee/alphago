import numpy as np
import copy

class Node(object):

	def __init__(self, parent, prior_prob):
		self.parent = parent
		self.children = {}
		self.N = 0
		self.W = 0.0
		self.Q = 0.0
		self.U = 0.0
		self.P = prior_prob

	def select(self):
		return max(self.children.items(),
			key=lambda action_node: action_node[1].get_value())

	def expand(self, actions_to_probs):
		for action, prob in actions_to_probs:
			if action not in self.children:
				self.children[action] = Node(self, prob)

	def get_value(self):
		self.U = 5 * self.P * np.sqrt(self.parent.N) / (1 + self.N)
		return self.Q + self.U

	def update(self, value):
		self.N += 1
		self.W += value
		self.Q = self.W / self.N
		if self.parent is not None:
			self.U = self.P * np.sqrt(self.parent.N) / (1 + self.N)

	def backpropagation(self, value):
		if self.parent != None:
			self.parent.backpropagation(-value)
		self.update(value)


class MCTS(object):

	def __init__(self, policy_value, simulation_count=1600):
		self.root = Node(None, 1.0)
		self.policy_value = policy_value
		self.simulation_count = simulation_count

	def get_action(self, state):
		self.root = Node(None, 1.0)
		available_actions = state.get_available_actions()
		action_probs = np.zeros(state.height * state.width)
		actions, probs = self.search(state)
		action_probs[list(actions)] = probs
		action = np.random.choice(actions, p=probs)
		return action, action_probs

	def simulation(self, state):
		node = self.root
		winner = -1
		while True:
			if len(node.children) == 0:
				break
			action, node = node.select()
			winner = state.do_action(action)

		action_probs, value = self.query_policy_value(state)
		if winner == -1:
			node.expand(action_probs)
		elif winner == 2:
			value = 0
		elif winner == state.get_current_player():
			value = 1
		else:
			value = -1

		node.backpropagation(-value)

	def search(self, origin_state):
		
		for i in range(self.simulation_count):
			state = copy.deepcopy(origin_state)
			self.simulation(state)

		temp = 1e-3
		actions_to_visits = [(action, node.N)
					for action, node in self.root.children.items()]
		actions, visits = zip(*actions_to_visits)
		action_probs = self.softmax(1.0/temp *
			np.log(np.array(visits) + 1e-10))

		return actions, action_probs

	def query_policy_value(self, state):
		available_actions = state.get_available_actions()
		current_state = state.get_current_state()
		action_probs, value = self.policy_value(current_state)
		actions_to_probs = \
			zip(available_actions, action_probs[available_actions])
		return actions_to_probs, value

	def softmax(self, x):
		ret = np.exp(x - np.max(x))
		ret /= np.sum(ret)
		return ret