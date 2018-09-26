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
		max_q_u = 0.0
		action = -1
		for a in self.children:
			q_u = self.children[a].Q + self.children[a].U
			if q_u > max_q_u:
				max_q_u = q_u
				action = a

		return action, self.children[action]

	def expand(self, actions, action_probs):
		for action, prob in actions, action_probs:
			if action not in self.children:
				self.children[action] = Node(self, prob)

	def update(self, value):
		self.N += 1
		self.W += value
		self.Q = self.W / N
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
			if node.children == {}:
				break
			action, node = node.select()
			winner = state.do_action(action)

		actions, action_probs, value = self.query_policy_value(state)
		if winner == -1:
			node.expand(actions, action_probs)
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
		action_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

		return actions, action_probs

	def query_policy_value(self, state):
		available_actions = state.get_available_actions()
		current_state = state.get_current_state()
		action_probs, value = self.policy_value(current_state)
		return available_actions, action_probs[available_actions], value

	def softmax(x):
		ret = np.exp(x - np.max(x))
		ret /= np.sum(ret)
		return ret
