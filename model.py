import tensorflow as tf
import numpy as np
import random
from collections import deque


class DQN:
    REPLAY_MEMORY = 10000
    BATCH_SIZE = 50
    GAMMA = 0.99
    LEARNING_RATE = 0.0001
    MOMENTUM = 0.95

    def __init__(self, session, width, height, n_action):
        self.session = session
        self.n_action = n_action
        self.width = width
        self.height = height
        self.memory = deque()
        self.state = None
        self.init = tf.global_variables_initializer()

        self.input_X = tf.placeholder(tf.float32, [None, width, height, 1])
        self.input_A = tf.placeholder(tf.int64, [None])
        self.input_Y = tf.placeholder(tf.float32, [None])

        self.Q = self._build_network('main')
        self.cost, self.train_op = self._build_op()
        self.target_Q = self._build_network('target')

    def _build_network(self, name):
        initializer = tf.contrib.layers.variance_scaling_initializer()
        with tf.variable_scope(name):
            model = tf.layers.conv2d(self.input_X, 32, [3, 3], padding='same', activation=tf.nn.relu,
                kernel_initializer=initializer)
            model = tf.layers.conv2d(model, 64, [3, 3], padding='same', activation=tf.nn.relu,
                kernel_initializer=initializer)
            model = tf.layers.conv2d(model, 128, [3, 3], padding='same', activation=tf.nn.relu,
                kernel_initializer=initializer)
            model = tf.contrib.layers.flatten(model)
            model = tf.layers.dense(model, 512, activation=tf.nn.relu,
                kernel_initializer=initializer)

            Q = tf.layers.dense(model, self.n_action, activation=None)
            self.Prob = tf.nn.softmax(Q)

        return Q

    def _build_op(self):
        """
        neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.Q, labels=self.input_A)
        loss = tf.reduce_mean(neg_log_prob * self.input_Y)
        train_op = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(loss)
        
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.Q, labels=self.input_A)
        grads_and_vars = optimizer.compute_gradients(cross_entropy)
        gradients = [grad for grad, variable in grads_and_vars]
        gradient_placeholders = []
        grads_and_vars_feed = []
        for grad, variable in grads_and_vars:
            gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
            gradient_placeholders.append(gradient_placeholder)
            grads_and_vars_feed.append((gradient_placeholder, variable))
        train_op = optimizer.apply_gradients(grads_and_vars_feed)
        """

        one_hot = tf.one_hot(self.input_A, self.n_action)
        Q_value = tf.reduce_sum(tf.multiply(self.Q, one_hot), axis=1, keepdims=True)
        error = tf.abs(self.input_Y - Q_value)
        clipped_error = tf.clip_by_value(error, 0.0, 1.0)
        linear_error = 2 * (error - clipped_error)
        loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)
        """
        global_step = tf.Variable(0, trainable=False, name='global_step')
        optimizer = tf.train.MomentumOptimizer(self.LEARNING_RATE, self.MOMENTUM, use_nesterov=True)
        """
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        train_op = optimizer.minimize(loss)

        return loss, train_op

    def update_target_network(self):
        copy_op = []

        main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='main')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')

        for main_var, target_var in zip(main_vars, target_vars):
            copy_op.append(target_var.assign(main_var.value()))

        self.session.run(copy_op)

    def get_action(self, state):
        prev = state
        state = np.reshape(state, (self.width, self.height, 1))
        Q_value = self.session.run(self.Q,
                                   feed_dict={self.input_X: [state]})

        candidates = Q_value[0].argsort()[::-1][:100]
        for a in candidates:
            x = int(a/self.height)
            y = a % self.width
            if prev[x][y] == 0:
                return a

        action = np.argmax(Q_value[0])
        return action

    def init_state(self, state):
        self.state = state

    def remember(self, state, action, reward):
        state = np.reshape(state, (self.width, self.height, 1))
        self.memory.append((state, action, reward))

        if len(self.memory) > self.REPLAY_MEMORY:
            self.memory.popleft()

    def _sample_memory(self):
        sample_memory = random.sample(self.memory, self.BATCH_SIZE)

        state = [memory[0] for memory in sample_memory]
        action = [memory[1] for memory in sample_memory]
        reward = [memory[2] for memory in sample_memory]

        return state, action, reward

    def train(self):
        state, action, reward = self._sample_memory()
        target_Q_value = self.session.run(self.target_Q,
                                          feed_dict={self.input_X: state})

        Y = []
        for i in range(self.BATCH_SIZE):
            Y.append(reward[i])

        self.session.run(self.train_op,
                         feed_dict={
                             self.input_X: state,
                             self.input_A: action,
                             self.input_Y: Y
                         })