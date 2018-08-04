import tensorflow as tf
import numpy as np

class PolicyValueNet:
    def __init__(self, session, width, height, model_file=None):
        self.session = session
        self.width = width
        self.height = height
        self.n_action = width * height

        self.init = tf.global_variables_initializer()
        self.session.run(self.init)
        self.initializer = tf.contrib.layers.variance_scaling_initializer()
        self.saver = tf.train.Saver()
        if model_file is not None:
            self.restore_model(model_file)

        self.input_state = tf.placeholder(tf.float32, [None, width, height, 1])
        self.input_action = tf.placeholder(tf.float32, [None])
        self.input_win = tf.placeholder(tf.float32, [None])

        self.Common_Network = _build_common_network()
        self.Policy_Network = _build_policy_network()
        self.Value_Network = _build_value_network()
        self.train_op = _build_train_op()
        self.learning_rate = tf.placeholder(tf.float32)
    
    def _build_common_network(self):
        model = tf.layers.conv2d(inputs=self.input_state,
                                      filters=32, kernel_size=[3, 3],
                                      padding="same",
                                      activation=tf.nn.relu,
                                      kernel_initializer=self.initializer)
        model = tf.layers.conv2d(inputs=model,
                                      filters=64, kernel_size=[3, 3],
                                      padding="same",
                                      activation=tf.nn.relu,
                                      kernel_initializer=self.initializer)
        model = tf.layers.conv2d(inputs=model,
                                      filters=128, kernel_size=[3, 3],
                                      padding="same",
                                      activation=tf.nn.relu,
                                      kernel_initializer=self.initializer)
        return model

    def _build_policy_network(self):
        model = tf.layers.conv2d(inputs=self.Common_Network,
                                      filters=4, kernel_size=[1, 1],
                                      padding="same",
                                      activation=tf.nn.relu,
                                      kernel_initializer=self.initializer)
        model = tf.contrib.layers.flatten(model)
        model = tf.layers.dense(model, self.n_action,
                                activation=tf.nn.log_softmax)

        return model

    def _build_policy_network(self):
        model = tf.layers.conv2d(inputs=self.Common_Network,
                                      filters=2, kernel_size=[1, 1],
                                      padding="same",
                                      activation=tf.nn.relu,
                                      kernel_initializer=self.initializer)
        model = tf.contrib.layers.flatten(model)
        model = tf.layers.dense(model, 64, activation=tf.nn.relu)
        model = tf.layers.dense(model, 1, activation=tf.nn.relu)

        return model

    def _build_train_op(self):
        action_one_hot = tf.one_hot(self.input_action, self.n_action)
        policy_loss = tf.negative(tf.reduce_mean(
            tf.reduce_sum(tf.multiply(one_hot, self.Policy_Network))))
        value_loss = tf.losses.mean_squared_error(self.input_win,
                                                  self.Value_Network)

        l2_penalty_beta = 1e-4
        vars = tf.trainable_variables()
        l2_penalty = l2_penalty_beta * tf.add_n(
            [tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()])

        loss = value_loss + policy_loss + l2_penalty
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.minimize(loss)

        return train_op

    def policy_value(self, state):
        state = np.reshape(state, (self.width, self.height, 1))
        action_probs, value = self.session.run(
            [self.Policy_Network, self.Value_Network],
            feed_dict = {self.input_state : state})
        action_probs = np.exp(action_probs)

        return action_probs, value

    def train(self, state_batch, action_batch, winner_batch, lr):
        self.session.run(self.train_op,
                         feed_dict={
                             self.input_state: state_batch,
                             self.input_action: action_batch,
                             self.input_win: winner_batch,
                             self.learning_rate: lr
                         })

    def save_model(self, model_path):
        self.saver.save(self.session, model_path)

    def restore_model(self, model_path):
        self.saver.restore(self.session, model_path)