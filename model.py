import tensorflow as tf
import numpy as np

class PolicyValueNet:
    def __init__(self, sess, width, height, model_file=None):
        self.session = sess
        self.width = width
        self.height = height
        self.n_action = width * height
        self.LR = 0.001

        #self.saver = tf.train.Saver()
        #if model_file is not None:
        #    self.restore_model(model_file)

        self.initializer = tf.contrib.layers.variance_scaling_initializer()
        self.input_state = tf.placeholder(tf.float32, [None, width, height, 1])
        self.input_action = tf.placeholder(tf.int32, [None])
        self.input_win = tf.placeholder(tf.float32, [None, 1])

        self.Common_Network = self._build_common_network()
        self.Policy_Network = self._build_policy_network()
        self.Value_Network = self._build_value_network()
        self.train_op = self._build_train_op()
        self.init = tf.global_variables_initializer()
        self.session.run(self.init)
    
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
        model = tf.reshape(model, [-1, 4 * self.width * self.height])
        model = tf.layers.dense(inputs=model, units=self.n_action,
                                activation=tf.nn.log_softmax)

        return model

    def _build_value_network(self):
        model = tf.layers.conv2d(inputs=self.Common_Network,
                                      filters=2, kernel_size=[1, 1],
                                      padding="same",
                                      activation=tf.nn.relu,
                                      kernel_initializer=self.initializer)
        model = tf.reshape(model, [-1, 2 * self.width * self.height])
        model = tf.layers.dense(model, 64, activation=tf.nn.relu)
        model = tf.layers.dense(model, 1, activation=tf.nn.relu)

        return model

    def _build_train_op(self):
        action_one_hot = tf.one_hot(self.input_action, self.n_action)
        policy_loss = tf.negative(tf.reduce_mean(
            tf.reduce_sum(tf.multiply(action_one_hot, self.Policy_Network))))
        
        value_loss = tf.losses.mean_squared_error(self.input_win,
                                                  self.Value_Network)

        l2_penalty_beta = 1e-4
        vars = tf.trainable_variables()
        l2_penalty = l2_penalty_beta * tf.add_n(
            [tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()])
        
        loss = value_loss + policy_loss + l2_penalty
        optimizer = tf.train.AdamOptimizer(learning_rate=self.LR)
        train_op = optimizer.minimize(loss)

        return train_op

    def policy_value(self, state):
        state = np.reshape(state,(self.width ,self.height,1))
        action_probs, value = self.session.run(
            [self.Policy_Network, self.Value_Network], 
            feed_dict = {self.input_state : [state]})
        action_probs = np.exp(action_probs)
        return action_probs.reshape(self.width,self.height)

    def train(self, state_batch, action_batch, winner_batch, lr):
      state_batch = np.array(state_batch).reshape(-1, self.width,
                    self.height, 1)
      winner_batch = np.array(winner_batch).reshape(-1, 1)
      self.session.run(self.train_op,
                         feed_dict={
                             self.input_state: state_batch,
                             self.input_action: action_batch,
                             self.input_win: winner_batch})