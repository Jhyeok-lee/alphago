import tensorflow as tf
import numpy as np

class PolicyValueNet:
    def __init__(self, width, height, max_state_size, learning_rate,
                 model_file=None):
        self.session = tf.Session()
        self.height = height
        self.width = width
        self.n_action = width * height
        self.learning_rate = learning_rate
        self.input_size = max_state_size*2 + 1
        if model_file is not None:
            self.restore_model(model_file)

        self.initializer = tf.contrib.layers.variance_scaling_initializer()
        self.input_state = tf.placeholder(tf.float32, [None, 
                              self.input_size, width, height])
        self.input_action = tf.placeholder(tf.float32, [None, width * height])
        self.input_value = tf.placeholder(tf.float32, [None, 1])

        self.Common_Network = self._build_common_network()
        self.Policy_Network = self._build_policy_network()
        self.Value_Network = self._build_value_network()
        self.train_op = self._build_train_op()
        self.entropy = self._build_policy_entropy()
        self.init = tf.global_variables_initializer()
        self.session.run(self.init)
        self.saver = tf.train.Saver()

        self.loss_data = tf.placeholder(tf.float32)
        self.entropy_data = tf.placeholder(tf.float32)
        tf.summary.scalar('loss', self.loss_data)
        tf.summary.scalar('entropy', self.entropy_data)
        self.writer = tf.summary.FileWriter('graph', self.session.graph)
        self.summary_merged = tf.summary.merge_all()
    
    def _build_common_network(self):
        model = tf.layers.conv2d(inputs=self.input_state,
                                      filters=64, kernel_size=[3, 3],
                                      padding="same",
                                      activation=tf.nn.relu,
                                      kernel_initializer=self.initializer)
        model = tf.layers.conv2d(inputs=model,
                                      filters=64, kernel_size=[3, 3],
                                      padding="same",
                                      activation=tf.nn.relu,
                                      kernel_initializer=self.initializer)
        model = tf.layers.conv2d(inputs=model,
                                      filters=64, kernel_size=[3, 3],
                                      padding="same",
                                      activation=tf.nn.relu,
                                      kernel_initializer=self.initializer)
        model = tf.layers.conv2d(inputs=model,
                                      filters=64, kernel_size=[3, 3],
                                      padding="same",
                                      activation=tf.nn.relu,
                                      kernel_initializer=self.initializer)
        return model

    def _build_policy_network(self):
        model = tf.layers.conv2d(inputs=self.Common_Network,
                                      filters=2, kernel_size=[1, 1],
                                      padding="same",
                                      activation=tf.nn.relu,
                                      kernel_initializer=self.initializer)
        model = tf.contrib.layers.flatten(model)
        model = tf.layers.dense(inputs=model, units=self.n_action,
                                activation=tf.nn.log_softmax)

        return model

    def _build_value_network(self):
        model = tf.layers.conv2d(inputs=self.Common_Network,
                                      filters=1, kernel_size=[1, 1],
                                      padding="same",
                                      activation=tf.nn.relu,
                                      kernel_initializer=self.initializer)
        model = tf.contrib.layers.flatten(model)
        model = tf.layers.dense(model, 64, activation=tf.nn.relu)
        model = tf.layers.dense(model, 1, activation=tf.nn.tanh)

        return model

    def _build_train_op(self):
        policy_loss = tf.negative(tf.reduce_mean(
            tf.reduce_sum(tf.multiply(self.input_action, self.Policy_Network),1)))
        
        value_loss = tf.losses.mean_squared_error(self.input_value,
                                                  self.Value_Network)

        l2_penalty_beta = 1e-4
        vars = tf.trainable_variables()
        l2_penalty = l2_penalty_beta * tf.add_n(
            [tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()])
        
        self.loss = value_loss + policy_loss + l2_penalty
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.minimize(self.loss)

        return train_op

    def _build_policy_entropy(self):
      return tf.negative(tf.reduce_mean(
        tf.reduce_sum(tf.exp(self.Policy_Network) * self.Policy_Network, 1)))

    def policy_value(self, state):
        action_probs, value = self.session.run(
            [self.Policy_Network, self.Value_Network], 
            feed_dict = {self.input_state : [state]})
        action_probs = np.exp(action_probs)
        return action_probs[0], value

    def train(self, state_batch, action_batch, value_batch, step):
      value_batch = np.array(value_batch).reshape(-1, 1)
      loss, entropy, _ = \
        self.session.run([self.loss, self.entropy, self.train_op],
                         feed_dict={
                             self.input_state: state_batch,
                             self.input_action: action_batch,
                             self.input_value: value_batch})
      self.summary = self.session.run(self.summary_merged,
                      feed_dict={self.loss_data : loss,
                                 self.entropy_data : entropy})
      self.writer.add_summary(self.summary, step)
      return loss, entropy

    def save_model(self, model_path, global_step):
      self.saver.save(self.session, model_path, global_step=global_step)

    def restore_model(self, model_path):
      self.saver.restore(self.session, model_path)