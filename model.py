import tensorflow as tf
import numpy as np

class PolicyValueNet:
    def __init__(self, width, height, max_state_size,
                 model_path=None, train_mode=True):
        self.session = tf.Session()
        self.height = height
        self.width = width
        self.n_action = width * height
        self.input_size = max_state_size*2 + 1
        self.train_mode = train_mode
        self.num_of_res_layer = 3

        self.initializer = tf.contrib.layers.variance_scaling_initializer()
        self.input_state = tf.placeholder(tf.float32, [None, 
                              self.input_size, width, height])
        self.input_action = tf.placeholder(tf.float32, [None, width * height])
        self.input_value = tf.placeholder(tf.float32, [None, 1])
        self.learning_rate = tf.placeholder(tf.float32)

        self.Input_Network = self._build_input_network()
        self.Common_Network = self._build_common_network()
        self.Policy_Network = self._build_policy_network()
        self.Value_Network = self._build_value_network()
        self.policy_entropy = self._build_policy_entropy()
        self.value_mse = self._build_value_mse()
        self.train_op = self._build_train_op()
        self.init = tf.global_variables_initializer()

        self.loss_data = tf.placeholder(tf.float32)
        self.value_data = tf.placeholder(tf.float32)
        self.entropy_data = tf.placeholder(tf.float32)
        if train_mode:
          tf.summary.scalar('loss', self.loss_data)
          tf.summary.scalar('value mse', self.value_data)
          tf.summary.scalar('policy cross entropy', self.entropy_data)
          self.summary_merged = tf.summary.merge_all()
          self.writer = tf.summary.FileWriter('graph', self.session.graph)
        self.session.run(self.init)
        self.saver = tf.train.Saver()
        if model_path is not None:
          self.restore_model(model_path)

    def _conv2(self, inputs, filters=32, kernel_size=3,
                             padding="same", data_format="channels_last",
                             use_bias=False):
      return tf.layers.conv2d(inputs=inputs,
                                 filters=filters,
                                 kernel_size=kernel_size,
                                 padding=padding,
                                 data_format=data_format,
                                 use_bias=use_bias)

    def _batch_norm(self, inputs, axis=-1, momentum=0.95, epsilon=1e-5,
                          center=True, scale=True, fused=True,
                          training=self.train_mode):
      return tf.layers.batch_normalization(inputs=inputs,
                                            axis=axis,
                                            momentum=momentum,
                                            epsilon=epsilon,
                                            center=center,
                                            scale=scale,
                                            fused=fused,
                                            training=training)

    def _build_input_network(self):
      model = self.input_state
      model = self._conv2(model)
      model = self._batch_norm(model)
      model = tf.nn.relu(model)
      return model

    def _build_common_network(self):
      model = self.Input_Network
      for i in range(self.num_of_res_layer):
        model = self._conv2(model)
        model = self._batch_norm(model)
        model = tf.nn.relu(model)
        model = self._conv2(model)
        model = self._batch_norm(model)
        model = tf.nn.relu(model + self.Input_Network)
      return model

    def _build_policy_network(self):
      model = self._conv2(self.Common_Network, filters=2, kernel_size=1)
      model = self._batch_norm(model, center=False, scale=False)
      model = tf.nn.relu(model)
      model = tf.reshape(model, [-1, 2 * self.width * self.height])
      self.logits = tf.layers.dense(model, self.width * self.height)
      model = tf.nn.softmax(self.logits)
      return model

    def _build_value_network(self):
        model = self._conv2(self.Common_Network, filters=1, kernel_size=1)
        model = self._batch_norm(model, center=False, scale=False)
        model = tf.nn.relu(model)
        model = tf.reshape(model, [-1, self.width * self.height])
        model = tf.layers.dense(model, 64)
        model = tf.nn.relu(model)
        model = tf.layers.dense(model, 1)
        model = tf.reshape(model, [-1])
        model = tf.nn.tanh(model)
        return model


    def _build_policy_entropy(self):
      
      ce = tf.nn.softmax_cross_entropy_with_logits_v2(
              logits=self.logits, labels=tf.stop_gradient(self.input_action))
      return tf.reduce_mean(ce)

    def _build_value_mse(self):
      return tf.losses.mean_squared_error(self.input_value,
                                          self.Value_Network)

    def _build_train_op(self):
        l2_penalty_beta = 1e-4
        vars = tf.trainable_variables()
        l2_penalty = l2_penalty_beta * tf.add_n(
            [tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()])
        
        self.loss = self.value_mse + self.policy_entropy + l2_penalty
        optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
        train_op = optimizer.minimize(self.loss)

        return train_op

    def policy_value(self, state):
        action_probs, value = self.session.run(
            [self.Policy_Network, self.Value_Network], 
            feed_dict = {self.input_state : [state]})
        return action_probs[0], value

    def train(self, state_batch, action_batch, value_batch, step, 
                learning_rate):
      value_batch = np.array(value_batch).reshape(-1, 1)
      loss, value_mse, policy_entropy, _ = \
        self.session.run([self.loss, self.value_mse, self.policy_entropy, self.train_op],
                         feed_dict={
                             self.input_state: state_batch,
                             self.input_action: action_batch,
                             self.input_value: value_batch,
                             self.learning_rate: learning_rate})
      self.summary = self.session.run(self.summary_merged,
                      feed_dict={self.loss_data : loss,
                                 self.value_data : value_mse,
                                 self.entropy_data : policy_entropy})
      self.write_graph(step)
      return loss, value_mse, policy_entropy

    def write_graph(self, step):
      self.writer.add_summary(self.summary, step)

    def save_model(self, model_path, step):
      self.saver.save(self.session, model_path, global_step=step)

    def restore_model(self, model_path):
      self.saver.restore(self.session, model_path)