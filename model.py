import tensorflow as tf
import numpy as np

class PolicyValueNet:
    def __init__(self, width, height, max_state_size, value_weight=1.0,
                 policy_weight=1.0, model_path=None, train_mode=True):
        self.session = tf.Session()
        self.height = height
        self.width = width
        self.n_action = width * height
        self.input_size = max_state_size*2 + 1
        self.train_mode = train_mode
        self.num_of_res_layer = 1
        self.value_head_weight = 1.0
        self.policy_entropy_weight = 1.0

        self.initializer = tf.contrib.layers.variance_scaling_initializer()
        self.input_state = tf.placeholder(tf.float32, [None, self.input_size,
                                            width, height])
        self.transpose_input_state = tf.transpose(self.input_state, [0, 2, 3, 1])
        self.input_action = tf.placeholder(tf.float32, [None, width * height])
        self.input_value = tf.placeholder(tf.float32, [None])
        self.learning_rate = tf.placeholder(tf.float32)

        self.Input_Network = self._build_input_network('current')
        self.Common_Network = self._build_common_network('current')
        self.Policy_Network = self._build_policy_network('current')
        self.Value_Network = self._build_value_network('current')
        self.Best_Input_Network = self._build_input_network('best')
        self.Best_Common_Network = self._build_common_network('best')
        self.Best_Policy_Network = self._build_policy_network('best')
        self.Best_Value_Network = self._build_value_network('best')
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

    def _conv2(self, inputs, filters=64, kernel_size=3,
                             padding="same", data_format="channels_last",
                             use_bias=False, name=None):
      return tf.layers.conv2d(inputs=inputs,
                                 filters=filters,
                                 kernel_size=kernel_size,
                                 padding=padding,
                                 data_format=data_format,
                                 use_bias=use_bias,
                                 kernel_initializer=self.initializer,
                                 name=name)

    def _batch_norm(self, inputs, axis=-1, momentum=0.95, epsilon=1e-5,
                          center=True, scale=True, fused=True):
      return tf.layers.batch_normalization(inputs=inputs,
                                            axis=axis,
                                            momentum=momentum,
                                            epsilon=epsilon,
                                            center=center,
                                            scale=scale,
                                            fused=fused,
                                            training=self.train_mode)

    def _build_input_network(self, name):
      with tf.variable_scope(name):
        model = self.transpose_input_state
        model = self._conv2(model, name="input_conv")
        #model = self._batch_norm(model)
        model = tf.nn.relu(model)
      return model

    def _build_common_network(self, name):
      with tf.variable_scope(name):
        model = self.Input_Network
        for i in range(self.num_of_res_layer):
          first_input = model
          model = self._conv2(model, name="common_conv" + str(i*2+1))
          #model = self._batch_norm(model)
          model = tf.nn.relu(model)
          model = self._conv2(model, name="common_conv" + str(i*2+2))
          #model = self._batch_norm(model)
          model = tf.nn.relu(model)
      return model

    def _build_policy_network(self, name):
      with tf.variable_scope(name):
        model = self._conv2(self.Common_Network, filters=2, kernel_size=1,
          name="policy_conv")
        #model = self._batch_norm(model, center=False, scale=False)
        model = tf.nn.relu(model)
        model = tf.reshape(model, [-1, 2 * self.width * self.height])
        #self.logits = tf.layers.dense(model, self.width * self.height)
        #model = tf.nn.softmax(self.logits)
        model = tf.layers.dense(inputs=model, units=self.width*self.height,
          activation=tf.nn.log_softmax, name="policy_dense")
      return model

    def _build_value_network(self, name):
      with tf.variable_scope(name):
        model = self._conv2(self.Common_Network, filters=1, kernel_size=1,
          name="value_conv")
        #model = self._batch_norm(model, center=False, scale=False)
        model = tf.nn.relu(model)
        model = tf.reshape(model, [-1, 1 * self.width * self.height])
        model = tf.layers.dense(model, 64, name="value_dense1")
        model = tf.nn.relu(model)
        model = tf.layers.dense(model, 1, name="value_dense2")
        model = tf.reshape(model, [-1])
        model = tf.nn.tanh(model)
        return model

    def update_best_model(self):
      copy_op = []
      current_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
        scope='current')
      best_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
        scope='best')

      for current_var, best_var in zip(current_vars, best_vars):
        copy_op.append(best_var.assign(current_var.value()))

      self.session.run(copy_op)

    def _build_policy_entropy(self):
      """
      ce = tf.nn.softmax_cross_entropy_with_logits_v2(
              logits=self.logits, labels=tf.stop_gradient(self.input_action)) * \
        self.policy_entropy_weight
      """
      ce = tf.negative(tf.reduce_mean(tf.reduce_sum(
        tf.multiply(self.input_action, self.Policy_Network), 1)))
      return ce #tf.reduce_mean(ce)

    def _build_value_mse(self):
      return tf.losses.mean_squared_error(self.input_value,
                                     self.Value_Network,
                                     weights=self.value_head_weight)


    def _build_train_op(self):
        l2_penalty_beta = 1e-4
        vars = tf.trainable_variables()
        l2_penalty = l2_penalty_beta * tf.add_n(
            [tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()])
        
        self.loss = self.value_mse + self.policy_entropy + l2_penalty
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.minimize(self.loss)

        return train_op

    def policy_value(self, state):
      action_probs, value = self.session.run(
          [self.Policy_Network, self.Value_Network], 
          feed_dict = {self.input_state : [state]})
      return np.exp(action_probs[0]), value

    def best_policy_value(self, state):
      action_probs, value = self.session.run(
          [self.Best_Policy_Network, self.Best_Value_Network], 
          feed_dict = {self.input_state : [state]})
      return np.exp(action_probs[0]), value

    def train(self, state_batch, action_batch, value_batch, step, 
                learning_rate):
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