import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class contextual_bandit():
    def __init__(self):
        self.state = 0
        self.bandits = np.array([[0.2, 0, -0.2, -5],
                                 [0.1, -5, 1, 0.25],
                                 [-5, 5, 5, 5]])
        self.num_bandits = self.bandits.shape[0]
        self.num_actions = self.bandits.shape[1]

    def get_bandit(self):
        self.state = np.random.randint(0,self.num_bandits)
        return self.state

    def pull_arm(self, action):
        bandit = self.bandits[self.state,action]
        result = np.random.randn(1)
        return 1 if result > bandit else -1


class agent():
    def __init__(self, learning_rate, s_size, a_size):
        self.state_in = tf.placeholder(shape=[1], dtype=tf.int32)
        state_in_OH = slim.one_hot_encoding(self.state_in, s_size)
        output = slim.fully_connected(state_in_OH,
                                      a_size,
                                      biases_initializer=None,
                                      activation_fn=tf.nn.sigmoid,
                                      weights_initializer= tf.ones_initializer())
        self.output = tf.reshape(output,[-1])
        self.chosen_action = tf.argmax(self.output,0)

        self.reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
        self.responsible_weight = th.slice(self.output, self.action_holder, [1])
        self.loss = - (tf.log(self.responsible_weight) * self.reward_holder)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        self.update = optimizer.minimize(self.loss)






