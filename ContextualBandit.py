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
        self.responsible_weight = tf.slice(self.output, self.action_holder, [1])
        self.loss = - (tf.log(self.responsible_weight) * self.reward_holder)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        self.update = optimizer.minimize(self.loss)


def train():
    tf.reset_default_graph()
    bandits = contextual_bandit()
    my_agent = agent(0.001,bandits.num_bandits,bandits.num_actions)
    weights = tf.trainable_variables()[0]

    total_episodes = 10000
    total_reward = np.zeros([bandits.num_bandits,bandits.num_actions])
    e=0.2

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        i=0
        while i < total_episodes:
            b = bandits.get_bandit()

            if np.random.rand(1) < e:
                action  = np.random.randint(bandits.num_actions)
            else:
                action = sess.run(my_agent.chosen_action, feed_dict={my_agent.state_in:[b]})

            reward = bandits.pull_arm(action)

            feed_dict = {my_agent.reward_holder:[reward],
                         my_agent.action_holder:[action],
                         my_agent.state_in:[b]}
            _,ww =sess.run([my_agent.update,weights], feed_dict=feed_dict)

            total_reward[b,action] += reward

            if i % 500 == 0:
                print('Mean reward for each of the {0}, bandits: {1}'.format(
                    bandits.num_bandits,
                    np.mean(total_reward,axis=1)))
            i += 1

    for a in range(bandits.num_bandits):
        print('The agent thinks action {0} for bandit {1} is the most promising'.format(
            np.argmax(ww[a])+1, a+1
        ))
        if np.argmax(ww[a]) == np.argmin(bandits.bandits[a]):
            print('...and it was right!')
        else:
            print('..and it was wrong!')


if __name__ == "__main__":
   train()




