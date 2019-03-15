"""
Actor-Critic with continuous action using TD-error as the Advantage, Reinforcement Learning.
Offline Training version
"""

import tensorflow as tf
import numpy as np

np.random.seed(2)
tf.set_random_seed(2)  # reproducible
GAMMA = 0.95


class Actor(object):
    def __init__(self, sess, n_features, action_bound, lr=0.0001):

        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.float32, None, name="act")  # only phi first
        self.td_error = tf.placeholder(tf.float32, None, name="td_error")  # TD_error

        self.mu_list = []
        self.sigma_list = []
        self.time_step = 0

        l1 = tf.layers.dense(
            inputs=self.s,
            units=30,  # number of hidden units
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.0),  # biases
            name='l1'
        )

        # l2 = tf.layers.dense(
        #     inputs=self.s,
        #     units=30,  # number of hidden units
        #     activation=tf.nn.relu,
        #     kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
        #     bias_initializer=tf.constant_initializer(0.0),  # biases
        #     name='l2'
        # )

        mu_acc = tf.layers.dense(
            inputs=l1,
            units=1,  # number of hidden units
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.),  # biases
            name='mu_a'
        )

        sigma_acc = tf.layers.dense(
            inputs=l1,
            units=1,  # output units
            activation=tf.nn.softplus,  # get action probabilities
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.),  # biases
            name='sigma_a'
        )

        mu_phi = tf.layers.dense(
            inputs=l1,
            units=1,  # number of hidden units
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.),  # biases
            name='mu_phi'
        )

        sigma_phi = tf.layers.dense(
            inputs=l1,
            units=1,  # output units
            activation=tf.nn.softplus,  # get action probabilities
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.),  # biases
            name='sigma_phi'
        )

        global_step = tf.Variable(0, trainable=False)

        self.mu_acc, self.sigma_acc = tf.squeeze(mu_acc*2), tf.squeeze(sigma_acc+0.1)  # 删除大小是1的维度
        self.mu_phi, self.sigma_phi = tf.squeeze(mu_phi * 2), tf.squeeze(sigma_phi + 0.1)  # 删除大小是1的维度
        # self.alpha = (self.mu[0] / self.sigma[0])**2
        # self.beta = self.mu[0] / (self.sigma[0] ** 2)

        self.normal_dist = tf.distributions.Normal([self.mu_acc, self.mu_phi], [self.sigma_acc, self.sigma_phi])
        self.action = tf.clip_by_value(self.normal_dist.sample(1), action_bound[0], action_bound[1])  # 从Normal分布中采样

        with tf.name_scope('exp_v'):
            log_prob = self.normal_dist.log_prob(self.a)  # loss without advantage
            self.exp_v = log_prob * self.td_error  # advantage (TD_error) guided loss
            # Add cross entropy cost to encourage exploration
            self.exp_v += 0.01*self.normal_dist.entropy()

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v, global_step)    # min(v) = max(-v)

        # save the parameters
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("actor_nn")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)

        mu1 = self.mu_acc.eval(session=self.sess, feed_dict=feed_dict)
        sigma1 = self.sigma_acc.eval(session=self.sess, feed_dict=feed_dict)
        mu2 = self.mu_phi.eval(session=self.sess, feed_dict=feed_dict)
        sigma2 = self.sigma_phi.eval(session=self.sess, feed_dict=feed_dict)

        self.mu_list.append([mu1, mu2])
        self.sigma_list.append([sigma1, sigma2])
        print("/ MEAN: ", [mu1, mu2], "  / SIGMA: ", [sigma1, sigma2])

        self.time_step += 1
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        #self.action = [self.action1, self.action2]
        return self.sess.run(self.action, {self.s: s})  # get probabilities for all actions

    def save_model(self, save_step):
        if self.time_step % save_step == 0:
            self.saver.save(self.sess, 'actor_nn/' + 'network' + '-actor', global_step=self.time_step)


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess
        with tf.name_scope('inputs'):
            self.s = tf.placeholder(tf.float32, [1, n_features], "state")
            self.v_ = tf.placeholder(tf.float32, [1, 1], name="v_next")
            self.r = tf.placeholder(tf.float32, name='r')
            self.time_step = 0

            self.saver = tf.train.Saver()
            checkpoint = tf.train.get_checkpoint_state("critic_nn")
            if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
                print("Successfully loaded:", checkpoint.model_checkpoint_path)
            else:
                print("Could not find old network weights")

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=30,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.0),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.0),  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = tf.reduce_mean(self.r + GAMMA * self.v_ - self.v)
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        v_ = self.sess.run(self.v, {self.s: s_})  # why?
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        self.time_step += 1
        return td_error

    def save_model(self, save_step):
        if self.time_step % save_step == 0:
            self.saver.save(self.sess, 'critic_nn/' + 'network' + '-critic', global_step=self.time_step)




