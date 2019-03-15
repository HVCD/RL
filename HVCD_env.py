"""
Read csv data and train the policy network
Author: zhs
Date: Mar 8, 2019
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from AC_continue import Actor, Critic

directory = 'HVCD_csv/'

state_dim = 4
action_dim = 2
r_index = (state_dim+action_dim)

def norm_rewards(ep_rewards):
    ep_rewards -= np.mean(ep_rewards)
    ep_rewards /= np.std(ep_rewards)
    return ep_rewards


def read_data(csv_name):
    episode = pd.read_csv(csv_name)
    episode = np.array(episode)
    # print(episode)
    states = episode[:, 0:state_dim]
    actions = episode[:, state_dim:r_index]
    reward = episode[:, r_index]
    reward /= 10e7
    reward = np.clip(reward, 0, 20)
    reward = norm_rewards(reward)
    # reward = np.nan_to_num(reward)

    return states, actions, reward


EPISODES = 9
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic
UPPER_BOUND = [0.50, 3.14/6]
LOWER_BOUND = [-0.1, -3.14/6]
SAVE_STEP = 1000


if __name__ == "__main__":
    sess = tf.Session()

    actor = Actor(sess=sess, n_features=state_dim, action_bound=[LOWER_BOUND, UPPER_BOUND]) # action1_bound=[0, 0.50], action2_bound=[-3.14/6, 3.14/6])
    critic = Critic(sess, state_dim)
    sess.run(tf.global_variables_initializer())

    for e in range(EPISODES):
        index = str(e + 1)
        print("Training the {} episode".format(index))
        print("######################\n")

        csv_name = directory + index + '.csv'
        ep_s, ep_a, ep_r = read_data(csv_name)

        for i in range(len(ep_r)):
            s = ep_s[i]
            # 与环境交互
            a = ep_a[i]
            # print("ACTION: ", a)
            try:
                s_, r = ep_s[i+1], ep_r[i]
                # print("REWARD: ", r)
            except:
                break

            # 训练actor和critic网络
            td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
            actor.learn(s, a, td_error)  # true_gradient = grad[logPi(s,a) * td_error]

            actor.save_model(SAVE_STEP)
            critic.save_model(SAVE_STEP)

    mu_df = pd.DataFrame(actor.mu_list)
    sigma_df = pd.DataFrame(actor.sigma_list)
    mu_df.to_csv('mean.csv')
    sigma_df.to_csv('sigma.csv')


