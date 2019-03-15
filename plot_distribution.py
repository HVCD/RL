"""
Plot data distribution
Author: zhs
Date: Mar 8, 2019
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


directory = 'HVCD_csv/'
e_index = 1
index = str(e_index+1)
csv_name = directory + index + '.csv'

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
    states = episode[:, 0:state_dim]
    actions = episode[:, state_dim:r_index]

    reward = episode[:, r_index]
    reward /= 10e7
    reward = np.clip(reward, 0, 20)
    reward = norm_rewards(reward)
    # reward = np.nan_to_num(reward)

    return states, actions, reward


# ep_s, ep_a, ep_r = read_data(csv_name)
#
# plt.hist(ep_r, 50)
# plt.show()

EPISODES = 9
for e in range(EPISODES):
    index = str(e + 1)
    print("Training the {} episode".format(index))

    csv_name = directory + index + '.csv'
    ep_s, ep_a, ep_r = read_data(csv_name)
    plt.hist(ep_r, 50)
    plt.show()