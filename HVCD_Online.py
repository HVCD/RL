"""
Read csv data, online training the policy network
Author: zhs
Date: Mar 12, 2019
"""
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from AC_continue_online import Actor, Critic

directory = '..\\..\\HVCD\\'

state_dim = 4
action_dim = 2
r_index = state_dim
done_index = 5


def reset(state_csv):
    initial_state = pd.read_csv(state_csv)
    initial_state = np.array(initial_state)
    return initial_state


def output_action(action):
    acc = str(action[0][0])
    phi = str(action[0][1])
    output_string = acc + '\t' + phi + '\n'
    action_file = open(directory + 'action.txt', mode='w')
    action_file.write(output_string)

def norm_rewards(ep_rewards):
    ep_rewards -= np.mean(ep_rewards)
    ep_rewards /= np.std(ep_rewards) + 1e-7
    return ep_rewards

def step(step_csv):
    while True:
        try:
            episode = pd.read_csv(step_csv)
            episode = np.array(episode)
            # print('@', episode)
            if len(episode) != 0:
                break
        except:
            episode = pd.read_csv(step_csv)
            episode = np.array(episode)
            print('#', episode)
            if len(episode) != 0:
                break
    #episode = np.array(episode)
    
    next_state = episode[:, :state_dim]
    reward = episode[:, r_index]
    # reward /= 10e7
    reward = np.clip(reward, 0, 20)
    reward = norm_rewards(reward)

    done = episode[:, done_index]

    time.sleep(0.1)
    return next_state, reward, done


EPISODES = 1  # 最大的序列数目
LR_A = 0.001  # learning rate for actor
LR_C = 0.01  # learning rate for critic
UPPER_BOUND = [0.50, 3.14 / 6]
LOWER_BOUND = [0, -3.14 / 6]
SAVE_STEP = 1000
STATE_CSV = 'state.csv'
STEP_CSV = directory + 'step.csv'


if __name__ == "__main__":
    # 初始化
    sess = tf.Session()
    actor = Actor(sess, state_dim, action_bound=[LOWER_BOUND, UPPER_BOUND])
    critic = Critic(sess, state_dim)
    sess.run(tf.global_variables_initializer())

    episode_r = []
    for e in range(EPISODES):
        index = str(e + 1)
        print("Training the {} episode".format(index))

        s, _, _ = step(STEP_CSV)
        #print(s.shape)
        #s = tf.reshape(s, [1, 4]) 
        while True:
            a = actor.choose_action(s)
            a = np.array(a)
            print('Action:', a)
            output_action(a)
            # 与环境交互
            s_, r, done = step(STEP_CSV)

            # 训练actor和critic网络
            td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
            actor.learn(s, a, td_error)  # true_gradient = grad[logPi(s,a) * td_error]

            actor.save_model(SAVE_STEP)
            critic.save_model(SAVE_STEP)

            s = s_
            episode_r.append(r)
            if done:
                output_pd = pd.DataFrame(episode_r)
                output_pd.to_csv('ep_reward.csv')
                break

