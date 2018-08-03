import tensorflow as tf
import numpy as np
import random
import time

from sixmok import Sixmok
from model import DQN


tf.app.flags.DEFINE_boolean("train", False, "학습모드. 게임을 화면에 보여주지 않습니다.")
FLAGS = tf.app.flags.FLAGS

LEARNING_EPISODE = 50000
TEST_EPISODE = 100
TARGET_UPDATE_INTERVAL = 1000
TRAIN_INTERVAL = 4
OBSERVE = 100
DISCOUNT_RATE = 0.99

WIDTH = 10
HEIGHT = 10
NUM_ACTION = WIDTH * HEIGHT

def discount_rewards(rewards, winner):
    if winner == 1:
        reward = 1
    elif winner == 2:
        reward = -1
    else:
        reward = 0

    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = reward + cumulative_rewards * DISCOUNT_RATE
        discounted_rewards[step] = cumulative_rewards

    mean = np.mean(discounted_rewards)
    std = np.std(discounted_rewards)
    return (discounted_rewards - mean) / std

def train():
    print('Train Mode')
    sess = tf.Session()

    game = Sixmok(HEIGHT, WIDTH, 3)
    brain = DQN(sess, HEIGHT, WIDTH, NUM_ACTION)

    rewards = tf.placeholder(tf.float32, [None])
    tf.summary.scalar('avg.reward/ep.', tf.reduce_mean(rewards))

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('logs', sess.graph)
    summary_merged = tf.summary.merge_all()

    brain.update_target_network()
    epsilon = 1.0
    time_step = 0
    total_reward_list = []

    for episode in range(LEARNING_EPISODE):
        winner = 0
        total_reward = 0
        player = 1
        p = 0
        state = game.reset(player)
        brain.init_state(state)

        data_states = []
        data_actions = []
        data_rewards = []

        while winner == 0:

            data_states.append(state)
            action = brain.get_action(state)

            if episode > OBSERVE:
                epsilon -= 1/10000

            next_state, reward, winner, last_action = game.step(player, action)
            data_actions.append(action)
            data_rewards.append(reward)
            
            total_reward += reward
            state = next_state

        print('게임횟수: %d 승자 : %d 점수: %d' % (episode + 1, winner, total_reward))

        data_rewards = discount_rewards(data_rewards, winner)

        for i in range(0, len(data_states)):
            brain.remember(data_states[i], data_actions[i], data_rewards[i])

        if (episode+1) % OBSERVE == 0:
            print("train")
            brain.train()

        if (episode+1) % TARGET_UPDATE_INTERVAL == 0:
                brain.update_target_network()

        total_reward_list.append(total_reward)

        if (episode+1) % 10 == 0:
            summary = sess.run(summary_merged, feed_dict={rewards: total_reward_list})
            writer.add_summary(summary, (episode+1))
            total_reward_list = []

        if (episode+1) % 100 == 0:
            saver.save(sess, 'model/dqn.ckpt', global_step = (episode+1))


def testPlay():
    print('Test Mode')
    sess = tf.Session()

    game = Sixmok(WIDTH, HEIGHT, 3)
    brain = DQN(sess, WIDTH, HEIGHT, NUM_ACTION)

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('model')
    saver.restore(sess, ckpt.model_checkpoint_path)

    one = 0
    two = 0
    for episode in range(TEST_EPISODE):
        total_reward = 0
        player = 1
        state = game.reset(player)
        brain.init_state(state)
        winner = 0
        
        while winner == 0:
            
            action = brain.get_action(state)
            next_state, reward, winner, last_action = game.step(player, action)
            total_reward += reward
            state = next_state

            if winner == 1:
                one += 1
                break
            elif winner == 2:
                two += 1
                break
            elif winner == 3:
                break
                player = 1

        print('게임횟수: %d 승자 : %d 점수: %d' % (episode + 1, winner, total_reward))
        print('last action : ', winner, [int(last_action/HEIGHT), last_action%HEIGHT])
        #if winner == 1 and player == 1:
        print(np.array(state))
        #if winner == 2 and player == 1:
            #print(np.array(state))

    print("player 1 win : ", one)
    print("player 2 win : ", two)

def main(_):
    if FLAGS.train:
        train()
    else:
        testPlay()

if __name__ == '__main__':
    tf.app.run()