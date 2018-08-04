import tensorflow as tf
import numpy as np
import random
import time

from sixmok import Sixmok
from model import PolicyValueNet


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

    brain = PolicyValueNet(sess, HEIGHT, WIDTH)
    game = Sixmok(HEIGHT, WIDTH, brain)

    rewards = tf.placeholder(tf.float32, [None])
    tf.summary.scalar('avg.reward/ep.', tf.reduce_mean(rewards))

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('logs', sess.graph)
    summary_merged = tf.summary.merge_all()

    epsilon = 1.0
    time_step = 0
    total_reward_list = []

    for episode in range(LEARNING_EPISODE):
        game.reset()
        winner, turns, states, actions, winners = game.runSelfPlay()
        print('%d play : winner is %d' %(episode+1, winner))
        total_reward_list.append(winners[0])

        brain.train(states, actions, winners, 0.01)

        if (episode+1) % 10 == 0:
            summary = sess.run(summary_merged, feed_dict={rewards: total_reward_list})
            writer.add_summary(summary, (episode+1))
            total_reward_list = []

        if (episode+1) % 100 == 0:
            saver.save(sess, 'model/dqn.ckpt', global_step = (episode+1))


def testPlay():
    print('Test Mode')
    sess = tf.Session()

    brain = PolicyValueNet(sess, HEIGHT, WIDTH)
    game = Sixmok(WIDTH, HEIGHT, brain)
    
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('model')
    saver.restore(sess, ckpt.model_checkpoint_path)

    one = 0
    two = 0
    for episode in range(TEST_EPISODE):
        game.reset()
        winner, turns, states, actions, winners = game.runSelfPlay()
        print('%d play : winner is %d' %(episode+1, winner))
        print(np.array(states[len(states)-1]))
        if winner == 1:
            one += 1
        elif winner == 2:
            two += 1

    print("player 1 win : ", one)
    print("player 2 win : ", two)

def main(_):
    if FLAGS.train:
        train()
    else:
        testPlay()

if __name__ == '__main__':
    tf.app.run()