import tensorflow as tf
import numpy as np
import random
import time

from collections import deque
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

def augmenting_data(states, current_players, actions, winners):
    augmented_states = []
    augmented_actions = []
    augmented_winners = []

    for i in range(0, len(states)):
        state = states[i]
        action = actions[i]
        winner = winners[i]
        player = current_players[i]

        x = int(action / HEIGHT)
        y = action % HEIGHT
        action_2d = np.zeros(WIDTH * HEIGHT, dtype=int).reshape(WIDTH, HEIGHT)
        action_2d[x][y] = player

        for j in [1, 2, 3, 4]:
            rotate_state = np.rot90(state, j)
            rotate_action_2d = np.rot90(action_2d, j)
            index = np.where(rotate_action_2d != 0)
            x = index[0][0]
            y = index[1][0]
            rotate_action = x * HEIGHT + y
            augmented_states.append(rotate_state)
            augmented_actions.append(rotate_action)
            augmented_winners.append(winner)

        flip_state = np.fliplr(state)
        flip_action_2d = np.fliplr(action_2d)
        index = np.where(flip_action_2d != 0)
        x = index[0][0]
        y = index[1][0]
        flip_action = x * HEIGHT + y
        augmented_states.append(flip_state)
        augmented_actions.append(flip_action)
        augmented_winners.append(winner)

    return augmented_states, augmented_actions, augmented_winners


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

    one = 0
    two = 0
    data = deque(maxlen=10000)
    for episode in range(LEARNING_EPISODE):
        game.reset()
        winner, turns, states, current_players, actions, winners \
                = game.runSelfPlay()
        print('%d play : winner is %d' %(episode+1, winner))
        total_reward_list.append(winners[0])
        if winner == 1:
            one += 1
        elif winner == 2:
            two += 1

        if (episode+1) % OBSERVE == 0:
            print("player 1 : ", one)
            print("player 2 : ", two)
            one = 0
            two = 0

        augmented_states, augmented_actions, augmented_winners \
                = augmenting_data(states, current_players, actions, winners)

        play_data = list(zip(augmented_states, augmented_actions,
                        augmented_winners))[:]
        data.extend(play_data)

        if (episode+1) % 100 == 0:
            mini_batch = random.sample(data, 100)
            states_batch = [d[0] for d in mini_batch]
            actions_batch = [d[1] for d in mini_batch]
            winners_batch = [d[2] for d in mini_batch]
            brain.train(states_batch, actions_batch, winners_batch, 0.01)
            data = []

        """
        if (episode+1) % 100 == 0:
            summary = sess.run(summary_merged, feed_dict={rewards: total_reward_list})
            writer.add_summary(summary, (episode+1))
            total_reward_list = []

        if (episode+1) % 200 == 0:
            saver.save(sess, 'model/dqn.ckpt', global_step = (episode+1))
        """


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
        winner, turns, states, current_players, actions, winners = game.runSelfPlay()
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