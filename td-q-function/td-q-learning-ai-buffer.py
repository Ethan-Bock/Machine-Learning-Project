#!/usr/bin/env python3
import numpy as np 
import gym

import sys
import argparse
import logging
import os.path
import joblib
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd

import collections

class QFunction:
    def __init__(self, state_shape, n_actions):
        # state_shape = tuple of integers; number of state variables in each dimension
        self.Q = self.create_Q_function(state_shape, n_actions)
        self.state_shape = state_shape
        self.num_actions = n_actions
        self.replay_buffer = collections.deque(maxlen=10000)
        return

    def record_experience(self, state, action, reward, next_state, epoch_done, epoch_truncated):
        self.replay_buffer.append((state, action, reward, next_state, epoch_done, epoch_truncated))
        return

    def n_actions(self):
        return self.num_actions

    def actions(self):
        return [action for action in range(self.n_actions())]

    def create_Q_function(self, state_shape, n_actions):
        model = keras.models.Sequential()

        print("state_shape", state_shape)
        #Can Edit This but make note of it
        model.add(keras.layers.Input(shape=state_shape))
        model.add(keras.layers.Dense(32, activation="elu"))
        model.add(keras.layers.Dense(16, activation="elu"))
        model.add(keras.layers.Dense(n_actions, activation="linear"))

        model.compile(loss="mse", optimizer=keras.optimizers.Adam())
        print("model summary", model.summary())
        
        return model

    def sample_replay_buffer(self, batch_size):
        indices = np.random.randint(len(self.replay_buffer), size=batch_size)
        batch = [self.replay_buffer[index] for index in indices]
        states, actions, rewards, next_states, dones, truncateds = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(6)]
        return states, actions, rewards, next_states, dones, truncateds


    def train_from_experiences(self, batch_size, gamma):
        if len(self.replay_buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones, truncateds = self.sample_replay_buffer(batch_size)
        this_predictions = self.Q.predict(states)
        next_predictions = self.Q.predict(next_states)
        
        max_next_predictions = np.max(next_predictions, axis=1)
        completeds = dones | truncateds
        target_values = (rewards + (1-completeds)*gamma*max_next_predictions)
        target_values = target_values.reshape(-1, 1)

        mask = tf.one_hot(actions, self.num_actions)
        
        new_values = mask * target_values + (1-mask)*this_predictions

        self.Q.fit(states, new_values, epochs=1, verbose=0)
        return

    def update(self, state, action, next_state, reward, gamma, prediction, next_prediction):
        target_quality_value = reward + gamma * np.max(next_prediction)

        target_vec = prediction
        target_vec[0][action] = target_quality_value

        state = np.array([state])
        
        self.Q.fit(state, target_vec, epochs=1, verbose=0)
        return

    def predict(self, state):
        state = state.reshape(-1, state.shape[0])
        prediction = self.Q.predict(state)
        return prediction
    
    def get_best_action(self, state):
        state = state.reshape(-1, state.shape[0])
        prediction = self.Q.predict(state)
        action = np.argmax(prediction[0])
        return action

    def get_Q_value(self, state, action):
        state = state.reshape(-1, state.shape[0])
        prediction = self.Q.predict(state)
        return prediction[0][action]

    def get_best_action_value(self, state):
        state = state.reshape(-1, state.shape[0])
        prediction = self.Q.predict(state)
        best_action = np.argmax(prediction[0])
        best_Q_value = prediction[0][action]
        return best_action, best_Q_value

    def show(self):
        state = np.array([[0.0,0.0,0.0,0.0]])
        prediction = self.Q.predict(state)
        print(prediction)
        return

    def save(self, model_file):
        self.Q.save(model_file)
        return

    def load(self, model_file):
        self.Q = keras.models.load_model(model_file)
        return

def get_model_filename(model_file, environment_name):
    if model_file == "":
        model_file = "{}-model.keras".format(environment_name)
    return model_file

def get_rewards_filename(model_file, environment_name):
    if model_file == "":
        model_file = "{}-rewards.csv".format(environment_name)
    return model_file

def load_environment(my_args):
    if my_args.track_steps:
        render_mode = "human"
    else:
        render_mode = None
    if my_args.environment == 'car':
        env = gym.make('MountainCar-v0', render_mode=render_mode)
    else:
        raise Exception("Unexpected environment: {}".format(my_args.environment))
    return env

def learn_epoch(Q, env, chance_epsilon, gamma, batch_size, my_args):
    action_list = Q.actions()
    state, info = env.reset()
    prediction = Q.predict(state)
    
    epoch_total_reward = 0
    epoch_done = False
    epoch_truncated = False

    while (not epoch_done) and (not epoch_truncated):
        
        chance = np.random.sample(1)[0]
        if chance < chance_epsilon:
            action = np.random.choice(action_list)
        else:
            action = Q.get_best_action(state)

        next_state, reward, epoch_done, epoch_truncated, info = env.step(action)
        
        ### BUMP
        position = next_state[0]
        velocity = next_state[1]
        if (position == 0.6):
            reward += 250
        else:
            #Change
            reward += abs(position + 0.5) + ( 10 * (velocity ** 2))

        Q.record_experience(state, action, reward, next_state, epoch_done, epoch_truncated)
        next_prediction = Q.predict(next_state)

        Q.train_from_experiences(batch_size, gamma)
        epoch_total_reward += reward
        state = next_state
        prediction = next_prediction

    return state, epoch_total_reward

def evaluate_epoch(Q, env, my_args):
    action_list = Q.actions()

    state, info = env.reset()
    epoch_total_reward = 0
    epoch_done = False
    epoch_truncated = False

    while (not epoch_done) and (not epoch_truncated):
        action = Q.get_best_action(state)

        next_state, reward, epoch_done, epoch_truncated, info = env.step(action)

        #### BUMP
        position = next_state[0]
        velocity = next_state[1]
        if (position == 0.6):
            reward += 250
        else:
            reward += abs(position + 0.5) + ( 10 * (velocity ** 2))

        epoch_total_reward += reward
        state = next_state

    return state, epoch_total_reward

def Q_learn(Q, env, my_args):
    almost_one = my_args.epsilon_chance_factor
    gamma = my_args.gamma
    batch_size = my_args.batch_size
    epoch_rewards = []
    chance_epsilon = almost_one

    for epoch_number in range(my_args.n_epochs):
        state, epoch_total_reward = learn_epoch(Q, env, chance_epsilon, gamma, batch_size, my_args)
        epoch_rewards.append(epoch_total_reward)
        if my_args.track_epochs:
            print("epoch: {}  reward: {}".format(epoch_number, epoch_total_reward))
            sys.stdout.flush()

        if epoch_total_reward > 40:
            chance_epsilon *= almost_one
            chance_epsilon = max(chance_epsilon, 0.01)
        if my_args.early_stop:
            if len(epoch_rewards) > 5:
                if sum(epoch_rewards[len(epoch_rewards)-5:]) == 5*500:
                    break
        
    return epoch_rewards

def Q_evaluate(Q, env, my_args):
    epoch_rewards = []

    for epoch_number in range(my_args.n_epochs):
        state, epoch_total_reward = evaluate_epoch(Q, env, my_args)
        epoch_rewards.append(epoch_total_reward)
        if my_args.track_epochs:
            print("epoch: {}  reward: {}".format(epoch_number, epoch_total_reward))
        
    return epoch_rewards

def do_learn(my_args):
    env = load_environment(my_args)
    Q = QFunction(env.observation_space.shape, env.action_space.n)

    model_file = get_model_filename(my_args.model_file, my_args.environment)
    if os.path.exists(model_file):
        print("Model loading from {}.".format(model_file))
        Q.load(model_file)
    
    epoch_rewards = Q_learn(Q, env, my_args)

    print("Learn: Average reward on all epochs " + str(sum(epoch_rewards)/my_args.n_epochs))

    model_file = get_model_filename(my_args.model_file, my_args.environment)
    Q.save(model_file)
    print("Model saved to {}.".format(model_file))

    rewards_file = get_rewards_filename(my_args.rewards_file, my_args.environment)
    df = pd.DataFrame(columns = ["epoch","reward"])
    for i in range(0, len(epoch_rewards)):
        df.loc[i] = [i, epoch_rewards[i]]
    df.to_csv(rewards_file, index=False)
    
    return

def do_score(my_args):
    env = load_environment(my_args)

    Q = QFunction([0], 0)
    model_file = get_model_filename(my_args.model_file, my_args.environment)
    print("Model loading from {}.".format(model_file))
    Q.load(model_file)

    epoch_rewards = Q_evaluate(Q, env, my_args)
    print("Score: Average reward on all epochs " + str(sum(epoch_rewards)/my_args.n_epochs))
    return

def parse_args(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description='Q-Table Learning')
    parser.add_argument('action', default='learn',
                        choices=[ "learn", "score", ], 
                        nargs='?', help="desired action")
    
    parser.add_argument('--environment',   '-e', default="car", type=str,  choices=('car', ), help="name of the OpenAI gym environment")
    parser.add_argument('--model-file',    '-m', default="",    type=str,   help="name of file for the model (default is constructed from environment)")
    parser.add_argument('--rewards-file',  '-r', default="",    type=str,   help="name of file for the rewards (default is constructed from environment)")

    # hyper parameters
    parser.add_argument('--gamma', '-g', default=0.5,  type=float, help="Q-learning hyper parameter (default=0.5)")
    parser.add_argument('--epsilon-chance-factor', '-c', default=0.1,  type=float, help="Scaling factor for learning policy chance of choosing random action (default=0.1)")
    
    #Make note of change
    parser.add_argument('--batch-size', '-b',   default=8, type=int,   help="number of experiences to learn from (default=16).")

    parser.add_argument('--n-epochs', '-n',   default=10, type=int,   help="number of episodes to run (default=10).")

    # debugging/observations
    parser.add_argument('--track-epochs',    '-t', default=0,         type=int,   help="0 = don't display per-epoch information, 1 = do display per-epoch information (default=0)")
    parser.add_argument('--track-steps',     '-s', default=0,         type=int,   help="0 = don't display per-step information, 1 = do display per-step information (default=0)")
    parser.add_argument('--early-stop',                         action='store_true',  help="Stop learning if perfect score is found.")
    parser.add_argument('--no-early-stop',   dest="early_stop", action='store_false', help="Do not stop learning if perfect score is found.")
    parser.set_defaults(early_stop=False)

    my_args = parser.parse_args(argv[1:])
    return my_args

def main(argv):
    my_args = parse_args(argv)
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)

    if my_args.action == 'learn':
        do_learn(my_args)
    elif my_args.action == 'score':
        do_score(my_args)
    else:
        raise Exception("Action: {} is not known.".format(my_args.action))

    return

if __name__ == "__main__":
    main(sys.argv)

    