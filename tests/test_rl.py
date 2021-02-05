from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.utils import common


import tensorflow as tf
from PIL import Image
import numpy as np

###
num_iterations = 20000 # @param {type:"integer"}
 
initial_collect_steps = 1000  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}
 
batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}
 
num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}


env_name = 'CartPole-v0'
env = suite_gym.load(env_name)

env.reset()
Image.fromarray(env.render())

# Print Envs
print('Observation Spec:')
print(env.time_step_spec().observation)

print('Reward Spec:')
print(env.time_step_spec().reward)

print('Action Spec:')
print(env.action_spec())

time_step = env.reset()
print('Time step:')
print(time_step)
 
action = np.array(1, dtype=np.int32)
 
next_time_step = env.step(action)
print('Next time step:')
print(next_time_step)


train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

fc_layer_params = (100,)
 
q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)
 
agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)
 
agent.initialize()


# env = gym.make('Pendulum-v0')
# env.reset()

# for i in range(10):
#     env.render()
#     action = env.action_space.sample()
#     state, reward, done, info = env.step(action)
#     print("action",action , " reward", reward, " done" , done , " info",info)



# env = gym.make('CarRacing-v0')
# observation = env.reset()
# for t in range(1000):
#     env.render()
#     observation, reward, done, info = env.step(env.action_space.sample())
# env.close()

# 上記を実行することで1000ステップのゲームの実行を行います。画像を出力したい際は下記のコードを実行することで出力することができます。

# import matplotlib.pyplot as plt
# plt.imshow(observation)
# plt.show()
