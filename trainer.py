from common import *
from student_agent import Agent

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

resolution = (91, 91)
frame_batch = 4

# apply preprocessing (the chain only works like this, do not change)
env = NvidiaFrameGeneration(env, n_frame = frame_batch)
env = Grayscaler(env)
env = Downscaler(env, resolution)
env = FrameStacker(env, num_stack = frame_batch)

# Train
agent = Agent(env, (frame_batch, *resolution), 12,
    epsilon_max = 1.0, epsilon_min = 0.02, epsilon_decay = 0.9999, gamma = 0.99,
    replay_buf_size = 20000, batch_size = 128, update_period = 300,
    learning_rate = 1e-4
)

NUM_EPISODES = 50
REPORT_INTERVAL = 1
SAVE_INTERVAL = 2

reward_history = [] # Store the total rewards for each episode

for episode in range(NUM_EPISODES):
    # TODO: Reset the environment
    state = agent.env.reset()
    total_reward = 0
    done = False

    while not done:
        # select action
        action = agent.act(state)
        next_state, reward, done, info = agent.step(state, action)

        # train
        agent.update()

        # Update the state and total reward
        total_reward += reward
        state = next_state

        # Mario
        if info["flag_get"]:
            break

    if (episode + 1) % REPORT_INTERVAL == 0:
        print(f"Episode {episode + 1}, Reward: {np.mean(reward_history[-REPORT_INTERVAL:])}, Eps: {agent.epsilon}")
    reward_history.append(total_reward)

    if (episode + 1) % SAVE_INTERVAL == 0:
        name = f"thing_at_{episode + 1}.bin"
        torch.save(
            dict(
                model = agent.dqn.state_dict(),
                epsilon = agent.epsilon
            ),
            name
        )
        print(f"Saved @ {episode + 1}")