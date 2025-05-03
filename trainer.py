from common import *
from student_agent import Agent
import time

class TrainingAgent(Agent):
    def __init__(self, env,
        obs_size, act_size,
        k_max = 5.0, k_min = 0.1, k_decay = 0.9999, gamma = 0.99,
        replay_buf_size = 20000, batch_size = 128, update_period = 300,
        learning_rate = 1e-4, load = False,
        alpha = 0.6, beta = 0.5
    ):
        super().__init__(obs_size, act_size, load = load)

        # parameters
        if not load:
            self.temp    = k_max
            self.beta    = beta

        self.env           = env
        self.k_min         = k_min
        self.k_decay       = k_decay
        self.gamma         = gamma
        self.batch_size    = batch_size
        self.update_period = update_period
        self.update_until  = update_period

        # replay buffer
        self.mem           = ReplayBuffer(replay_buf_size, obs_size, batch_size, alpha = alpha)

        # networks
        # -- dqn network handled in Agent --
        self.dqn_target    = Yugi(self.obs_size, self.act_size).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        self.optimizer = optim.Adam(self.dqn.parameters(), lr = learning_rate)

    def act(self, observation):
        state = observation[0].__array__() if isinstance(observation, tuple) else observation.__array__()
        state = torch.tensor(state, device = self.device).unsqueeze(0)
        ratings = self.dqn(state).detach().numpy().squeeze()

        ratings = ratings - np.max(ratings)
        ratings_exp = np.exp(ratings / self.temp)
        probs = ratings_exp / np.sum(ratings_exp)

        selected_action = np.random.choice(range(self.act_size), p = probs)
        return selected_action

    def step(self, obs, action):
        next_obs, reward, done, info = self.env.step(action)
        self.mem.add(obs, action, next_obs, reward, done)
        return next_obs, reward, done, info

    def update(self):
        # check if we have enough experience
        if (len(self.mem) < self.batch_size):
            return None

        samples = self.mem.sample()
        loss    = self._calculate_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()

        # decrease temperature
        self.temp = max(self.k_min, self.temp * self.k_decay)

        # hard update
        self.update_until -= 1
        if self.update_until == 0:
            self._sync_target()
            self.update_until = self.update_period

        return loss.item()

    def _sync_target(self):
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def _calculate_loss(self, samples):
        dev = self.device

        states      = torch.FloatTensor(samples["obs"]).to(dev)
        next_states = torch.FloatTensor(samples["next_obs"]).to(dev)
        actions     = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(dev)
        rewards     = torch.FloatTensor(samples["rewards"].reshape(-1, 1)).to(dev)
        dones       = torch.BoolTensor(samples["dones"].reshape(-1, 1)).to(dev)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_q_value = self.dqn(states).gather(1, actions)
        next_q_value = self.dqn_target(next_states) \
            .gather(1, self.dqn(next_states).argmax(dim = 1, keepdim = True)) \
            .detach()

        target = (rewards + self.gamma * next_q_value * (~dones)).to(dev)

        # calculate dqn loss
        loss = F.smooth_l1_loss(curr_q_value, target)
        return loss

# Training loop

# Set random seed
"""
SEED = 727
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
"""

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
agent = TrainingAgent(env, (frame_batch, *resolution), 12,
    k_max = 10.0, k_min = 0.1, k_decay = 0.99999, gamma = 0.95,
    replay_buf_size = 100000, batch_size = 192, update_period = 384,
    learning_rate = 1e-4, load = False,
    alpha = 0.6, beta = 0.5
)

NUM_EPISODES = 1000
SAVE_INTERVAL = 10

for episode in range(NUM_EPISODES):
    start = time.time()
    frame_count = 0

    # Reset the environment
    obs = agent.env.reset()
    done = False

    # No-op start
    for _ in range(random.randint(1, 4)):
        obs, _, done, _ = agent.env.step(env.action_space.sample())
        if done:
            obs = agent.env.reset()

    total_reward = 0

    while not done:
        # select action
        act = agent.act(obs)
        next_obs, reward, done, info = agent.step(obs, act)

        # train
        agent.update()

        # increase beta
        fraction = min((episode + 1) / NUM_EPISODES, 1.0)
        agent.beta = agent.beta + fraction * (1.0 - agent.beta)

        # Update the state and total reward
        total_reward += reward
        obs = next_obs
        frame_count += 1

        # Mario
        if info["flag_get"]:
            break

    end = time.time()
    elapsed = end - start

    print(f"[{elapsed:.0f}s] Episode {episode}\tReward {total_reward}\tTemp {agent.temp:.2f}\t{frame_count / elapsed}FPS")

    if (episode + 1) % SAVE_INTERVAL == 0:
        name = f"thing_at_{episode + 1}.bin"
        torch.save(
            dict(
                model = agent.dqn.state_dict(),
                temp = agent.temp,
                beta = agent.beta
            ),
            name
        )
        print(f"Saved @ {episode + 1}")