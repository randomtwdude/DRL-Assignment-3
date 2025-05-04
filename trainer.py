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
        state = torch.tensor(observation, device = self.device).unsqueeze(0)
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
        indices = samples["indices"]
        weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)

        elementwise_loss = self._calculate_loss(samples)
        loss             = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()

        # update priority
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior - MATH_EPSILON
        self.mem.update(indices, new_priorities)

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

        # select with softmax weights
        with torch.no_grad():
            q_next_online = self.dqn(next_states)

            q_next_online = q_next_online / self.temp
            next_w = torch.softmax(q_next_online, dim = 1)

            q_next_target = self.dqn_target(next_states)
            next_q_value = torch.sum(q_next_target * next_w, dim = 1, keepdim = True)

        target = (rewards + self.gamma * next_q_value * (~dones)).to(dev)
        loss = F.smooth_l1_loss(curr_q_value, target, reduction = "none")
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

resolution = (84, 84)
frame_batch = 4

# apply preprocessing (the chain only works like this, do not change)
env = NvidiaFrameGeneration(env, n_frame = frame_batch)
env = Grayscaler(env)
env = Downscaler(env, resolution)
env = FrameStacker(env, num_stack = frame_batch)

# Train
agent = TrainingAgent(env, (frame_batch, *resolution), 12,
    k_max = 10.0, k_min = 0.25, k_decay = 0.9999, gamma = 0.92,
    replay_buf_size = 20000, batch_size = 192, update_period = 8192,
    learning_rate = 1e-4, load = False,
    alpha = 0.6, beta = 0.4
)

NUM_EPISODES = 1000
SAVE_INTERVAL = 25

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
        if frame_count % 4 == 0:
            agent.update()

        # Update the state and total reward
        total_reward += reward
        obs = next_obs
        frame_count += 1

        # Mario
        if info["flag_get"]:
            break

    # increase beta
    fraction = min((episode + 1) / NUM_EPISODES, 1.0)
    agent.beta = agent.beta + fraction * (1.0 - agent.beta)

    end = time.time()
    elapsed = end - start

    print(f"[{elapsed:.0f}s]\tEpisode\t{episode}\tReward\t{total_reward}\tTemp\t{agent.temp:.2f}\t{frame_count / elapsed:.2f}FPS")

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