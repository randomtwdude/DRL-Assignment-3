from common import *

class Agent(object):
    def __init__(self, env,
        obs_size, act_size,
        epsilon_max = 1.0, epsilon_min = 0.02, epsilon_decay = 0.9999, gamma = 0.99,
        replay_buf_size = 20000, batch_size = 128, update_period = 300,
        learning_rate = 1e-4
    ):
        self.obs_size      = obs_size
        self.act_size      = act_size
        self.env           = env

        # parameters
        self.epsilon       = epsilon_max
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma         = gamma
        self.batch_size    = batch_size
        self.update_period = update_period
        self.update_until  = update_period

        # replay buffer
        self.mem           = ReplayBuffer(replay_buf_size, batch_size)

        # networks
        self.device        = "cpu"
        self.dqn           = Yugi(self.obs_size, self.act_size).to(self.device)
        self.dqn_target    = Yugi(self.obs_size, self.act_size).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        self.optimizer = optim.Adam(self.dqn.parameters(), lr = learning_rate)

    def act(self, observation):
        if self.epsilon > np.random.random():
            selected_action = self.env.action_space.sample()
        else:
            state = observation[0].__array__() if isinstance(observation, tuple) else observation.__array__()
            state = torch.tensor(state, device = self.device).unsqueeze(0)
            selected_action = self.dqn(state).argmax().item()

        return selected_action

    def step(self, state, action):
        next_state, reward, done, info = self.env.step(action)
        self.mem.add(state, action, next_state, reward, done)
        return next_state, reward, done, info

    def update(self):
        # check if we have enough experience
        if (len(self.mem) < self.batch_size):
            return None

        samples = self.mem.sample().to(self.device)
        loss    = self._calculate_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        # clip gradients
        nn.utils.clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()

        # decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

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

        states      = torch.FloatTensor(samples["states"]).to(dev)
        next_states = torch.FloatTensor(samples["states2"]).to(dev)
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