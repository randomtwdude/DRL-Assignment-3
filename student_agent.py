from common import *

class Agent(object):
    def __init__(self, load = True):
        self.obs_size = (4, 84, 84)
        self.act_size = 12

        # networks
        self.device = "cpu"
        self.dqn    = Yugi(self.obs_size, self.act_size).to(self.device)

        if load:
            with open("thing_at_1000.bin", "rb") as data:
                stuff = torch.load(data)
                self.dqn.load_state_dict(stuff["model"])
                self.temp = stuff["temp"]
                self.beta = stuff["beta"]

        # Why have us implement proper wrappers and all
        # if we can't use them anyways?

        # preprocessing
        self.pp = T.Compose(
            [T.Grayscale(), T.Resize((84, 84), antialias = True), T.Normalize(0, 255)]
        )

        # frame stack & skipping
        self.skips_per_cycle = 3
        self.skip_count = 0
        self.frames = deque(maxlen = self.skips_per_cycle + 1)
        self.started = False
        self.last_act = 0

        self.gc_countdown = 0

        # rig
        SEED = 727
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)

    def act(self, observation):
        self.gc_countdown += 1
        if self.gc_countdown % 100 == 0:
            gc.collect()

        # preprocess
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        frame = self.pp(observation).squeeze(0).numpy()

        # init frame stack
        if not self.started:
            self.frames.clear()
            for _ in range(self.skips_per_cycle):
                self.frames.append(frame)
            self.started = True

        # frame skipping
        if self.skip_count > 0:
            self.skip_count -= 1
            return self.last_act

        self.skip_count = self.skips_per_cycle

        # add new frame to stack
        self.frames.append(frame)

        # stack frames for state
        f = np.stack(self.frames, axis = 0)
        state = torch.from_numpy(f).unsqueeze(0).to(self.device)

        selected_action = self.dqn(state).detach().numpy().squeeze().argmax().item()

        self.last_act = selected_action
        return selected_action