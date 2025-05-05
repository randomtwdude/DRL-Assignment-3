from common import *

class Agent(object):
    # todo: remove sizes
    def __init__(self, load = True):
        self.obs_size = (4, 84, 84)
        self.act_size = 12

        # networks
        self.device = "cpu"
        self.dqn    = Yugi(self.obs_size, self.act_size).to(self.device)

        if load:
            with open("thing_at_700.bin", "rb") as data:
                stuff = torch.load(data)
                self.dqn.load_state_dict(stuff["model"])
                self.temp = stuff["temp"]
                self.beta = stuff["beta"]

    def act(self, observation):
        state = observation[0].__array__() if isinstance(observation, tuple) else observation.__array__()
        state = torch.tensor(state, device = self.device).unsqueeze(0)
        selected_action = self.dqn(state).argmax().item()

        return selected_action