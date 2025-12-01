import gymnasium as gym

class RailroadEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = None
        self.action_space = None

    def reset(self, seed=None, options=None):
        return None, {}

    def step(self, action):
        return None, 0, False, False, {}
