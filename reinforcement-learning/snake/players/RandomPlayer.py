

class RandomPlayer():

    def choose_action(self, env):
        return env.action_space.sample()
