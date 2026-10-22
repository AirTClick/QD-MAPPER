"""
QD-MAPPER addition to make LTF one-shot.
"""
import gymnasium
import numpy as np

WAIT = 0  # POGEMA's action 0 is "stay" (GridConfig.MOVES[0] == [0, 0])


class StopAtFirstGoal(gymnasium.Wrapper):

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        self._reached = np.zeros(self.env.unwrapped.grid_config.num_agents,
                                 dtype=bool)
        self._update_reached()
        return observations, infos

    def step(self, actions):
        actions = np.array(actions)
        actions[self._reached] = WAIT
        observations, rewards, terminated, truncated, infos = self.env.step(
            actions)
        self._update_reached()
        return observations, rewards, terminated, truncated, infos

    def _update_reached(self):
        was_on_goal = self.env.unwrapped.was_on_goal
        if was_on_goal is not None:
            self._reached |= np.array(was_on_goal, dtype=bool)
