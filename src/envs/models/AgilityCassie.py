from brax.mjx.base import State as MjxState

from src.envs.base.SpeedEnvironment import SpeedEnvironment, create_ones
from jax import numpy as jp


class CassieSpeedEnvironment(SpeedEnvironment):
    def __init__(self, model, physics_steps_per_env_step=1, max_episode_time: int = 10, q_pos_init_noise=0.1,
                 q_vel_init_noise=0.1, buffer_size=1, dtype=jp.float32, force_weights_fn=create_ones,
                 power_weights_fn_jit=create_ones, x_weights_fn_jit=create_ones, yz_weights_fn_jit=create_ones,
                 velocity_weights_fn_jit=create_ones):
        self.max_episode_time = max_episode_time
        n_contacts = 2
        super().__init__(model, physics_steps_per_env_step, q_pos_init_noise, q_vel_init_noise, buffer_size, dtype,
                         force_weights_fn, power_weights_fn_jit, x_weights_fn_jit, yz_weights_fn_jit,
                         velocity_weights_fn_jit, n_contacts)

    def bad_state(self, state: MjxState) -> bool:
        """
        Determines whether the current state is bad, indicating that the episode should be terminated.

        Parameters:
            state (MjxState): The current state of the environment.

        Returns:
            bool: True if the state is bad, False otherwise.
        """
        return len(state.data.contact.geom1) > 2

    def done(self, state: MjxState) -> bool:
        """
        Determines whether the episode is done based on the current state.

        Each simulation for Cassie will end when the time limit is reached or the number of contacts is greater than 2
        (i.e. the robot has fallen or is in a bad state).

        Parameters:
            state (MjxState): The current state of the environment.

        Returns:
            bool: True if the episode is done, False otherwise.
        """
        data = state.data
        return data.time > self.max_episode_time or len(data.contact.geom1) > 2

