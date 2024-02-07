import logging

import jax
from jax import numpy as jp
import numpy as np
from brax.envs.base import MjxEnv
from brax.envs.base import State as BrxState
from brax.mjx.base import State as MjxState
import mujoco
from mujoco import mjx
from typing import Union, List, Dict
from abc import ABC, abstractmethod
from src.utils import Buffer
from logging import getLogger

np.set_printoptions(precision=3, suppress=True, linewidth=100)


class Environment(MjxEnv, ABC):
    """
    The `Environment` class serves as a foundation for creating reinforcement learning environments
    that combine JAX's numerical computation capabilities with MuJoCo's physics simulation strengths.
    It extends `MjxEnv` from Brax to facilitate environments that are compatible with both Brax's mjx
    and the standard MuJoCo physics models.

    Attributes:
        - q_pos_init_noise (float): Noise level for initial positions, helping introduce variability.
        - q_vel_init_noise (float): Noise level for initial velocities, aiding in diverse initial states.
        - buffer_size (int): Specifies how many past states or actions the environment should remember.
        - dtype (jp.dtype): Sets the data type for JAX numpy arrays, usually jp.float32 for efficiency.
        - velocity_weights, power_weights, x_weights, yz_weights, force_weights (jp.ndarray):
          These arrays define the weights used in various calculations, such as rewards based on velocity,
          power consumption, or force application.

    Parameters:
        - model (Union[mjx.Model, mujoco.MjModel]): The physics model underlying the environment. Supports
          both Brax's mjx.Model and MuJoCo's MjModel for flexibility.
        - metrics (List[str], optional): Metrics to track within the environment for analysis or debugging.
        - physics_steps_per_env_step (int, optional): Determines the granularity of the simulation by specifying
          the number of physics simulation steps per environment step.
        - q_pos_init_noise, q_vel_init_noise (float, optional): Initial noise levels for positions and velocities.
        - buffer_size (int, optional): Size of the internal buffer for state or action storage.
        - dtype (jp.dtype, optional): Preferred data type for JAX operations within the environment.
        - velocity_weights, power_weights, x_weights, yz_weights, force_weights (jp.ndarray, optional):
          Custom weights for calculating various aspects of the reward function.

    The class provides a structured way to implement custom environments by defining physics models,
    initial state noise for variability, buffer sizes for past state tracking, and weightings for
    different components of the reward function.
    """

    def __init__(self, model: Union[mjx.Model, mujoco.MjModel], physics_steps_per_env_step: int = 1,
                 q_pos_init_noise: float = 0.1, q_vel_init_noise: float = 0.1, dtype: jp.dtype = jp.float32):
        self.logger = getLogger(self.__class__.__name__)

        self.q_pos_init_noise = q_pos_init_noise
        self.q_vel_init_noise = q_vel_init_noise
        self.dtype = dtype

        self.buffers = self.initialize_buffers()
        super().__init__(model=model, n_frames=physics_steps_per_env_step)

    @abstractmethod
    def initialize_buffers(self) -> Dict[str, Buffer]:
        """
        Initializes the internal buffers for state or action storage.
        """
        raise NotImplementedError("`initialize_buffers` method must be implemented in subclass")

    def reset(self, random_number_array: jp.ndarray) -> BrxState:
        """
        Resets the environment to an initial state with optional noise.

        Parameters:
            random_number_array (jp.ndarray): An array of random numbers for generating initial state noise.

        Returns:
            BrxState: The new initial state of the environment.
        """
        # Split the provided random number array into separate keys for position and velocity noise
        random_number_array, q_pos_key, q_vel_key = jax.random.split(key=random_number_array, num=3)

        # Apply noise to the initial position and velocity based on the environment's configuration
        q_pos = self.sys.qpos0 + jax.random.uniform(key=q_pos_key, shape=(self.sys.nq,),
                                                    minval=-self.q_pos_init_noise, maxval=+self.q_pos_init_noise,
                                                    dtype=self.dtype)
        q_vel = jax.random.uniform(key=q_vel_key, shape=(self.sys.nv,),
                                   minval=-self.q_vel_init_noise, maxval=+self.q_vel_init_noise, dtype=self.dtype)

        # Initialize the simulation with the noisy state
        starting_state: MjxState = self.pipeline_init(q_pos, q_vel)

        # Create a new state object and populate it
        new_state = BrxState()
        new_state.pipeline_state = starting_state
        new_state.obs = self.get_observation(starting_state)
        new_state.reward = self.reward(starting_state)
        new_state.done = self.done(starting_state)
        new_state.metrics = self.log_metrics(starting_state)

        # Optionally append the new state to a buffer for historical tracking
        self.append_buffer(new_state.pipeline_state)
        return new_state

    def step(self, state: BrxState, action: jp.ndarray) -> BrxState:
        """
        Advances the environment by one step using the given action.

        Parameters:
            state (BrxState): The current state before the action is applied.
            action (jp.ndarray): The action to apply.

        Returns:
            BrxState: The updated state after applying the action.
        """
        # Advance the simulation and get the new state
        state_i: MjxState = state.pipeline_state
        state_j: MjxState = self.pipeline_step(state_i, action)

        # Append the new state to the buffer and update metrics
        self.append_buffer(state_j)
        state.metrics.update(self.log_metrics())

        # Update the BrxState object with new information
        state.replace(
            pipeline_state=state_j,
            obs=self.get_observation(state_j),
            reward=self.reward(state_j),
            done=self.done(state_j),
        )
        return state

    @abstractmethod
    def append_buffer(self, state: MjxState):
        """
        Appends the given state to the environment's internal buffers for future reference.

        Parameters:
            state (MjxState): The state to append.
        """
        raise NotImplementedError("`append_buffer` method must be implemented in subclass")

    @abstractmethod
    def get_observation(self, state: MjxState) -> jp.ndarray:
        """
        Gets the observation of the robot as a jax numpy tensor, matrix, or vector

        Parameters:
            state (MjxState): The state to get an observation from
        """
        raise NotImplementedError("`get_observation` method must be implemented in subclass")

    @abstractmethod
    def done(self, state: MjxState) -> bool:
        """
        Checks if the current episode is done.

        Parameters:
            state (MjxState): The current state to check for completion.

        Returns:
            bool: True if the episode is completed, False otherwise.
        """
        raise NotImplementedError("`done` method must be implemented in subclass")

    @abstractmethod
    def reward(self, state: MjxState):
        """
        Computes the reward for the current state of the environment.

        Parameters:
            state (MjxState): The current state for which to calculate the reward.

        Returns:
            float: The computed reward value.
        """
        raise NotImplementedError("`reward` method must be implemented in subclass")

    @abstractmethod
    def log_metrics(self, state: MjxState) -> Dict[str, jp.ndarray]:
        """
        Logs and returns metrics for the current state of the environment.

        Parameters:
            state (MjxState): The current state from which to log metrics.

        Returns:
            Dict[str, jp.ndarray]: A dictionary containing logged metrics.
        """
        raise NotImplementedError("`log_metrics` method must be implemented in subclass")
