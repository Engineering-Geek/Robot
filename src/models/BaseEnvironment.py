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

np.set_printoptions(precision=3, suppress=True, linewidth=100)


class Environment(MjxEnv, ABC):
    """
    A base environment class for reinforcement learning that integrates with both JAX for numerical
    computations and MuJoCo for physics simulations. This class extends `MjxEnv` from Brax, providing
    a framework for environments that can operate with either mjx.Model or mujoco.MjModel as their
    underlying physics models.

    Attributes:
        q_pos_init_noise (float): Standard deviation of the noise added to the initial positions.
        q_vel_init_noise (float): Standard deviation of the noise added to the initial velocities.
        buffer_size (int): Size of the buffer for storing past environment states or actions.
        dtype (jp.dtype): Data type for JAX numpy operations, typically jp.float32.
        velocity_weights (jp.ndarray): Weights for velocity-related calculations or rewards.
        power_weights (jp.ndarray): Weights for power-related calculations or rewards.
        x_weights (jp.ndarray): Weights for x-axis position-related calculations or rewards.
        yz_weights (jp.ndarray): Weights for y and z-axis positions-related calculations or rewards.
        force_weights (jp.ndarray): Weights for force-related calculations or rewards.

    Parameters:
        model (Union[mjx.Model, mujoco.MjModel]): The physics model to use for the environment, supporting
                                                  both mjx.Model from Brax's mjx and mujoco.MjModel.
        metrics (List[str], optional): A list of metric names to track within the environment.
        physics_steps_per_env_step (int, optional): Number of physics simulation steps per environment step.
        q_pos_init_noise (float, optional): Initial position noise standard deviation.
        q_vel_init_noise (float, optional): Initial velocity noise standard deviation.
        buffer_size (int, optional): Buffer size for storing environment states or actions.
        dtype (jp.dtype, optional): The data type for JAX numpy computations.
        velocity_weights (jp.ndarray, optional): Velocity weights for reward calculations.
        power_weights (jp.ndarray, optional): Power weights for reward calculations.
        x_weights (jp.ndarray, optional): X-axis position weights for reward calculations.
        yz_weights (jp.ndarray, optional): Y and Z-axis position weights for reward calculations.
        force_weights (jp.ndarray, optional): Force weights for reward calculations.
    """
    def __init__(self, model: Union[mjx.Model, mujoco.MjModel], metrics: List[str] = None,
                 physics_steps_per_env_step: int = 1, q_pos_init_noise: float = 0.1, q_vel_init_noise: float = 0.1,
                 buffer_size: int = 1, dtype: jp.dtype = jp.float32, velocity_weights: jp.ndarray = None,
                 power_weights: jp.ndarray = None, x_weights: jp.ndarray = None, yz_weights: jp.ndarray = None,
                 force_weights: jp.ndarray = None, ):
        self.q_pos_init_noise = q_pos_init_noise
        self.q_vel_init_noise = q_vel_init_noise
        metrics = metrics if metrics is not None else []
        self.metrics = {metric: jp.zeros(1) for metric in metrics}
        self.dtype = dtype

        self.power_buffer = Buffer(buffer_size=buffer_size, state_shape=(self.sys.nu,))
        self.x_buffer = Buffer(buffer_size=buffer_size, state_shape=(1,))
        self.yz_buffer = Buffer(buffer_size=buffer_size, state_shape=(2,))
        self.force_buffer = Buffer(buffer_size=buffer_size, state_shape=(self.sys.nu,))
        self.buffer_size = buffer_size

        if power_weights.shape != (buffer_size, self.sys.nu):
            logging.warning(f"Power weights shape {power_weights.shape} does not match buffer size {buffer_size} and "
                            f"actuator count {self.sys.nu}. Using ones instead.")
            power_weights = jp.ones(buffer_size, self.sys.nu)
        if velocity_weights.shape != (buffer_size, 3):
            logging.warning(f"Velocity weights shape {velocity_weights.shape} does not match buffer size {buffer_size}."
                            f" Using ones instead.")
            velocity_weights = jp.ones((buffer_size, 3))
        if x_weights.shape != (buffer_size, 1):
            logging.warning(f"X weights shape {x_weights.shape} does not match buffer size {buffer_size}. "
                            f"Using ones instead.")
            x_weights = jp.ones((buffer_size, 1))
        if yz_weights.shape != (buffer_size, 2):
            logging.warning(f"YZ weights shape {yz_weights.shape} does not match buffer size {buffer_size}. "
                            f"Using ones instead.")
            yz_weights = jp.ones((buffer_size, 2))
        if force_weights.shape != (buffer_size, self.sys.nu):
            logging.warning(f"Force weights shape {force_weights.shape} does not match buffer size {buffer_size} and "
                            f"actuator count {self.sys.nu}. Using ones instead.")
            force_weights = jp.ones(buffer_size, self.sys.nu)

        self.power_weights = power_weights
        self.velocity_weights = velocity_weights
        self.x_weights = x_weights
        self.yz_weights = yz_weights
        self.force_weights = force_weights

        # make sure all weights are of the same length except for velocity weights, which should be one less
        if len(self.power_weights) != len(self.velocity_weights) + 1:
            raise ValueError("Power weights and velocity weights should have the same length")
        if len(self.power_weights) != len(self.x_weights):
            raise ValueError("Power weights and x weights should have the same length")
        if len(self.power_weights) != len(self.yz_weights):
            raise ValueError("Power weights and yz weights should have the same length")
        if len(self.power_weights) != len(self.force_weights):
            raise ValueError("Power weights and force weights should have the same length")

        super().__init__(model=model, n_frames=physics_steps_per_env_step)

    @staticmethod
    def power(state: MjxState) -> jp.ndarray:
        # Retrieve actuator forces and velocities
        applied_actuator_forces = state.data.actuator_force         # Shape: (n_actuators,)
        applied_actuator_velocities = state.data.actuator_velocity  # Shape: (n_actuators,)
        # Calculate the power on each actuator
        power = jp.sum(applied_actuator_forces * applied_actuator_velocities)
        return power

    @staticmethod
    def force(state: MjxState) -> jp.ndarray:
        return state.data.actuator_force

    def reset(self, random_number_array: jp.ndarray) -> BrxState:
        # Split the provided random number array into separate keys for position and velocity noise
        random_number_array, q_pos_key, q_vel_key = jax.random.split(key=random_number_array, num=3)

        # Apply noise to the initial position and velocity based on the environment's configuration
        q_pos = self.sys.qpos0 + jax.random.uniform(key=q_pos_key, shape=(self.sys.nq,),
                                                    minval=-self.q_pos_init_noise, maxval=+self.q_pos_init_noise,
                                                    dtype=self.dtype)
        q_vel = jax.random.uniform(key=q_vel_key, shape=(self.sys.nv,),
                                   minval=-self.q_vel_init_noise, maxval=+self.q_vel_init_noise, dtype=self.dtype)

        # Initialize the simulation with the noisy state
        starting_state = self.pipeline_init(q_pos, q_vel)

        # Get the initial observation, set initial reward and done flag
        observation = self._get_observation(starting_state)
        reward, done = jp.zeros(2)

        # Initialize metrics for the new episode
        metrics = {metric: jp.zeros(1) for metric in self.metrics}

        # Create a new state object and populate it
        new_state = BrxState()
        new_state.pipeline_state = starting_state
        new_state.obs = observation
        new_state.reward = reward
        new_state.done = done
        new_state.metrics = metrics

        # Optionally append the new state to a buffer for historical tracking
        self.append_buffer(new_state)
        return new_state

    def reward(self, state: MjxState) -> jp.float32:
        """
        Calculates and returns the reward for the current state of the environment.

        .. math::
            \sum_{i=1}^{n} w_i \cdot m_i

        where :math:`w_i` is the weight for metric :math:`m_i` for each metric :math:`m_i` in the environment.

        Returns:
            float: The calculated reward.
        """
        force_matrix = self.force_buffer.matrix                                     # Shape: (buffer_size, n_actuators)
        power_matrix = self.power_buffer.matrix                                     # Shape: (buffer_size, n_actuators)
        x_matrix = self.x_buffer.matrix                                             # Shape: (buffer_size, 1)
        yz_matrix = self.yz_buffer.matrix                                           # Shape: (buffer_size, 2)
        velocity_matrix = (x_matrix - jp.roll(x_matrix, shift=1, axis=0)) / self.dt # Shape: (buffer_size, 1)

        force_reward = jp.sum(jp.dot(force_matrix, self.force_weights))
        power_reward = jp.sum(jp.dot(power_matrix, self.power_weights))
        x_reward = jp.sum(jp.dot(x_matrix, self.x_weights))
        yz_reward = jp.sum(jp.dot(yz_matrix, self.yz_weights))
        velocity_reward = jp.sum(jp.dot(velocity_matrix, self.velocity_weights))

        return force_reward + power_reward + x_reward + yz_reward + velocity_reward

    def append_buffer(self, state: MjxState):
        """
        Appends the given state to the environment's buffers.

        Parameters:
            state (MjxState): The state to append to the buffer.
        """
        pos = state.data.subtree_com[1]

        power = self.power(state)
        force = self.force(state)
        x = pos[0]
        yz = pos[1:]

        self.power_buffer.append(power)
        self.x_buffer.append(x)
        self.yz_buffer.append(yz)
        self.force_buffer.append(force)

    def step(self, state: BrxState, action: jp.ndarray) -> BrxState:
        """
        Advances the environment by one step based on the given action and updates the state.

        Parameters:
            state (BrxState): The current state of the environment.
            action (jp.ndarray): The action to apply to the environment.

        Returns:
            BrxState: The updated state after applying the action.
        """
        # Advance the simulation and get the new state
        state_i: MjxState = state.pipeline_state
        state_j: MjxState = self.pipeline_step(state_i, action)

        # Append the new state to the buffer and update metrics
        self.append_buffer(state_j)
        state.metrics.update(**self.log_metrics())

        # Update the BrxState object with new information
        state.replace(
            pipeline_state=state_j,
            obs=self._get_observation(state_j),
            reward=self.reward(state_j),
            done=self.done(state_j),
        )
        return state

    @abstractmethod
    def done(self, state: MjxState) -> bool:
        """
        Determines whether the current episode has ended.

        Must be implemented by subclasses.

        Returns:
            bool: True if the episode is done, otherwise False.
        """
        raise NotImplementedError("`done` method must be implemented in subclass")

    @abstractmethod
    def log_metrics(self, state: MjxState) -> Dict[str, jp.ndarray]:
        """
        Logs and returns metrics related to the environment's current state.

        Must be implemented by subclasses.

        Returns:
            Dict[str, jp.ndarray]: A dictionary of logged metrics.
        """
        raise NotImplementedError("`log_metrics` method must be implemented in subclass")

    @abstractmethod
    def _get_observation(self, state: MjxState) -> jp.ndarray:
        """
        Retrieves and returns the current observation from the environment.

        Must be implemented by subclasses.

        Returns:
            jp.ndarray: The current observation.
        """
        raise NotImplementedError("`_get_observation` method must be implemented in subclass")
