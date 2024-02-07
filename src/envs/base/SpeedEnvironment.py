from src.envs.base.BaseEnvironment import Environment
from src.utils import Buffer
from typing import Tuple, Dict
from jax import numpy as jp
import jax
from brax.mjx.base import State as MjxState
import mujoco
from abc import ABC, abstractmethod


@jax.jit
def create_ones(size: Tuple[int, ...], dtype: jp.dtype = jp.float32) -> jp.ndarray:
    """
    Creates a JAX numpy array filled with ones.

    Parameters:
        size (Tuple[int, ...]): The shape of the array to create.
        dtype (jp.dtype, optional): The data type of the array, default is jp.float32.

    Returns:
        jp.ndarray: A JAX numpy array filled with ones.
    """
    return jp.ones(size, dtype=dtype)


class SpeedEnvironment(Environment, ABC):
    """
    Focus on making the robot zoom zoom zoom as fast as possible.
    """

    def __init__(self, model: mujoco.MjModel, physics_steps_per_env_step: int = 1, q_pos_init_noise: float = 0.1,
                 q_vel_init_noise: float = 0.1, buffer_size: int = 1, dtype: jp.dtype = jp.float32,
                 force_weights_fn: callable = create_ones, power_weights_fn_jit: callable = create_ones,
                 x_weights_fn_jit: callable = create_ones, yz_weights_fn_jit: callable = create_ones,
                 velocity_weights_fn_jit: callable = create_ones, n_contacts: int = 2):
        self.buffer_size = buffer_size
        self.n_contacts = n_contacts

        # Initialize weights for reward function
        self.force_weights = force_weights_fn((buffer_size, model.nu), dtype=dtype)
        self.power_weights = power_weights_fn_jit((buffer_size, 1), dtype=dtype)
        self.x_weights = x_weights_fn_jit((buffer_size, 1), dtype=dtype)
        self.yz_weights = yz_weights_fn_jit((buffer_size, 2), dtype=dtype)
        self.velocity_weights = velocity_weights_fn_jit((buffer_size - 1, 1), dtype=dtype)

        super().__init__(model, physics_steps_per_env_step, q_pos_init_noise, q_vel_init_noise, dtype)

    def initialize_buffers(self) -> Dict[str, Buffer]:
        """
        Initializes buffers for tracking past states or actions.

        Returns:
            Tuple[Buffer, Buffer, Buffer, Buffer]: The initialized buffers.
        """
        force_buffer = Buffer(self.buffer_size, (self.sys.nu,), dtype=self.dtype)
        power_buffer = Buffer(self.buffer_size, (1,), dtype=self.dtype)
        x_buffer = Buffer(self.buffer_size, (1,), dtype=self.dtype)
        yz_buffer = Buffer(self.buffer_size, (2,), dtype=self.dtype)
        return {
            "force": force_buffer,
            "power": power_buffer,
            "x": x_buffer,
            "yz": yz_buffer
        }

    def bad_state(self, state: MjxState) -> bool:
        """
        Determines whether the current state is bad, indicating that the episode should be terminated.

        Parameters:
            state (MjxState): The current state of the environment.

        Returns:
            bool: True if the state is bad, False otherwise.
        """
        return len(state.data.contact.geom1) > self.n_contacts

    @staticmethod
    def power(state: MjxState) -> jp.ndarray:
        """
        Calculates the power used by actuators in the given state.

        Parameters:
            state (MjxState): The current state from which to calculate power.

        Returns:
            jp.ndarray: The calculated power for each actuator.
        """
        # Retrieve actuator forces and velocities
        applied_actuator_forces = state.data.actuator_force  # Shape: (n_actuators,)
        applied_actuator_velocities = state.data.actuator_velocity  # Shape: (n_actuators,)
        # Calculate the power on each actuator
        power = jp.sum(applied_actuator_forces * applied_actuator_velocities)
        return power

    @staticmethod
    def force(state: MjxState) -> jp.ndarray:
        """
        Extracts the force applied by actuators in the given state.

        Parameters:
            state (MjxState): The current state from which to extract force data.

        Returns:
            jp.ndarray: The forces applied by each actuator.
        """
        return state.data.qfrc_actuator

    def reward(self, state: MjxState) -> jp.float32:
        """
        Computes the reward for the current state of the environment.

        Parameters:
            state (MjxState): The current state for which to calculate the reward.

        Returns:
            float: The computed reward value.
        """
        force_matrix = self.buffers["force"].matrix         # Shape: (buffer_size, n_actuators)
        power_matrix = self.buffers["power"].matrix         # Shape: (buffer_size, 1)
        x_matrix = self.buffers["x"].matrix                 # Shape: (buffer_size, 1)
        yz_matrix = self.buffers["yz"].matrix               # Shape: (buffer_size, 2)
        velocity_matrix = self.buffers["x"].matrix[1:] - self.buffers["x"].matrix[:-1]  # Shape: (buffer_size - 1, 1)

        force_reward = jp.sum(force_matrix * self.force_weights)
        power_reward = jp.sum(power_matrix * self.power_weights)
        x_reward = jp.sum(x_matrix * self.x_weights)
        yz_reward = jp.sum(yz_matrix * self.yz_weights)
        velocity_reward = jp.sum(velocity_matrix * self.velocity_weights)

        return force_reward + power_reward + x_reward + yz_reward + velocity_reward

    def append_buffer(self, state: MjxState):
        """
        Appends the given state to the environment's internal buffers for future reference.

        Parameters:
            state (MjxState): The state to append.
        """
        pos = state.data.subtree_com[1]

        power = self.power(state)
        force = self.force(state)
        x = pos[0]
        yz = pos[1:]

        self.buffers["force"].append(force)
        self.buffers["power"].append(jp.array([power]))
        self.buffers["x"].append(jp.array([x]))
        self.buffers["yz"].append(yz)

    def log_metrics(self, state: MjxState) -> Dict[str, jp.ndarray]:
        """
        Logs and returns metrics for the current state of the environment.

        Parameters:
            state (MjxState): The current state from which to log metrics.

        Returns:
            Dict[str, jp.ndarray]: A dictionary of logged metrics.
        """
        return {
            "q": state.q,
            "x": state.x,
            "qd": state.qd,
            "xd": state.xd,
            "actuator_force": state.data.qfrc_actuator,
            "actuator_velocity": state.data.actuator_velocity,
            "power": self.power(state),
        }

    @abstractmethod
    def get_observation(self, state: MjxState) -> jp.ndarray:
        """
        Gets the observation of the robot as a jax numpy tensor, matrix, or vector

        Parameters:
            state (MjxState): The state to get an observation from
        """
        raise NotImplementedError("`get_observation` method must be implemented in subclass")
