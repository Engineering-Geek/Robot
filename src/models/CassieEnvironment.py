from typing import Union, List, Dict

import mujoco
import numpy as np
from brax.mjx.base import State as MjxState
from jax import numpy as jp
from mujoco import mjx

from src.models.BaseEnvironment import Environment
from src.utils import Buffer

np.set_printoptions(precision=3, suppress=True, linewidth=100)


class CassieEnvironment(Environment):
    """
    Environment for the Cassie humanoid robot, interfacing with the Brax physics engine.

    This class provides a simple interface for working with the Cassie humanoid robot within the Brax physics engine.
    """

    def __init__(self, model: Union[mjx.Model, mujoco.MjModel], metrics: List[str] = None,
                 physics_steps_per_env_step: int = 1, q_pos_init_noise: float = 0.1, q_vel_init_noise: float = 0.1,
                 dtype: jp.dtype = jp.float32, buffers: Dict[str, Buffer] = None):
        """
        Initializes the environment with the specified model, metrics, and configuration.
        :param model: The physical model to use for the environment. This can be a mujoco.MjModel or mjx.Model.
        :param metrics: Names of the metrics to track within the environment.
        :param physics_steps_per_env_step: Number of physics simulation steps per environment step.
        :param q_pos_init_noise: Initial noise range for positions.
        :param q_vel_init_noise: Initial noise range for velocities.
        :param dtype: Data type for JAX computations.
        :param buffers: Buffers for storing state or other information.
        """
        super().__init__(model, metrics, physics_steps_per_env_step, q_pos_init_noise, q_vel_init_noise, buffers, dtype)

    def reward(self, state: MjxState) -> float:
        """
        Calculates the reward for the given state.

        Parameters:
            state (MjxState): The current state of the Cassie robot.

        Returns:
            float: The calculated reward.
        """
        # Placeholder: Compute the reward based on the state of the Cassie robot.
        return 0.0  # Replace with actual reward calculation.

    def done(self, state: MjxState) -> bool:
        """
        Determines whether the episode is done based on the given state.

        Parameters:
            state (MjxState): The current state of the Cassie robot.

        Returns:
            bool: True if the episode is done, False otherwise.
        """
        # Placeholder: Determine if the episode is done (e.g., robot has fallen over).
        return False  # Replace with actual logic.

    def log_metrics(self, state: MjxState) -> Dict[str, jp.ndarray]:
        """
        Logs metrics based on the current state.

        Parameters:
            state (MjxState): The current state of the Cassie robot.

        Returns:
            Dict[str, jp.ndarray]: A dictionary of logged metrics.
        """
        # Placeholder: Log various metrics, such as distance traveled, energy used, etc.
        return {}  # Replace with actual metrics.

    def _get_observation(self, state: MjxState) -> jp.ndarray:
        """
        Retrieves the current observation of the environment.

        Parameters:
            state (MjxState): The current state of the Cassie robot.

        Returns:
            jp.ndarray: The current observation.
        """
        # Placeholder: Extract the observation from the state.
        camera_obs = state.data.cvel
        return jp.array([])  # Replace with actual observation extraction.

    def append_buffer(self, state: MjxState):
        """
        Appends the given state to the environment's buffer(s).

        Parameters:
            state (MjxState): The state to append to the buffer.
        """
        # Placeholder: Implement the logic to append the state to the buffers.
        pass  # Replace with actual buffer appending logic.
