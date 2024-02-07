from typing import Tuple

from jax import numpy as jp
from jax import random
import jax


class Buffer:
    """
    A simple replay buffer designed for storing and sampling states in reinforcement learning scenarios.
    It utilizes JAX for efficient numerical computations and random sampling, supporting operations on
    JAX arrays.

    Attributes:
        buffer_size (int): Maximum capacity of the buffer to hold states.
        state_shape (Tuple[int, ...]): Shape of the states that will be stored in the buffer, accommodating
                                       multidimensional states.
        position (int): Current write position in the buffer, indicating where the next state will be inserted.
        size (int): Current size of the buffer, representing how many states are stored.
        seed (int): Seed for initializing the random number generator, ensuring reproducible sampling.
        rng (jp.ndarray): JAX random number generator for performing stochastic operations.
        buffer (jp.ndarray): The actual storage array holding the states, initialized based on `buffer_size`
                             and `state_shape`.

    Parameters:
        buffer_size (int): The maximum number of states the buffer can store.
        state_shape (Tuple[int, ...]): The shape of each state to be stored in the buffer.
        seed (int, optional): Initial seed for the random number generator. Default is 0.
        dtype (jp.dtype, optional): Data type of the stored states, typically set to jp.float32 for
                                    numerical stability and efficiency.
    """

    def __init__(self, buffer_size: int, state_shape: Tuple[int, ...], seed: int = 0, dtype=jp.float32):
        """
        Initializes the buffer with the given size, state shapes, and random seed.

        Parameters:
            buffer_size (int): The maximum number of states the buffer can hold.
            state_shape (tuple): The shapes of the state components to store.
            seed (int): Seed for the random number generator. Default is 0.
        """
        self.buffer_size = buffer_size
        self.state_shape = state_shape
        self.position = 0
        self.size = 0
        self.seed = seed
        self.rng = random.PRNGKey(seed)
        # Initialize buffers for each state component with zeros
        self.buffer = jp.zeros((buffer_size,) + state_shape, dtype=dtype)

    def append(self, item: jp.ndarray):
        """
        Adds a new state to the buffer, overwriting the oldest state if the buffer is full.

        Parameters:
            item (jp.ndarray): The state to add to the buffer.
        """
        # Add the new item to the buffer at the current position using the .at property
        self.buffer = self.buffer.at[self.position].set(item)
        # Update the position (increment and wrap around if necessary)
        self.position = (self.position + 1) % self.buffer_size
        # Update the size (ensure it does not exceed the buffer size)
        self.size = min(self.size + 1, self.buffer_size)

    def __getitem__(self, index: int) -> jp.ndarray:
        """
        Retrieves a state by its index in the buffer, with support for negative indices
        relative to the current 'position' in the circular buffer.

        Parameters:
            index (int): The index of the state to retrieve.

        Returns:
            jp.ndarray: The state at the specified index.
        """
        if index >= self.size or index < -self.size:
            raise IndexError("Index out of bounds")
        # Adjust the index for circular nature relative to the current position
        # If index is negative, it's a backward lookup from the current position
        true_index = (self.position + index) % self.size if index < 0 else index
        return self.buffer[true_index]

    def sample(self, batch_size: int) -> jp.ndarray:
        """
        Samples a batch of states from the buffer randomly.

        Parameters:
            batch_size (int): The number of states to sample.

        Returns:
            jp.ndarray: An array of states sampled randomly from the buffer.
        """
        # Ensure we can sample the requested batch size
        if batch_size > self.size:
            raise ValueError("Sample batch size exceeds buffer size")
        if batch_size < 0:
            raise ValueError("Sample batch size cannot be negative")
        # Generate random indices for sampling
        indices = random.choice(self.rng, self.size, shape=(batch_size,), replace=False)
        # Update the RNG key
        self.rng, new_rng = random.split(self.rng)
        # Return the sampled states
        return self.buffer[indices]

    @property
    def matrix(self) -> jp.ndarray:
        """
        Returns the buffer as a numpy tensor with the current position being the last row.

        Returns:
            jp.ndarray: The buffer as a numpy tensor. Shape is (buffer_size, state_shape).
        """
        if self.size < self.buffer_size:
            # If the buffer is not fully populated, return only the populated part, properly ordered.
            return jp.concatenate((self.buffer[self.position:self.size], self.buffer[:self.position]), axis=0)
        else:
            # If the buffer is fully populated, rotate it so the current position is at the end.
            return jp.concatenate((self.buffer[self.position:], self.buffer[:self.position]), axis=0)

    def __len__(self) -> int:
        """
        Returns the number of states currently in the buffer.

        Returns:
            int: The number of states currently in the buffer.
        """
        return self.size

