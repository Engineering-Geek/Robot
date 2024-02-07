from typing import Tuple
from jax import numpy as jp
from jax import random


class Buffer:
    """
    The `Buffer` class serves as a flexible storage system for reinforcement learning applications, utilizing
    JAX for efficient management and random sampling of states. It is designed to store a history of states and
    retrieve them in a randomized manner, which is essential for algorithms involving temporal data.

    Attributes:
        buffer_size (int): The total capacity, indicating the maximum number of states it can store.
        state_shape (Tuple[int, ...]): The shape of each state, allowing for multi-dimensional data storage.
        position (int): Points to the next insertion spot, managed internally for circular buffer utilization.
        size (int): The current number of states stored, increases with each new state until reaching `buffer_size`.
        seed (int): The seed value for the random number generator, ensuring reproducible sampling operations.
        rng (jp.ndarray): JAX's random number generator instance, used for random sampling.
        buffer (jp.ndarray): The main storage array, initialized based on `buffer_size` and `state_shape`.

    Methods:
        __init__: Initializes a new buffer with the specified size, state shapes, and optional seed and dtype.
        append: Adds a new state to the buffer, overwriting the oldest state if the buffer is full.
        __getitem__: Retrieves a state by its index, supporting negative indexing relative to the current position.
        sample: Randomly samples a batch of states, with the batch size not exceeding the current buffer size.
        matrix: Provides a view of the buffer's contents, adjusting so the most recent state is last.
        __len__: Returns the current number of states in the buffer.
    """

    def __init__(self, buffer_size: int, state_shape: Tuple[int, ...], seed: int = 0, dtype=jp.float32):
        """
        Initializes a new buffer instance.

        Parameters:
            buffer_size (int): Maximum number of states the buffer can hold.
            state_shape (Tuple[int, ...]): Shape of each state, allowing multi-dimensional storage.
            seed (int, optional): Seed for the random number generator, defaulting to 0.
            dtype (jp.dtype, optional): Data type of stored states, default is jp.float32.
        """
        self.buffer_size = buffer_size
        self.state_shape = state_shape
        self.position = 0
        self.size = 0
        self.seed = seed
        self.rng = random.PRNGKey(seed)
        self.buffer = jp.zeros((buffer_size,) + state_shape, dtype=dtype)

    def append(self, item: jp.ndarray):
        """
        Adds a new state to the buffer, overwriting the oldest state if full.

        Parameters:
            item (jp.ndarray): The state to be added, matching the specified `state_shape`.
        """
        self.buffer = self.buffer.at[self.position].set(item)
        self.position = (self.position + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def __getitem__(self, index: int) -> jp.ndarray:
        """
        Retrieves a state by its index, supporting negative indexing.

        Parameters:
            index (int): The index of the desired state.

        Returns:
            jp.ndarray: The state at the specified index.
        """
        if index >= self.size or index < -self.size:
            raise IndexError("Index out of bounds")
        true_index = (self.position + index) % self.size if index < 0 else index
        return self.buffer[true_index]

    def sample(self, batch_size: int) -> jp.ndarray:
        """
        Randomly samples a batch of states from the buffer.

        Parameters:
            batch_size (int): The number of states to sample, not exceeding `size`.

        Returns:
            jp.ndarray: An array of randomly sampled states.
        """
        if batch_size > self.size:
            raise ValueError("Sample batch size exceeds buffer size")
        if batch_size < 0:
            raise ValueError("Sample batch size cannot be negative")
        indices = random.choice(self.rng, self.size, shape=(batch_size,), replace=False)
        self.rng, _ = random.split(self.rng)
        return self.buffer[indices]

    @property
    def matrix(self) -> jp.ndarray:
        """
        Provides a view of the buffer's contents, with the most recent state last.

        Returns:
            jp.ndarray: The buffer's contents, properly ordered.
        """
        if self.size < self.buffer_size:
            return jp.concatenate((self.buffer[self.position:self.size], self.buffer[:self.position]), axis=0)
        else:
            return jp.concatenate((self.buffer[self.position:], self.buffer[:self.position]), axis=0)

    def __len__(self) -> int:
        """
        Returns the current number of states in the buffer

        Returns:
            int: The number of states currently stored.
        """
        return self.size
