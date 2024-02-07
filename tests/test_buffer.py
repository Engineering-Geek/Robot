from src.utils import Buffer
import unittest
from jax import numpy as jp


class BufferTest(unittest.TestCase):
    def setUp(self):
        # Initialize a buffer with a size of 5 and state shapes of 2
        self.buffer = Buffer(buffer_size=5, state_shape=(2,))
        # Fill the buffer with test data
        for i in range(5):
            self.buffer.append(jp.array([i, i + 1]))

    def test_get_item_positive_index(self):
        # Test retrieval with positive indices
        self.assertTrue(jp.all(self.buffer[0] == jp.array([0, 1])))
        self.assertTrue(jp.all(self.buffer[1] == jp.array([1, 2])))

    def test_get_item_negative_index(self):
        # Test retrieval with negative indices
        # Assuming the buffer is full and position is at index 0
        self.assertTrue(jp.all(self.buffer[-1] == jp.array([4, 5])))
        self.assertTrue(jp.all(self.buffer[-2] == jp.array([3, 4])))

    def test_get_item_out_of_bounds(self):
        # Test out-of-bounds access
        with self.assertRaises(IndexError):
            _ = self.buffer[5]
        with self.assertRaises(IndexError):
            _ = self.buffer[-6]

    def test_get_item_circular(self):
        # Test that the circular nature works
        # Simulate the buffer being full and the position being at index 3
        self.buffer.position = 3
        self.assertTrue(jp.all(self.buffer[-1] == jp.array([2, 3])))  # Should be the item at index 2
        self.assertTrue(jp.all(self.buffer[-3] == jp.array([0, 1])))  # Should wrap and be the item at index 0


if __name__ == '__main__':
    unittest.main()
