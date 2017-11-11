import unittest

import numpy as np

import data_utils


class TestShuffleSplit(unittest.TestCase):

    def test_shuffle_split_splits_based_on_fraction(self):
        (data1, labels1), (data2, labels2) = self.do_shuff_split(0.25)

        self.assertEquals(data1.shape, (1, 2))
        self.assertEquals(labels1.shape, (1, 1))
        self.assertEquals(data2.shape, (3, 2))
        self.assertEquals(labels2.shape, (3, 1))

    def test_shuffle_split_handles_0_fraction(self):
        (data1, labels1), (data2, labels2) = self.do_shuff_split(0.0)

        self.assertEquals(data1.shape, (0, 2))
        self.assertEquals(labels1.shape, (0, 1))
        self.assertEquals(data2.shape, (4, 2))
        self.assertEquals(labels2.shape, (4, 1))

    def test_shuffle_split_handles_1_fraction(self):
        (data1, labels1), (data2, labels2) = self.do_shuff_split(1.0)

        self.assertEquals(data1.shape, (4, 2))
        self.assertEquals(labels1.shape, (4, 1))
        self.assertEquals(data2.shape, (0, 2))
        self.assertEquals(labels2.shape, (0, 1))

    def test_shuffle_split_throws_on_bad_fraction(self):
        with self.assertRaises(ValueError):
            self.do_shuff_split(-1.0)
        with self.assertRaises(ValueError):
            self.do_shuff_split(1.01)

    def do_shuff_split(self, fraction):
        data = np.asarray([[1, 1], [2, 2], [3, 3], [4, 4]])
        labels = np.asarray([[1], [2], [3], [4]])
        return data_utils.shuffle_split(data, labels, fraction)

if __name__ == '__main__':
    unittest.main()
