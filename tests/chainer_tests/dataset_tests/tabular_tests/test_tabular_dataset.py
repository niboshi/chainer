import numpy as np
import unittest

from chainer import testing
from chainer.dataset import TabularDataset


class DummyDataset(TabularDataset):

    def __init__(self, mode, get_examples):
        self._mode = mode
        self._get_examples = get_examples

    def __len__(self):
        return 10

    @property
    def keys(self):
        return ('a', 'b', 'c')

    @property
    def mode(self):
        return self._mode

    def get_examples(self, indices, key_indices):
        return self._get_examples(indices, key_indices)


@testing.parameterize(*testing.product({
    'integer': [int, np.int32],
    'mode': [tuple, dict],
}))
class TestTabularDataset(unittest.TestCase):

    def test_getitem_index(self):
        def get_examples(indices, key_indices):
            self.assertEqual(indices, [3])
            self.assertIsNone(key_indices)
            return [3], [1], [4]

        dataset = DummyDataset(self.mode, get_examples)
        output = dataset[self.integer(3)]
        if self.mode is tuple:
            self.assertEqual(output, (3, 1, 4))
        elif self.mode is dict:
            self.assertEqual(output, {'a': 3, 'b': 1, 'c': 4})

    def test_getitem_indices_list(self):
        def get_examples(indices, key_indices):
            self.assertEqual(indices, [3, 1])
            self.assertIsNone(key_indices)
            return [3, 5], [1, 9], [4, 2]

        dataset = DummyDataset(self.mode, get_examples)
        output = dataset[[self.integer(3), self.integer(1)]]
        if self.mode is tuple:
            self.assertEqual(output, [(3, 1, 4), (5, 9, 2)])
        elif self.mode is dict:
            self.assertEqual(
                output, [{'a': 3, 'b': 1, 'c': 4}, {'a': 5, 'b': 9, 'c': 2}])

    def test_getitem_indices_array(self):
        def get_examples(indices, key_indices):
            self.assertEqual(indices, [3, 1])
            self.assertIsNone(key_indices)
            return [3, 5], [1, 9], [4, 2]

        dataset = DummyDataset(self.mode, get_examples)
        output = dataset[np.array((3, 1))]
        if self.mode is tuple:
            self.assertEqual(output, [(3, 1, 4), (5, 9, 2)])
        elif self.mode is dict:
            self.assertEqual(
                output, [{'a': 3, 'b': 1, 'c': 4}, {'a': 5, 'b': 9, 'c': 2}])

    def test_getitem_indices_slice(self):
        def get_examples(indices, key_indices):
            self.assertEqual(indices, slice(3, -1, -2))
            self.assertIsNone(key_indices)
            return [3, 5], [1, 9], [4, 2]

        dataset = DummyDataset(self.mode, get_examples)
        output = dataset[3::-2]
        if self.mode is tuple:
            self.assertEqual(output, [(3, 1, 4), (5, 9, 2)])
        elif self.mode is dict:
            self.assertEqual(
                output, [{'a': 3, 'b': 1, 'c': 4}, {'a': 5, 'b': 9, 'c': 2}])

    def test_as_tuple(self):
        dataset = DummyDataset(self.mode, None)
        view = dataset.as_tuple()
        self.assertIsInstance(view, TabularDataset)
        self.assertEqual(len(view), len(dataset))
        self.assertEqual(view.keys, dataset.keys)
        self.assertEqual(view.mode, tuple)

    def test_as_dict(self):
        dataset = DummyDataset(self.mode, None)
        view = dataset.as_dict()
        self.assertIsInstance(view, TabularDataset)
        self.assertEqual(len(view), len(dataset))
        self.assertEqual(view.keys, dataset.keys)
        self.assertEqual(view.mode, dict)


testing.run_module(__name__, __file__)
