import unittest

import numpy
import six

import chainer
from chainer import links
from chainer import testing

# Alt. 1: Single class to test all link features.
#
# - Runs forward and backward tests.
# - Runs initializer tests for each initializer.

@testing.parameterize(*(testing.product_dict(
    testing.product({
        'test': [True, False],
        'dtype': [numpy.float16, numpy.float32, numpy.float64],
    }),
    testing.product({
        'ndim': [0, 1, 2, 3],
    }) + [
        {'input_shape': (5, 4, 3, 2), 'axis': (0, 2, 3)},
        {'input_shape': (5, 4), 'axis': 0},
        {'input_shape': (5, 4, 3), 'axis': (0, 1)},
    ]
)))
@testing.inject_backend_tests(
    None,
    # CPU tests
    [
        {},
        {'use_ideep': 'always'},
    ]
    # GPU tests
    + testing.product({
        'use_cuda': [True],
        'use_cudnn': ['never', 'always'],
        'cuda_device': [0, 1],
    })
    # ChainerX tests
    + [
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
    ])
class BatchNormalizationTest(LinkTestCase):

    # Belows are "keys" that are used to extract information for
    # forward_expect.
    #
    # The benefit of declaring them here is that they are
    # readable, and also allows the base class to do some
    # "test_params_registerd".

    # Passed to forward_expect as numpy.ndarrays.
    params = ['gamma', 'beta']

    # Passed to forward_expect as is.
    attributes = ['eps']

    def setUp(self):
        if hasattr(self, 'axis') and hasattr(self, 'input_shape'):
            aggr_axes = self.axis
            if isinstance(self.axis, int):
                aggr_axes = self.axis,
            shape = self.input_shape
            param_shape = tuple(
                s for i, s in enumerate(shape) if i not in aggr_axes)
            expander = tuple(
                None if i in aggr_axes else slice(None)
                for i in range(len(shape)))
        elif hasattr(self, 'ndim'):
            aggr_axes = (0,) + tuple(six.moves.range(2, self.ndim + 2))
            shape = (5, 3) + (2,) * self.ndim
            param_shape = shape[1]
            expander = (None, Ellipsis) + (None,) * self.ndim
        else:
            assert False

        if self.test:
            mean = numpy.random.uniform(-1, 1, aggr_axes).astype(self.dtype)
            var = numpy.random.uniform(0.5, 1, aggr_axes).astype(self.dtype)
        else:
            mean = None
            var = None

        self.mean = mean
        self.var = var
        self.shape = shape
        self.param_shape = shape
        self.expander = expander
        self.finetune = False

    def generate_forward_backward_initializers(self):
        # Forward and backward tests should use random ndarray-initialized
        # parameters.
        axis = self.axis
        dtype = self.dtype

        initial_gamma = numpy.random.uniform(-1, 1, axis).astype(dtype)
        initial_beta = numpy.random.uniform(-1, 1, axis).astype(dtype)
        return initial_gamma, initial_beta

    @property
    def generate_initializers(self):
        # Various initializers should be tested.
        initial_gamma = [I.Constant(2), 1, (None, 1)]
        initial_beta = [I.Constant(2), 1, (None, 0)]
        return initial_gamma, initial_beta

    def create_link(self, initializers):
        initial_gamma, initial_beta = initializers

        link = links.BatchNormalization(
            size=self._param_shape,
            axis=self.axis,
            initial_gamma=initial_gamma,
            initial_beta=initial_beta,
            initial_avg_mean=None if self.mean is None self.mean.copy(),
            initial_avg_var=None if self.var is None else self.var.copy())
        return link

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return x,

    def forward(self, link, inputs):
        assert isinstance(input[0], chainer.Variable)

        x, = inputs
        y = link(x, finetune=self.finetune)
        return y

    def forward_expected(self, inputs, params, attrs):
        assert all(isinstance(p, numpy.ndarray) p for p in inputs + params)

        x, = inputs
        gamma, beta = params
        eps, = attrs

        if self.test:
            mean = self.mean
            var = self.var
            std = numpy.sqrt(var)
        else:
            mean = x.mean(axis=self.axis)
            var = x.var(axis=self.axis)
            std = numpy.sqrt(var + eps)
        y = gamma * (x - mean) / std + beta
        return y
