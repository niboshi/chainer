import chainer
import chainerx
import numpy

from chainerx_tests import math_utils
from chainerx_tests import op_utils


def _parameterize_exp():
    return chainer.testing.parameterize(*(
        # Special shapes
        chainer.testing.product({
            'shape': [(), (1,), (1, 1, 1), (2, 3)],
            'in_dtypes,out_dtype': math_utils.in_out_dtypes_math_functions,
            'input': [0, 2, -2],
        })
        # Special shapes (array.size = 0)
        + chainer.testing.product({
            'shape': [(0), (2, 0, 3)],
            'in_dtypes,out_dtype': math_utils.in_out_dtypes_math_functions,
            'input': [0, 2, -2],
            'check_numpy_strides_compliance': [False],
        })
        # Special values
        + chainer.testing.product({
            'shape': [(2, 3)],
            'in_dtypes,out_dtype': math_utils.in_out_float_dtypes_math_functions,
            'input': [float('inf'), -float('inf'), float('nan')],
            'skip_backward_test': [True],
            'skip_double_backward_test': [True],
        })
    ))


def _parameterize_log(xmin):
    return chainer.testing.parameterize(*(
        # Special shapes
        chainer.testing.product({
            'shape': [(), (1,), (1, 1, 1), (2, 3)],
            'in_dtypes,out_dtype': math_utils.in_out_dtypes_math_functions,
            'input': [
                xmin + 1,
                xmin + 3],
        })
        # Special shapes (array.size = 0)
        + chainer.testing.product({
            'shape': [(0,), (2, 0, 3)],
            'in_dtypes,out_dtype': math_utils.in_out_dtypes_math_functions,
            'input': [
                xmin + 1,
                xmin + 3],
            'check_numpy_strides_compliance': [False],
        })
        # Special values
        + chainer.testing.product({
            'shape': [(2, 3)],
            'in_dtypes,out_dtype': math_utils.in_out_float_dtypes_math_functions,
            'input': [
                float('inf'), -float('inf'), float('nan'),
                xmin,  # on the border
                xmin - 1,  # out of domain
            ],
            'skip_backward_test': [True],
            'skip_double_backward_test': [True],
        })
    ))


@op_utils.op_test(['native:0', 'cuda:0'])
@_parameterize_exp()
class TestErf(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        if xp is numpy:
            return chainer.functions.erf(a).array
        assert xp is chainerx
        return xp.erf(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@_parameterize_exp()
class TestExp(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.exp(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@_parameterize_exp()
class TestExpm1(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.expm1(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@_parameterize_exp()
class TestExp2(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.exp2(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@_parameterize_log(0)
class TestLog(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.log(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@_parameterize_log(0)
class TestLog10(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.log10(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@_parameterize_log(0)
class TestLog2(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.log2(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@_parameterize_log(-1)
class TestLog1p(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.log1p(a)
