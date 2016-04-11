# Author: Carlos Xavier Hernandez <cxh@stanford.edu>
# Contributors:
# Copyright (c) 2016, Stanford University and the Authors
# All rights reserved.

from __future__ import print_function, division, absolute_import

from sklearn import preprocessing

from .base import (MultiSequencePreprocessingMixin,
                   MultiSequenceOnlinePreprocessingMixin)


class Binarizer(MultiSequencePreprocessingMixin, preprocessing.Binarizer):
    __doc__ = preprocessing.Binarizer.__doc__

# Older versions of sklearn might not have this
if hasattr(preprocessing, 'FunctionTransformer'):
    class FunctionTransformer(MultiSequencePreprocessingMixin,
                              preprocessing.FunctionTransformer):
        __doc__ = preprocessing.FunctionTransformer.__doc__


class Imputer(MultiSequencePreprocessingMixin, preprocessing.Imputer):
    __doc__ = preprocessing.Imputer.__doc__


class KernelCenterer(MultiSequencePreprocessingMixin,
                     preprocessing.KernelCenterer):
    __doc__ = preprocessing.KernelCenterer.__doc__


class LabelBinarizer(MultiSequencePreprocessingMixin,
                     preprocessing.LabelBinarizer):
    __doc__ = preprocessing.LabelBinarizer.__doc__


class MultiLabelBinarizer(MultiSequencePreprocessingMixin,
                          preprocessing.MultiLabelBinarizer):
    __doc__ = preprocessing.MultiLabelBinarizer.__doc__

# Older versions of sklearn might not have this
if hasattr(preprocessing.MinMaxScaler, 'partial_fit'):
    class MinMaxScaler(MultiSequenceOnlinePreprocessingMixin,
                       preprocessing.MinMaxScaler):
        __doc__ = preprocessing.MinMaxScaler.__doc__

# Older versions of sklearn might not have this
if hasattr(preprocessing, 'MaxAbsScaler'):
    class MaxAbsScaler(MultiSequenceOnlinePreprocessingMixin,
                       preprocessing.MaxAbsScaler):
        __doc__ = preprocessing.MaxAbsScaler.__doc__


class Normalizer(MultiSequencePreprocessingMixin, preprocessing.Normalizer):
    __doc__ = preprocessing.Normalizer.__doc__

# Older versions of sklearn might not have this
if hasattr(preprocessing, 'RobustScaler'):
    class RobustScaler(MultiSequencePreprocessingMixin,
                       preprocessing.RobustScaler):
        __doc__ = preprocessing.RobustScaler.__doc__

# Older versions of sklearn might not have this
if hasattr(preprocessing.StandardScaler, 'partial_fit'):
    class StandardScaler(MultiSequenceOnlinePreprocessingMixin,
                         preprocessing.StandardScaler):
        __doc__ = preprocessing.StandardScaler.__doc__


class PolynomialFeatures(MultiSequencePreprocessingMixin,
                         preprocessing.PolynomialFeatures):
    __doc__ = preprocessing.PolynomialFeatures.__doc__