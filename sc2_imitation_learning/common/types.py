from typing import Union, Iterable, Mapping, Text

from sonnet.src.types import ShapeLike
from tensorflow import DType, TensorSpec

ShapeNest = Union[ShapeLike, Iterable['ShapeNest'], Mapping[Text, 'ShapeNest'], ]
DTypeNest = Union[DType, Iterable['DTypeNest'], Mapping[Text, 'DTypeNest'], ]
TensorSpecNest = Union[TensorSpec, Iterable['TensorSpecNest'], Mapping[Text, 'TensorSpecNest'], ]
