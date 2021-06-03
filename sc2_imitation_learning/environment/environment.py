import collections
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

import tree

StepOutput = collections.namedtuple('StepOutput', ['reward', 'info', 'done', 'observation'])


class Space(ABC):
    @property
    @abstractmethod
    def specs(self) -> Dict:
        pass

    def dtypes(self) -> Dict:
        return tree.map_structure(lambda s: s.dtype, self.specs)

    def shapes(self, as_tensor_shapes: bool = False) -> Dict:
        if as_tensor_shapes:
            import tensorflow as tf  # load tensorflow lazily
            return tree.map_structure(lambda s: tf.TensorShape(list(s.shape)), self.specs)
        else:
            return tree.map_structure(lambda s: s.shape, self.specs)


class ActionSpace(Space):
    @abstractmethod
    def no_op(self) -> Dict:
        pass

    @abstractmethod
    def transform(self, action: Dict) -> Tuple[Any, int]:
        pass

    @abstractmethod
    def transform_back(self, action: Any, step_mul: int) -> Dict:
        pass


class ObservationSpace(Space):
    @abstractmethod
    def transform(self, observation: Dict) -> Dict:
        pass

    @abstractmethod
    def transform_back(self, observation: Dict) -> Dict:
        pass


class EnvMeta(ABC):
    @abstractmethod
    def launch(self) -> None:
        pass

    @property
    @abstractmethod
    def level_name(self) -> str:
        pass

    @property
    @abstractmethod
    def action_space(self) -> ActionSpace:
        pass

    @property
    @abstractmethod
    def observation_space(self) -> ObservationSpace:
        pass

