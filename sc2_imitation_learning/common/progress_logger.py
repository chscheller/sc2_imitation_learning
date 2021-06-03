import datetime
import threading
import timeit
from abc import abstractmethod
from collections import defaultdict
from typing import Optional

import tensorflow as tf
import numpy as np

from sc2_imitation_learning.common.utils import flatten_nested_dicts


class ProgressLogger(object):
    def __init__(self,
                 logging_interval: float = 10.,
                 initial_step: int = 0) -> None:
        super().__init__()
        self._logging_interval = logging_interval
        self._initial_step = initial_step
        self._step = self._initial_step
        self._lock = threading.Lock()
        self._terminated = threading.Event()
        self._logger_thread = threading.Thread(target=self._run)
        self._logs = defaultdict(list)

    def start(self):
        self._logger_thread.start()

    def shutdown(self, block: bool = True):
        assert self._logger_thread.is_alive()
        self._terminated.set()
        if block:
            self._logger_thread.join()

    def log_dict(self, values: dict, step: int):
        assert step >= self._step
        flattened = flatten_nested_dicts(values)
        with self._lock:
            for key, value in flattened.items():
                self._logs[key].append(np.copy(value))
            self._step = step

    @abstractmethod
    def _log(self, values: dict, step: int):
        pass

    def _run(self):
        last_log, last_step = timeit.default_timer(), self._initial_step
        while not self._terminated.isSet():
            with self._lock:
                # only lock critical section
                if self._step != last_step:
                    logs = {k: np.mean(v) for k, v in self._logs.items() if len(v) > 0}
                    step = self._step
                    self._logs = defaultdict(list)
                else:
                    logs = None
            if logs is not None:
                self._log(logs, step)
                last_step = step
            now = timeit.default_timer()
            elapsed = now - last_log
            self._terminated.wait(max(0., self._logging_interval - elapsed))
            last_log = timeit.default_timer()


class TensorboardProgressLogger(ProgressLogger):
    def __init__(self,
                 summary_writer: tf.summary.SummaryWriter,
                 logging_interval: float = 10.,
                 initial_step: int = 0) -> None:
        super().__init__(logging_interval, initial_step)
        self._summary_writer = summary_writer

    def _log(self, values: dict, step: int):
        with self._summary_writer.as_default():
            for key, value in values.items():
                tf.summary.scalar(key, value, step=np.int64(step))


class ConsoleProgressLogger(ProgressLogger):
    def __init__(self,
                 final_step: int,
                 batch_samples: int,
                 logging_interval: float = 10.,
                 initial_step: int = 0,
                 start_time: Optional[float] = None) -> None:
        super().__init__(logging_interval, initial_step)
        self._final_step = final_step
        self._batch_samples = batch_samples
        self._start_time = timeit.default_timer() if start_time is None else start_time
        self._last_log_time = timeit.default_timer()

    def _log(self, values: dict, step: int):
        print(f"Train | "
              f"step={step} | "
              f"samples={self._batch_samples * step} | "
              f"progress={round(100 * step / float(self._final_step), 1):5.1f}% | "
              f"time={datetime.timedelta(seconds=round(timeit.default_timer() - self._start_time))} | "
              f"loss={values['loss/loss']:.3f} | "
              f"samples/sec={values['samples_per_second']:.2f} | "
              f"lr={values['learning_rate']:.3e}",
              flush=True)
