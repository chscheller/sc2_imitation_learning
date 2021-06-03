import glob
import os
import random
import tempfile
import time
from unittest import TestCase
import tensorflow as tf

from sc2_imitation_learning.common.progress_logger import ConsoleProgressLogger, TensorboardProgressLogger


class Test(TestCase):
    def test_tensorboard_progress_logger(self):
        final_step = 100

        with tempfile.TemporaryDirectory() as log_dir:
            summary_writer = tf.summary.create_file_writer(log_dir)

            initial_size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(log_dir, '*')) if os.path.isfile(f))

            progress_logger = TensorboardProgressLogger(
                 summary_writer=summary_writer,
                 logging_interval=1.)
            progress_logger.start()

            for i in range(final_step):
                progress_logger.log_dict({
                    'loss/loss': 10 * (final_step-i) / final_step,
                    'samples_per_second': 5.0 + 10.0 * random.random(),
                    'learning_rate': 1e-4
                }, tf.constant(i, dtype=tf.int32))
                time.sleep(0.1 * random.random())

            progress_logger.shutdown()

            final_size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(log_dir, '*')) if os.path.isfile(f))

            self.assertGreater(final_size, initial_size)

    def test_console_progress_logger(self):
        final_step = 200

        progress_logger = ConsoleProgressLogger(
            final_step=final_step,
            batch_samples=10,
            logging_interval=1.)
        progress_logger.start()

        for i in range(final_step):
            progress_logger.log_dict({
                'loss/loss': 10 * (final_step-i) / final_step,
                'samples_per_second': 5.0 + 10.0 * random.random(),
                'learning_rate': 1e-4
            }, i)
            time.sleep(0.1 * random.random())

        progress_logger.shutdown()
