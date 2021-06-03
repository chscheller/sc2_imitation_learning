import tensorflow as tf
import numpy as np

from sc2_imitation_learning.behaviour_cloning.learner import compute_correct_predictions, compute_neg_log_probs


class Test(tf.test.TestCase):
    def test_compute_correct_predictions(self):
        targets = np.asarray([0, -1, 1, -1])
        predictions = np.asarray([0, -1, 0, 0])
        correct_predictions, total_predictions = compute_correct_predictions(targets, predictions)
        self.assertEqual(correct_predictions, 1)
        self.assertEqual(total_predictions, 2)

    def test_compute_neg_log_probs(self):
        # test without masked labels
        labels = np.asarray([0, 1])
        logits = np.asarray([[0.5, 1.5], [-1.0, 2.0]])
        log_probs = tf.math.log_softmax(logits, axis=-1)
        label_mask_value = -1
        neg_log_probs = compute_neg_log_probs(labels, logits, label_mask_value)

        self.assertAllClose(neg_log_probs, [-log_probs[0, labels[0]], -log_probs[1, labels[1]]])

        # test with masked labels
        labels = np.asarray([0, -1])
        logits = np.asarray([[0.5, 1.5], [-1.0, 2.0]])
        log_probs = tf.math.log_softmax(logits, axis=-1)
        label_mask_value = -1
        neg_log_probs = compute_neg_log_probs(labels, logits, label_mask_value)

        self.assertAllClose(neg_log_probs, [-log_probs[0, labels[0]], 0.])
