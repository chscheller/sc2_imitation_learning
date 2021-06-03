from unittest import TestCase

from sc2_imitation_learning.dataset.dataset import EpisodeIterator, EpisodeSlice


class EpisodeIteratorTest(TestCase):

    def test_iterate(self):
        it = EpisodeIterator(episode_id=1, episode_path='test', episode_length=3, sequence_length=2)
        slices = [next(it) for _ in range(4)]
        self.assertTrue(slices == [
            EpisodeSlice(episode_id=1, episode_path='test', start=0, length=2, wrap_at_end=True),
            EpisodeSlice(episode_id=1, episode_path='test', start=2, length=2, wrap_at_end=True),
            EpisodeSlice(episode_id=1, episode_path='test', start=1, length=2, wrap_at_end=True),
            EpisodeSlice(episode_id=1, episode_path='test', start=0, length=2, wrap_at_end=True),
        ])
