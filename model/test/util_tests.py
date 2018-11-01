import unittest
import numpy as np
from util import compute_pca, compute_song_vector, compute_artist_vector, tokenize_csv


class UtilTests(unittest.TestCase):

    def test_compute_pca(self):
        # Test on a 4 x 5 matrix
        shape = (4, 5)
        A = np.random.randn(*shape)

        # Should work for all values <= min(shape)
        for dim in range(1, 1 + min(shape)):
            self.assertTrue(compute_pca(A, dim).shape == (4, dim))

        # Should throw an error for any higher value
        with self.assertRaises(ValueError):
            compute_pca(A, 5)
            compute_pca(A, 6)

    def test_compute_song_vector(self):
        # Compute song vector with standard mean function
        W = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

        self.assertTrue(np.array_equal(compute_song_vector(W), np.mean(W, axis=0)))

    def test_compute_artist_vector(self):
        # Compute artist vector with standard mean function
        S = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

        self.assertTrue(np.array_equal(compute_artist_vector(S), np.mean(S, axis=0)))

    def test_tokenize_csv(self):
        artist_dict = tokenize_csv('test/song_test.csv', artist_col=0, lyric_col=2)
        self.assertTrue(artist_dict['artist1'], ['hey', 'this', 'is', 'text'])
        self.assertTrue(artist_dict['artist2'], ['more', 'words', 'and', 'lyrics', 'hello', 'world'])
