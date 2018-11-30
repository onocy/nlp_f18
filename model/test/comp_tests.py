import unittest
import numpy as np
from util.computation import compute_pca, compute_song_vector, compute_artist_vector, lyrics_to_word_matrix


class ComputationTests(unittest.TestCase):

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

    def test_lyrics_to_word_matrix(self):
        sample_vocab = {'a': np.array([1, 2, 3]),
                        'hi': np.array([0, 5, 7]),
                        'bye': np.array([3, 3, 3])}
        lyrics = ['a', 'a', 'hi', 'bye', 'bye']

        M = lyrics_to_word_matrix(lyrics, sample_vocab)
        self.assertTrue(np.array_equal(M, np.array([sample_vocab[w] for w in lyrics])))
