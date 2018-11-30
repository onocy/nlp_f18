import unittest
import numpy as np
from util.io_helper import tokenize_csv, load_word_embeddings


class FileTests(unittest.TestCase):

    def test_tokenize_csv(self):
        artist_dict = tokenize_csv('test/test_songs.csv', artist_col=0, song_col=1, lyric_col=2)
        self.assertTrue(artist_dict['artist1']['song1'], ['hey', 'this', 'is', 'text'])
        self.assertTrue(artist_dict['artist2']['song2'], ['more', 'words', 'and', 'lyrics'])
        self.assertTrue(artist_dict['artist2']['song3'], ['hello', 'world'])

    def test_load_word_embeddings(self):
        vocab = load_word_embeddings('test/test_embeddings.txt')
        self.assertEquals(len(vocab), 3)

        # Embeddings should be case-insensitive
        self.assertTrue(np.array_equal(vocab['world'], np.array([3, 4, 5, 6, 7], dtype='float32')))

        # Words not in vocab should throw a KeyError
        with self.assertRaises(KeyError):
            print(vocab['badword'])

        # Our function should strip newline characters
        self.assertTrue(len(vocab['language']), 3)
        self.assertTrue(np.array_equal(vocab['language'], np.array([8, 4, 1, 2, 3], dtype='float32')))