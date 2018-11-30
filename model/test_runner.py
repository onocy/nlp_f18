import unittest

from test.comp_tests import ComputationTests
from test.file_tests import FileTests


def generate_test_suite():
    suite = unittest.TestSuite()

    # Add Computation tests to suite
    add_computation_tests(suite)

    # Add File I/O tests to suite
    add_file_io_tests(suite)

    return suite


def add_computation_tests(suite):
    suite.addTest(ComputationTests('test_compute_pca'))
    suite.addTest(ComputationTests('test_compute_song_vector'))
    suite.addTest(ComputationTests('test_compute_artist_vector'))
    suite.addTest(ComputationTests('test_lyrics_to_word_matrix'))


def add_file_io_tests(suite):
    suite.addTest(FileTests('test_tokenize_csv'))
    suite.addTest(FileTests('test_load_word_embeddings'))


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(generate_test_suite())
