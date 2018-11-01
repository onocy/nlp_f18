import unittest

from test.util_tests import UtilTests


def generate_test_suite():
    suite = unittest.TestSuite()

    # Add Util tests to suite
    add_util_tests(suite)

    return suite


def add_util_tests(suite):
    suite.addTest(UtilTests('test_compute_pca'))
    suite.addTest(UtilTests('test_compute_song_vector'))
    suite.addTest(UtilTests('test_compute_artist_vector'))
    suite.addTest(UtilTests('test_tokenize_csv'))

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(generate_test_suite())
