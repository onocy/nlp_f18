import numpy as np
from sklearn import decomposition
from collections import Counter


def compute_pca(A, num_dimensions):
    """
    Computes the Principal Component Analysis over the given matrix A with the specified dimensionality
    Args:
        A (np.ndarray): A matrix
        num_dimensions (int): An integer denoting the number of dimensions the rows of A will be reduced to
    Returns:
        np.ndarray: A matrix of size (len(A) x num_dimensions) containing the uncorrelated principal
          components of A
    """
    pca = decomposition.PCA(n_components=num_dimensions)
    return pca.fit_transform(A)


def compute_song_vector(W, reduce_to=None, f=lambda v: np.mean(v, axis=0)):
    """
    Computes a song vector given a matrix of words and a mapping function
    Args:
        W (np.ndarray): A matrix whose rows are word embeddings
    Optional:
        reduce_to (int): An integer denoting the number of components the resulting song vector should have. If not
          specified, no dimensionality reduction is performed
        f (np.ndarray -> np.ndarray): A mapping function from a word matrix to a song vector. By default, it
          it maps each matrix to the mean row vector
    Returns:
        np.ndarray: A vector embedding representation of a song
    """

    if reduce_to:
        return f(compute_pca(W, num_dimensions=reduce_to))
    return f(W)


def compute_artist_vector(S, reduce_to=None, f=lambda v: np.mean(v, axis=0)):
    """
    Computes an artist vector given a matrix of songs and a mapping function
    Args:
        S (np.ndarray): A matrix whose rows are song embeddings
    Optional:
        reduce_to (int): An integer denoting the number of components the resulting artist vector should have. If not
          specified, no dimensionality reduction is performed
        f (np.ndarray -> np.ndarray): A mapping function from a song matrix to an artist vector. By default, it
          it maps each matrix to the mean row vector
    Returns:
        np.ndarray: A vector embedding representation of an artist
    """

    if reduce_to:
        return f(compute_pca(S, num_dimensions=reduce_to))
    return f(S)


def tokenize_csv(f, artist_col, lyric_col):
    """
    Tokenizes a csv file containing (artist -> lyric) mappings
    Args:
        f (str): The name of the csv file to be tokenized
        artist_col (int): A 0-indexed integer denoting the column in the csv corresponding to artist names
        lyric_col (int): A 0-indexed integer denoting the column in the csv corresponding to lyric strings
    Returns:
        dict[str, list[str]]: A dictionary mapping artist names to a list containing all of their song lyrics
    """

    import nltk
    import csv
    import re

    # NLTK complains and fails to tokenize without this. May be different on your machine.
    nltk.download('punkt')

    artist_lyrics = {}
    with open(f) as file:
        file_csv = csv.reader(file, quotechar='"')

        for row in file_csv:
            artist = row[artist_col]
            lyrics = nltk.word_tokenize(re.sub('\'|,|\(|\)|\?|\!', '', row[lyric_col].lower()))
            if artist not in artist_lyrics:
                artist_lyrics[artist] = []
            artist_lyrics[artist] += lyrics

    return artist_lyrics
