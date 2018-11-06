"""Computation utility functions to manipulating word, song and artist vectors
"""

import numpy as np
import scipy as sp


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

    from sklearn import decomposition

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


def find_closest_artist(E, av):
    """
    Finds the closest artist to the given artist vector in a set of learned artist vectors
    Args:
        E  (dict[str, np.ndarray]): A dictionary mapping artist names to the artist embedding
        av (np.ndarray): A vector representing an arbitrary artist
    Returns:
        np.ndarray: A vector embedding representation of the closest artist in the embedding dict
    """
    return min(E.items(), key=lambda x: sp.spatial.distance.cosine(x[1], av)) 


def lyrics_to_word_matrix(lyrics, vocab):
    """
    Converts a string of song lyrics to a matrix whose rows are word embedding vectors
    Args:
        lyrics (list[str]): A list of strings representing a song
        vocab (dict[str: np.ndarray]): A dictionary mapping words to their embedding vectors
    Returns:
        np.ndarray: A (len(lyrics) x embedding_size) matrix with a word embedding row for each word in the lyric string
    """

    # We need some way to determine the size of each of the word embeddings. Here we are assuming the vocabulary
    # has an embedding for the word 'a', which seems like a reasonable assumption given the scope of this project
    M = np.zeros((len(lyrics), len(vocab['a'])))

    for i, word in enumerate(lyrics):
        embedding = vocab.get(word)
        if embedding is not None:
            M[i] = embedding

    return M


def project_onto_subspace(v, B):
    """
    Projects the given vector v onto a subspace spanned by orthogonal basis vectors B
    Args:
        v (np.ndarray): A vector
        B (np.ndarray): A matrix containing k orthogonal basis vectors
    Returns:
        np.ndarray: A (n x 1) vector that is the least squares projection of Bx = v
    """

    def project(v, u):
        """Projects a vector v onto a vector u
        """
        return (np.dot(v, u)/np.dot(u, u)) * u

    return sum(project(v, b_i) for b_i in B)[:len(B)]

