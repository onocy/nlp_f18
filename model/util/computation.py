"""Computation utility functions to manipulating word, song and artist vectors
"""

import numpy as np
import torch


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


def lyrics_to_vocab_idx(lyrics, vocab_index):
    """
    Converts a string of song lyrics to a matrix whose rows are word vocab indices
    Args:
        lyrics (list[str]): A list of strings representing a song
        vocab_index (dict[str: np.ndarray]): A dictionary mapping words to their unique indices
    Returns:
        torch.tensor: A (len(lyrics)) tensor with an index entry for each word in the string
    """

    input_vector = [vocab_index.get(word, -1) for word in lyrics]
    return torch.tensor(input_vector, dtype=torch.long)


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


def create_genre_index(genre_dict):
    """
    Creates a dictionary mapping genres to a unique integer index
    Args:
        genre_dict (dict[dict[str np.ndarray]]): A dictionary mapping genre names to dictionaries of artist names, to
          dictionaries of
    Returns:
        dict[str, int]: A dictionary mapping genre titles to a unique integer
    """

    sorted_genres = sorted(genre_dict.keys())
    return {genre : i for i, genre in enumerate(sorted_genres)}


def create_artist_index(artist_dict):
    """
    Creates a dictionary mapping artist names to a unique integer index
    Args:
        artist_dict (dict[dict[str np.ndarray]]): A dictionary mapping artist names to dictionaries of song names,
          which map song names to a matrix of word embeddings
    Returns:
        dict[str, int]: A dictionary mapping artist names to a unique integer
    """

    sorted_artists = sorted(artist_dict.keys())
    return {artist : i for i, artist in enumerate(sorted_artists)}


def create_vocab_index(artist_dict):
    """
    Creates a vocabulary mapping each word to a unique integer
    Args:
        artist_dict (dict[dict[str, list[str]]]): A dictionary mapping artist names to a dictionary mapping song names
          to a list of strings corresponding to lyrics to a particular
    Returns:
        dict[str, int]: A dictionary mapping every distinct word that appears in artist_dict to a unique integer
    """

    vocab_index = {'<PAD>': 0}
    wordset = {word for artist in artist_dict for song in artist_dict[artist] for word in artist_dict[artist][song]}

    # Sort our distinct set of words so the function produces a deterministic ordering
    distinct_sorted = sorted(list(wordset))

    for i, word in enumerate(distinct_sorted):
        vocab_index[word] = i + 1

    return vocab_index


def create_vocab_index_from_genre(genre_dict):
    """
    Creates a vocabulary mapping each word to a unique integer
    Args:
        genre_dict (dict[dict[str, list[str]]]): A dictionary mapping genres to dictionaries mapping artist names to
          dictionaries mapping song names to a list of strings corresponding to lyrics to a particular
    Returns:
        dict[str, int]: A dictionary mapping every distinct word that appears in artist_dict to a unique integer
    """

    vocab_index = {'<PAD>': 0}
    wordset = {word for genre in genre_dict for artist in genre_dict[genre] for song in genre_dict[genre][artist]
                for word in genre_dict[genre][artist][song] }

    # Sort our distinct set of words so the function produces a deterministic ordering
    distinct_sorted = sorted(list(wordset))

    for i, word in enumerate(distinct_sorted):
        vocab_index[word] = i + 1

    return vocab_index


def build_input_data(artist_dict, vocab_index, artist_indices):
    """
    Creates an input list where each element represents a song, and is a 2-tuple of the format (artist, np.ndarray).
    The first element in each tuple is an integer representing a unique artist, and the second element is a matrix
    with a row for each word embedding in a given song

    Args:
        artist_dict: (dict[dict[str, np.ndarray]]): A dictionary mapping artist names to dictionaries of song names,
          which map song names to a matrix of word embeddings
        vocab_index (dict[str: np.ndarray]): A dictionary mapping words to their unique word indices
        artist_indices (dict[str, int]): A dictionary mapping artist names to a unique integer
    Returns
        list((int, torch.tensor)): A list of tuples where the first element is an integer representing an artist, and
          the second element is a matrix of word embeddings for a particular song by that artist
    """

    input_set = []
    for artist in artist_dict:
        for song in artist_dict[artist]:
            lyric_indices = lyrics_to_vocab_idx(artist_dict[artist][song], vocab_index)
            input_set.append((artist_indices[artist], lyric_indices))

    return input_set


def build_genre_input_data(genre_dict, vocab_index, genre_indices):
    """
    Creates an input list where each element represents a song, and is a 2-tuple of the format (genre, np.ndarray).
    The first element in each tuple is an integer representing a unique genre, and the second element is a matrix
    with a row for each word embedding in a given song

    Args:
        genre_dict: (dict[str, dict[str, dict[str, torch.tensor]]]): A dictionary mapping music genres to dictionaries
          mapping artist names to dictionaries mapping song names to lists of word indices
        vocab_index (dict[str: np.ndarray]): A dictionary mapping words to their unique word indices
        genre_indices (dict[str, int]): A dictionary mapping genre names to a unique integer
    Returns
        list((int, torch.tensor)): A list of tuples where the first element is an integer representing a genre, and
          the second element is a matrix of word embeddings for a particular song in that genre
    """

    songs_per_genre = {genre : 0 for genre in genre_dict.keys()}

    input_set = []
    for genre in genre_dict:
        for artist in genre_dict[genre]:
            for song in genre_dict[genre][artist]:
                if songs_per_genre[genre] < 2000:
                    lyric_indices = lyrics_to_vocab_idx(genre_dict[genre][artist][song], vocab_index)
                    input_set.append((genre_indices[genre], lyric_indices))
                    songs_per_genre[genre] += 1

    return input_set


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


def build_distance_matrix(artists, artist_index):
    """
    Builds a Euclidean Distance Matrix (EDM) from the unpickled `rnn.pickle` network object and artists
    Returns:
        np.ndarray: A square matrix A with entry (i,j) containing the Euclidean distance from artist i to artist j
    """
    from util.io_helper import unpickle_object
    from rnn import ArtistLSTM

    n = len(artists)
    net = unpickle_object('rnn.pickle')
    p = [np.array(x.detach()) for x in net.out.weight]
    EDM = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                a1, a2 = artist_index[artists[i]], artist_index[artists[j]]
                dist = np.linalg.norm(p[a1] - p[a2])
                EDM[i][j] = dist
            else:
                EDM[i][j] = float('inf')
    return EDM

def nearest_neighbors(E, av, k):
    """
    Finds the nearest neighbor the given artist vector in a set of learned artist vectors
    Args:
        E  (dict[str, np.ndarray]): A dictionary mapping artist names to the artist embedding
        av (np.ndarray): A vector representing an arbitrary artist
        k : number of nearest neighbors to return
    Returns:
        list: A list of tuples containing the k closest artist vector in E and its similarity metric
    """
    import scipy as sp

    similarities = map(lambda x: (x[0], sp.spatial.distance.cosine(x[1], av)), E.items())
    sorted_similarities = sorted(similarities, key=lambda x: x[1])
    return sorted_similarities[:k]
