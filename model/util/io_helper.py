"""File I/O utilities for tokenizing CSV data and loading word embeddings.
"""

import numpy as np


def tokenize_string(s, regex=None):
    """
    Tokenizes a string into a list of words
    Args:
        s (str): The string to be tokenized
    Optional:
        regex (re.Pattern): A compiled regex pattern that specifies which characters in the string to remove
    Returns:
        list[str]: A list of tokenized words
    """

    import nltk # Remember to have the `punkt` package downloaded
    import re

    if regex:
        s = re.sub(regex, '', s)
    return nltk.word_tokenize(s)


def tokenize_csv(f, artist_col, song_col, lyric_col):
    """
    Tokenizes a csv file containing (artist -> lyric) mappings
    Args:
        f (str): The name of the csv file to be tokenized
        artist_col (int): A 0-indexed integer denoting the column in the csv corresponding to artist names
        song_col (int): A 0-indexed integer denoting the column in the csv corresponding to song names
        lyric_col (int): A 0-indexed integer denoting the column in the csv corresponding to lyric strings
    Returns:
        dict[str, dict[str], list[str]]]: A dictionary mapping artist names to a dictionary mapping song names to a list
          of words in each song
    """

    import csv
    import re

    artists = {}
    with open(f) as file:
        file_csv = csv.reader(file, quotechar='"')

        # Skip the header row
        next(file_csv)

        for row in file_csv:
            artist = row[artist_col]
            song = row[song_col]
            lyrics = tokenize_string(row[lyric_col].lower(), regex=re.compile('\'|,|\(|\)|\?|\!'))
            if artist not in artists:
                artists[artist] = {}
            artists[artist][song] = lyrics

    return artists

def tokenize_csv_pandas(f):
    import pandas as pd
    import nltk
    nltk.download('punkt')
    p = pd.read_csv("../lyrics.csv")
    d = {}
    for _, val in p.iterrows():
        artist = val["artist"]
        if artist in d:
            d[artist].append(tokenize(clean(val["lyrics"])))
        else:
            d[artist] = [tokenize(clean(val["lyrics"]))]

def tokenize(s):
    import nltk
    import re
    return nltk.word_tokenize(re.sub('[^\w\s]+', '', s))


def clean(s):
    s = str(s)
    s = s.lower()
    s = s.replace('\n',' ')
    s = s.replace(',',' ')
    return s


def load_word_embeddings(f, unzip=False):
    """
    Loads word embeddings from a specified file
    Args:
        f (str): The path to a file containing word embeddings
    Optional:
        unzip (bool): True if the file specified is a .zip, False by default
    Returns:
        dict[str, np.ndarray]: A mapping from words to their embedding vectors
    """
    if unzip:
        import zipfile
        vec_file = zipfile.ZipFile(f, 'r').open(f[:-4], 'r')
    else:
        vec_file = open(f, 'r')

    vocab = {}
    for line in vec_file:
        split = line.strip().split()
        word = split[0].lower()
        if word not in vocab:
            vocab[word] = np.zeros((1, len(split) - 1))
        vocab[word] = np.array(split[1:], dtype='float32')

    return vocab


def pickle_object(obj, filename):
    """
    Pickles a given object into a file with the given file name
    Args:
        obj (object): A python object
        filename (str): The name of the file to write the object 
    """

    import pickle

    outfile = open(filename, 'wb')
    pickle.dump(obj, outfile)


def unpickle_object(filename):
    """
    Unpickles a file into an object with the given filename
    Args: 
        filename (str): The name of the file read that is loaded back into an object.
    """
    import pickle 

    infile = open(filename, 'rb')
    return pickle.load(infile)
