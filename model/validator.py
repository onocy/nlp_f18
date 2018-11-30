from util.computation import *
import random

def validate(artist_embeddings, artist_dict, vocab):
    """
    Validate the learned vector artist embeddings by choosing random artists from 
    the dataset and validating that each artists' song produces the correct artist vector.   
    Args:
        E  (dict[str, np.ndarray]): A dictionary mapping artist names to the artist embedding
        av (np.ndarray): A vector representing an arbitrary artist
        k : number of nearest neighbors to return  
    """
    artists = list(artist_dict.keys())
    # change second argument value to desired # of artists to test
    # there are 643 total artists in the dataset
    test_artists = random.sample(artists, 643)

    print("sanity validation {}%".format(sanity_validate(artist_embeddings, artist_dict, vocab, test_artists)[0]))
    print("bow validation {}%".format(bow_validate(artist_embeddings, artist_dict, vocab, test_artists)[0]))


def sanity_validate(artist_embeddings, artist_dict, vocab, test_artists):
    """
    Validate the learned vector artist embeddings by checking if an artist vector generated
    from a single song is most similar (based on Cosine distance) to that of the artist's learned 
    embedding during the learning phase.   
    Args:
        artist_embeddings dict[str, np.ndarray]: A dictionary mapping artist names to the artist embedding
        artist_dict dict[str, dict[str], list[str]]]: A dictionary mapping artist names to a dictionary 
            mapping song names to a list of words in each song
        vocab dict[str, np.ndarray]: A mapping from words to their embedding vectors
    Returns:
        float: percent accuracy
    """
    num_correct = 0
    num_total_data_points = 0

    for artist in test_artists:
        num_total_data_points += 1

        song = random.sample(list(artist_dict[artist]), 1)[0]
        W = lyrics_to_word_matrix(artist_dict[artist][song], vocab)
        sv = compute_song_vector(W)
        S  = [sv]
        av = compute_artist_vector(np.array(S))

        if artist in [neigbour[0] for neigbour in nearest_neighbors(artist_embeddings, av, 50)]:
            num_correct += 1
    return ((num_correct/num_total_data_points)*100, num_total_data_points)


def bow_validate(artist_embeddings, artist_dict, vocab, test_artists):
    """
    Validate the learned vector artist embeddings by checking if an artist vector generated from a 
    random subset of words from an artist's lyrics bank is most similar (based on Cosine distance) 
    to that of the artist's learned embedding during the learning phase.   
    Args:
        artist_embeddings dict[str, np.ndarray]: A dictionary mapping artist names to the artist embedding
        artist_dict dict[str, dict[str], list[str]]]: A dictionary mapping artist names to a dictionary 
            mapping song names to a list of words in each song
        vocab dict[str, np.ndarray]: A mapping from words to their embedding vectors
    Returns:
        float: percent accuracy
    """
    num_correct = 0
    num_total_data_points = 0

    for artist in test_artists:
        num_total_data_points += 1

        lyric_bow = set()
        for song in artist_dict[artist]:
            lyric_bow.update(artist_dict[artist][song])

        random_lyric_words = random.sample(lyric_bow, 30)
        
        W = lyrics_to_word_matrix(random_lyric_words, vocab)
        sv = compute_song_vector(W)
        S  = [sv]
        av = compute_artist_vector(np.array(S))

        if artist in [neigbour[0] for neigbour in nearest_neighbors(artist_embeddings, av, 50)]:
            num_correct += 1
    return ((num_correct/num_total_data_points)*100, num_total_data_points)


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