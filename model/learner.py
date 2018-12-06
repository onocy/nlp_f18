from util.computation import *
from util.io_helper import *
import random

class Learner(object):

    def __init__(self, artist_dict, word_embeddings=None):
        self.artist_dict = artist_dict
        if word_embeddings is not None:
            self.vocab = word_embeddings
        else:
            self.vocab = load_word_embeddings('../glove.6B.50d.txt')

    def learn_artists(self):
        """
        Learns the vector embeddings for all artists in the artist dictionary
        Returns:
            dict[str, np.ndarray]: A dictionary mapping artist names to a
              (len(artist_dict) x embedding_size) matrix with an artist embedding for each row
        """

        embeddings = {}
        for artist in self.artist_dict:
            # Song matrix for the current artist
            S = []

            for song in self.artist_dict[artist]:
                W = lyrics_to_word_matrix(self.artist_dict[artist][song], self.vocab)
                sv = compute_song_vector(W)
                S.append(sv)

            av = compute_artist_vector(np.array(S))
            embeddings[artist] = av

        return embeddings
    
    def test(self, test_data, artist_embeddings): 
        """
        Tests the Learner on the given test data
        Args:
            test_data (list[(int, np.ndarray)]): A list of test tuples of the form (artist_idx, word_embeddings)
            artist_embeddings dict[str, np.ndarray]: A dictionary of learned artist embeddings.
        """

        num_correct = 0
        for artist, song, lyrics in test_data:
            W = lyrics_to_word_matrix(lyrics, self.vocab)
            sv = compute_song_vector(W)
            S  = [sv]
            av = compute_artist_vector(np.array(S))

            if artist in [neigbour[0] for neigbour in nearest_neighbors(artist_embeddings, av, 50)]:
                num_correct += 1

        print("Number correct: {}".format(num_correct))
        print("Number total: {}".format(len(test_data)))
        print("Learner accuracy {}%".format((num_correct/len(test_data))*100))


def split_train_test(artist_dict):
    """
    Splits artist_dict into 80-20 train test data.
    Args:
        dict[str, np.ndarray]: A dictionary mapping artist names to a
              (len(artist_dict) x embedding_size) matrix with an artist embedding for each row
    Returns
        training_artist_dict: artist_dict with onlt the training data
        test_data: List of tuples (artist, song, lyrics) with only the test data
    """

    # flatten artist_dict to [(artist, title, lyrics)]
    flattened_artists_dict = []
    for artist in artist_dict:
        for song in artist_dict[artist]:
            flattened_artists_dict.append((artist, song, artist_dict[artist][song]))

    # train test 80 20 split
    random.seed(69)
    random.shuffle(flattened_artists_dict)
    training_length = int(0.8 * len(flattened_artists_dict))
    
    training_data = flattened_artists_dict[:training_length]
    test_data = flattened_artists_dict[training_length:]

    # rebuild training artist_dict
    training_artist_dict = {}
    for artist, song, lyrics in training_data:
        if artist not in training_artist_dict:
            training_artist_dict[artist] = {}
        training_artist_dict[artist][song] = lyrics
        
    return training_artist_dict, test_data


if __name__ == '__main__':
    from plotter import plot_coords_with_labels
    import sys
    import os.path

    if os.path.isfile('artists.pickle'):
        artist_dict = unpickle_object('artists.pickle')
        print("Unpickled the artist dictionary.")
    else:
        artist_dict = tokenize_csv('../songdata.csv', 0, 1, 3)

    training_artist_dict, test_data = split_train_test(artist_dict)

    if os.path.isfile('learner.pickle'):
        print('Reading from pickle file...')
        learner = unpickle_object('learner.pickle')
    else:
        learner = Learner(training_artist_dict)
        
    artists = learner.learn_artists()      
    pickle_object(learner, 'learner.pickle')
    learner.test(test_data, artists)

    # # 2-D orthogonal basis
    # B = np.eye(50)[:, :2].T
    # artist_projections = {artist : project_onto_subspace(v, B) for artist, v in artists.items()}
    # # print(artist_projections)

    # artists = ['Kanye West', 'Eminem', 'Drake', 'Lil Wayne', 'Chris Brown', 'Flo-Rida', 'J Cole', 'Mc Hammer',
    #             'Migos', 'Ne-Yo', 'Nicki Minaj', 'Pitbull', 'Snoop Dogg', 'The Weeknd']
    # coord_dict = {artist : artist_projections[artist] for artist in artists}
    # plot_coords_with_labels(coord_dict)
