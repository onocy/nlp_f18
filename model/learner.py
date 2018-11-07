from util.computation import *
from util.io_helper import *
from validator import validate


class Learner(object):

    def __init__(self, csv_file, word_embeddings=None):
        self.artist_dict = tokenize_csv(csv_file, 0, 1, 3)
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


if __name__ == '__main__':
    from plotter import plot_coords_with_labels

    learner = Learner('../songdata.csv')
    artists = learner.learn_artists()
    pickle_object(learner, 'learner.pickle')

    # 2-D orthogonal basis
    B = np.eye(50)[:, :2].T
    artist_projections = {artist : project_onto_subspace(v, B) for artist, v in artists.items()}
    print(artist_projections)

    artists = ['Kanye West', 'Eminem', 'Drake', 'Lil Wayne', 'Chris Brown', 'Flo-Rida', 'J Cole', 'Mc Hammer',
                'Migos', 'Ne-Yo', 'Nicki Minaj', 'Pitbull', 'Snoop Dogg', 'The Weeknd']
    coord_dict = {artist : artist_projections[artist] for artist in artists}
    plot_coords_with_labels(coord_dict)

    validation_score = validate(artists, learner.artist_dict, learner.vocab)
