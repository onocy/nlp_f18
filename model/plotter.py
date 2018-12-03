import matplotlib.pyplot as plt


def plot_coords_with_labels(coord_dict):
    """
    Plots a list of (x, y) coordinates with their respective label given by coord_dict
    Args:
        coord_dict (dict[str, np.ndarray]): A dictionary mapping labels to (x, y) coordinate points
    """

    for label in coord_dict:
        x, y = coord_dict[label]
        plt.plot(x, y, 'bo')
        plt.text(x * (1 + 0.01), y * (1 + 0.01), label, fontsize=12)

    plt.show()


def create_correlation_plot(EDM, artists):
    """
    Creates and plots a correlation heatmap graph from a given EDM matrix and list of artist labels
    """
    from skbio.stats.distance import DissimilarityMatrix
    import matplotlib.pyplot as plt

    dm = DissimilarityMatrix(EDM, artists)
    fig = dm.plot(cmap='Reds', title='Lyrical Similarity')
    fig.show()
    plt.pause(500)

if __name__ == '__main__':
    from util.io_helper import unpickle_object
    from util.computation import create_artist_index, build_distance_matrix
    from rnn import ArtistLSTM

    # First load the artist dictionary and choose a subset of the artists to compare
    artist_dict = unpickle_object('artists.pickle')
    artist_index = create_artist_index(artist_dict)
    selected_artists = ['Kanye West', 'Michael Jackson', 'Drake', 'Eminem', 'Stevie Wonder', 'Madonna', 'Lil Wayne',
                        'Rihanna', 'Bon Jovi', 'Linkin Park', 'Young Jeezy', 'The Beatles', 'Britney Spears', 'Coldplay']

    # Build the distance matrix from the selected artists and their index mappings
    EDM = build_distance_matrix(selected_artists, artist_index)

    # Plot the correlation heat map!
    create_correlation_plot(EDM, selected_artists)
