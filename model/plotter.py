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
