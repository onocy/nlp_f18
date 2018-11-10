import torch
from torch import nn

from util.io_helper import *
from util.computation import *

import random


class ArtistNet(nn.Module):

    def __init__(self, d_embedding, artist_classes, vocab):
        """
        Initializes our ArtistNet Neural Network
        Params:
            d_embedding (int): The number of dimensions in each of the word embedding vectors
            artist_classes (dict[str, int]): A dictionary mapping artist names to a unique integer
            vocab (dict[str, np.ndarray]): A dictionary mapping words to their word embeddings
        """
        super(ArtistNet, self).__init__()

        self.d_emb = d_embedding
        self.artist_classes = artist_classes
        self.idx_to_artist = {v : k for k, v in artist_classes.items()}
        self.vocab = vocab

        # Non-linear activation functions for hidden layers
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # Create hidden layers
        self.h1 = nn.Linear(self.d_emb, self.d_emb)
        self.h2 = nn.Linear(self.d_emb, self.d_emb)
        self.h3 = nn.Linear(self.d_emb, self.d_emb)

        # Make linear layer to project onto all of our artist classes
        self.out = nn.Linear(self.d_emb, len(artist_classes))
        self.softmax = nn.LogSoftmax()

    def forward(self, inputs):
        """
        Computes a probability distribution over the output classes
        Args:
            inputs (torch.Tensor): A tensor containing one or more input vectors of dimension `self.d_emb`
        Returns:
            torch.Tensor: One or more probability distributions over the output classes
        """

        h1_out = self.sigmoid(self.h1(inputs))
        h2_out = self.sigmoid(self.h2(h1_out))
        h3_out = self.sigmoid(self.h3(h2_out))

        z = self.out(h3_out)
        return self.softmax(z)

    def train_network(self, training_data, batch_size, learning_rate = 0.05, num_epochs=10, loss_fn=torch.nn.NLLLoss,
                      opt_algo=torch.optim.Adam):
        """
        Trains the neural network for the given number of epochs
        Args:
            training_data (list[(int, np.ndarray)]): A list of training tuples of the form (artist_idx, word_embeddings)
            batch_size (int): The number of training samples in each batch
        Optional:
            learning_rate (float): The learning rate. Defaults to 0.05
            num_epochs (int): The number of epochs to train for. If not specified, defaults to 10
            loss_fn (func): The loss function to train the network on. Defaults to torch.nn.NLLLoss
            opt_algo (func): The optimizer algorithm to use. Defaults to torch.optim.Adam
        """

        # Declare our loss functions and optimizers
        loss = loss_fn()
        optimizer = opt_algo(self.parameters(), lr=learning_rate)

        # Loop over every epoch
        for ep in range(num_epochs):
            ep_loss = 0.

            # Shuffle the training data before selecting out batches
            random.shuffle(training_data)

            # Loop over each batch
            for start in range(0, len(training_data), batch_size):
                batch = training_data[start:start + batch_size]
                if len(batch) < batch_size:
                    break

                in_mat = torch.zeros(batch_size, self.d_emb)
                out_vec = torch.zeros(len(batch), dtype=torch.long)

                for i, (artist, lyrics) in enumerate(batch):
                    song_vector = compute_song_vector(lyrics)
                    out_vec[i] = artist
                    for j, elem in enumerate(song_vector):
                        in_mat[i, j] = elem

                preds = self(in_mat)
                batch_loss = loss(preds, out_vec)
                ep_loss += batch_loss

                # Compute the gradients

                optimizer.zero_grad()  # First, clear the gradients from the previous batch
                batch_loss.backward()  # Performs our back-propagation by computing gradients
                optimizer.step()       # Updates our parameters using the calculated gradients

            # Print out our loss at the end of every epoch
            print("Epoch #{}: {}".format(ep, ep_loss))

    def predict(self, lyrics):
        """
        Predicts the artist that made the song containing the corresponding lyrics
        Args:
            lyrics (str): A string of words representing lyrics to a song
        Returns:
            str: The name of the artist who created the song
        """
        import re

        word_list = tokenize_string(lyrics, regex=re.compile('\'|,|\(|\)|\?|\!'))
        W = lyrics_to_word_matrix(word_list, self.vocab)
        input_tensor = torch.tensor(torch.mean(torch.tensor(W), 0), dtype=torch.float)

        # Get the artist with the highest probability
        prediction = torch.argmax(self(input_tensor)).item()
        return self.idx_to_artist[prediction]

    def test(self, test_data):
        """
        Tests the network on the given test data
        Args:
            test_data (list[(int, np.ndarray)]): A list of test tuples of the form (artist_idx, word_embeddings)
        """

        num_correct = 0
        for artist, lyrics in test_data:
            input_vector = torch.zeros(self.d_emb)
            song_vector = compute_song_vector(lyrics)
            for i in range(len(song_vector)):
                input_vector[i] = song_vector[i]
            prediction = torch.argmax(self(input_vector)).item()
            if prediction == artist:
                num_correct += 1

        print("Test accuracy: {:.2f}%".format(num_correct / len(test_data) * 100))


if __name__ == '__main__':

    # Load our word embeddings and artist dictionary
    vocab = load_word_embeddings('../glove.6B.50d.txt')
    # artist_dict = tokenize_csv('../songdata.csv', 0, 1, 3)
    # pickle_object(artist_dict, 'artists.pickle')
    artist_dict = unpickle_object('artists.pickle')

    artist_indices = create_artist_index(artist_dict)

    # Our input data
    input_data = build_input_data(artist_dict, vocab, artist_indices)
    print("Built input data.")

    # Shuffle our input data and split it into 20% test, 80% training data
    random.shuffle(input_data)
    training_length = int(0.8 * len(input_data))

    training_data = input_data[:training_length]
    test_data = input_data[training_length:]

    # Create our neural network
    net = ArtistNet(50, artist_indices, vocab)

    print("About to train the network!")

    # Training time!
    net.train_network(training_data, batch_size=16, num_epochs=10)
    pickle_object(net, 'net.pickle')
