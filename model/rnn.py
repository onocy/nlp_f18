import torch
from torch import nn
import torch.nn.functional as F

from util.io_helper import *
from util.computation import *

import random


class ArtistLSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, output_size, vocab_size, max_input_len):
        """
        Initializes our ArtistNet LSTM network
        Params:
            embedding_dim (int): The number of dimensions in each of the word embedding vectors
            hidden_dim (int): The number of dimensions in each hidden unit
            output_size (int): The number of output classes that the network will run a prediction over
            vocab_size (int): The number of distinct words in our vocabulary
            max_input_len (int): The fixed-size of the input layer to the network
        """
        super(ArtistLSTM, self).__init__()

        self.embedding_dim = embedding_dim
        self.max_input_length = max_input_len

        # Setup the dimensions of our hidden states and LSTM model
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # Initialize our output linear layer
        self.out = nn.Linear(hidden_dim, output_size)
        self.hidden = self.__init_hidden()

    def __init_hidden(self):
        """
        Initializes our hidden states with the dimensions specified by the constructor parameters.
        The axes semantics are: (num_layers, minibatch_size, hidden_dim)
        """
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, lyrics):
        """
        Computes a probability distribution over the output classes
        Args:
            lyrics (torch.Tensor): A tensor containing word indices corresponding to lyrics from a song
        Returns:
            torch.Tensor: One or more probability distributions over the output classes
        """

        # Grab the word embeddings for our input lyrics
        embeddings = self.word_embeddings(lyrics)
        avg_embeddings = torch.mean(embeddings, 1)

        # Run the embeddings sequentially through all hidden states of the LSTM
        lstm_out, self.hidden = self.lstm(avg_embeddings.view(len(lyrics), 1, -1), self.hidden)

        # Apply the results of our hidden states to a linear layer, and then create a softmax output distribution
        artist_mat = self.out(lstm_out.view(len(lyrics), -1))
        return F.log_softmax(artist_mat, dim=1)

    def __input_to_vector(self, s, vocab):
        """
        Converts an input string of lyrics s into an input vector to the neural network
        Args:
            s (str): A string of lyrics
        Returns:
            torch.Tensor: A tensor that is an input formation of the string s
        """
        import re

        input_tensor = torch.zeros(1, self.max_input_length, dtype=torch.long)
        word_list = tokenize_string(s, regex=re.compile('\'|,|\(|\)|\?|\!'))

        for i, word in enumerate(word_list[:self.max_input_length]):
            # If the given word does not exist in the vocabulary index, just map it to -1
            input_tensor[0][i] = vocab.get(word, -1)

        return input_tensor

    def train_network(self, training_data, batch_size, learning_rate=0.001, num_epochs=10, loss_fn=torch.nn.NLLLoss,
                      opt_algo=torch.optim.Adam):
        """
        Trains the neural network for the given number of epochs
        Args:
            training_data (list[(int, np.ndarray)]): A list of training tuples of the form (artist_idx, word_embeddings)
            batch_size (int): The number of training samples in each batch
        Optional:
            learning_rate (float): The learning rate. Defaults to 0.001
            num_epochs (int): The number of epochs to train for. If not specified, defaults to 10
            loss_fn (func): The loss function to train the network on. Defaults to torch.nn.NLLLoss
            opt_algo (func): The optimizer algorithm to use. Defaults to torch.optim.Adam
        """

        import time

        # Declare our loss functions and optimizers
        loss = loss_fn()
        optimizer = opt_algo(self.parameters(), lr=learning_rate)

        # Loop over every epoch
        for ep in range(num_epochs):
            ep_loss = 0.
            start_time = time.time()

            # Shuffle the training data before selecting our batches
            random.shuffle(training_data)

            # Loop over each batch
            for start in range(0, len(training_data), batch_size):
                # print("Batch start: {}".format(start))

                # Clear the hidden states from the previous batch
                self.hidden = self.__init_hidden()
                batch = training_data[start:start + batch_size]
                if len(batch) < batch_size:
                    break

                in_mat = torch.zeros(batch_size, self.max_input_length, dtype=torch.long)
                out_vec = torch.zeros(len(batch), dtype=torch.long)

                for i, (artist, lyrics) in enumerate(batch):
                    out_vec[i] = artist
                    for j, lyric in enumerate(lyrics[:self.max_input_length]):
                        in_mat[i, j] = lyric

                preds = self(in_mat)
                batch_loss = loss(preds, out_vec)
                ep_loss += batch_loss

                # Compute the gradients

                optimizer.zero_grad()  # First, clear the gradients from the previous batch
                batch_loss.backward()  # Performs our back-propagation by computing gradients
                optimizer.step()       # Updates our parameters using the calculated gradients

            # Print out our loss and elapsed time at the end of every epoch
            print("Epoch #{}: {} ({} minutes)".format(ep, ep_loss, (time.time() - start_time) / 60))

    def predict(self, lyrics, vocab):
        """
        Predicts the artist that made the song containing the corresponding lyrics (using vocabulary indices)
        Args:
            lyrics (str): A string of words representing lyrics to a song
        Returns:
            str: The name of the artist who created the song
        """
        input_tensor = self.__input_to_vector(lyrics, vocab)

        # Get the artist with the highest probability
        return torch.argmax(self(input_tensor)).item()

    def test(self, test_data, k=1):
        """
        Tests the network on the given test data
        Args:
            test_data (list[(int, np.ndarray)]): A list of test tuples of the form (artist_idx, word_embeddings)
        Optional:
            k (int): The number of nearest neighbors to calculate the prediction results for. By default, k=1
        """
        num_correct = 0
        for artist, lyrics in test_data:

            # Construct the fixed-length input vector and fill it up
            input_vec = torch.zeros(1, self.max_input_length, dtype=torch.long)
            for i in range(min(self.max_input_length, len(lyrics))):
                input_vec[0][i] = lyrics[i]

            # Grab the k-highest prediction indices and check if the actual artist is one of them
            predictions = self(input_vec).detach().numpy().argsort()[0][::-1][:k]
            if artist in predictions:
                num_correct += 1

        print("Number correct: {}".format(num_correct))
        print("Number total: {}".format(len(test_data)))
        print("Test accuracy: {:.2f}%".format(num_correct / len(test_data) * 100))


def run_artist_net(train=True):
    """
    Runs our LSTM RNN on the artist dataset
    Optional:
        train (bool): If True, re-trains the network. If False, only the tests are run. An exception will be raised
          if there is no pickled network object in the current directory
    """


def run_genre_net(train=True):
    """
    Runs our LSTM RNN on the genre dataset
    Optional:
        train (bool): If True, re-trains the network. If False, only the tests are run. An exception will be raised
          if there is no pickled network object in the current directory
    """

    # Import the genre dictionary if it exists, otherwise create it from the lyrics CSV
    if os.path.isfile('genre.pickle'):
        genre_dict = unpickle_object('genre.pickle')
        print("Unpickled the genre dictionary.")
    else:
        genre_dict = tokenize_csv_pandas('../lyrics.csv')
        print("Tokenized the genre dictionary from the csv file.")

    vocab_index = create_vocab_index_from_genre(genre_dict)
    genre_indices = create_genre_index(genre_dict)
    print("Created vocab and genre indices.")

    input_data = build_genre_input_data(genre_dict, vocab_index, genre_indices)

    # Shuffle our input data and split it into 20% test, 80% training data
    random.shuffle(input_data)
    training_length = int(0.8 * len(input_data))

    training_data = input_data[:training_length]
    test_data = input_data[training_length:]

    # If the training argument was set to True, train a newly constructed network using the following hyperparameters
    if train:
        EMBEDDING_DIM = 100
        HIDDEN_DIM = 8
        OUTPUT_SIZE = len(genre_dict)
        VOCAB_SIZE = len(vocab_index)
        MAX_INPUT_LEN = 300

        net = ArtistLSTM(EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_SIZE, VOCAB_SIZE, MAX_INPUT_LEN)

        print("Training network...")
        net.train_network(training_data, batch_size=64, num_epochs=20, learning_rate=0.005)
    else:
        net = unpickle_object('rnn.pickle') # NOTE: This will throw an error if there is no pickle!

    # Test the trained network with k closest predicted neighbors
    print("Testing the network...")
    net.test(test_data, k=2)


def run_artist_net(train=True):
    """
    Runs our LSTM RNN on the artist dataset
    Optional:
        train (bool): If True, re-trains the network. If False, only the tests are run. An exception will be raised
          if there is no pickled network object in the current directory
    """

    # Import the artist dictionary if it exists, otherwise create it from the lyrics CSV
    if os.path.isfile('artists.pickle'):
        artist_dict = unpickle_object('artists.pickle')
        print("Unpickled the artist dictionary.")
    else:
        artist_dict = tokenize_csv('../songdata.csv')
        print("Tokenized the artist dictionary from the csv file.")

    vocab_index = create_vocab_index(artist_dict)
    artist_indices = create_artist_index(artist_dict)
    print("Created vocab and artist indices.")

    input_data = build_input_data(artist_dict, vocab_index, artist_indices)

    # Shuffle our input data and split it into 20% test, 80% training data
    random.shuffle(input_data)
    training_length = int(0.8 * len(input_data))

    training_data = input_data[:training_length]
    test_data = input_data[training_length:]

    # If the training argument was set to True, train a newly constructed network using the following hyperparameters
    if train:
        EMBEDDING_DIM = 100
        HIDDEN_DIM = 8
        OUTPUT_SIZE = len(artist_dict)
        VOCAB_SIZE = len(vocab_index)
        MAX_INPUT_LEN = 300

        net = ArtistLSTM(EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_SIZE, VOCAB_SIZE, MAX_INPUT_LEN)

        print("Training network...")
        net.train_network(training_data, batch_size=64, num_epochs=20, learning_rate=0.005)
    else:
        net = unpickle_object('rnn.pickle') # NOTE: This will throw an error if there is no pickle!

    # Test the trained network with k closest predicted neighbors
    print("Testing the network...")
    net.test(test_data, k=2)


if __name__ == '__main__':
    import os.path

    run_artist_net()
    # run_genre_net()
