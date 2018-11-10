import torch
from torch import nn
import torch.nn.functional as F

from util.io_helper import *
from util.computation import *

import random


class ArtistLSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, output_size, vocab_size):
        """
        Initializes our ArtistNet LSTM network
        Params:
            embedding_dim (int): The number of dimensions in each of the word embedding vectors
            output_size (int): The number of distinct artists that the network will run a prediction over
            vocab_size (int): The number of distinct words in our vocabulary
        """
        super(ArtistLSTM, self).__init__()

        self.embedding_dim = embedding_dim

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

    def __input_to_vector(self, s):
        """
        Converts an input string of lyrics s into an input vector to the neural network
        Args:
            s (str): A string of lyrics
        Returns:
            torch.Tensor: A tensor that is an input formation of the string s
        """
        import re

        input_tensor = torch.zeros(self.embedding_dim, dtype=torch.float)
        word_list = tokenize_string(s, regex=re.compile('\'|,|\(|\)|\?|\!'))

        for i, word in enumerate(word_list[:self.d_emb]):
            # If the given word does not exist in the vocabulary index, just map it to -1
            input_tensor[i] = self.vocab.get(word, -1)

        return input_tensor

    def train_network(self, training_data, batch_size, learning_rate=0.001, num_epochs=10, loss_fn=torch.nn.NLLLoss,
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

        MAX_INPUT_LENGTH = 50

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

                # Clear the hidden states from the previous batch
                self.hidden = self.__init_hidden()
                batch = training_data[start:start + batch_size]
                if len(batch) < batch_size:
                    break

                in_mat = torch.zeros(batch_size, MAX_INPUT_LENGTH, dtype=torch.long)
                out_vec = torch.zeros(len(batch), dtype=torch.long)

                for i, (artist, lyrics) in enumerate(batch):
                    out_vec[i] = artist
                    for j, lyric in enumerate(lyrics[:MAX_INPUT_LENGTH]):
                        in_mat[i, j] = lyric

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
        Predicts the artist that made the song containing the corresponding lyrics (using vocabulary indices)
        Args:
            lyrics (str): A string of words representing lyrics to a song
        Returns:
            str: The name of the artist who created the song
        """
        input_tensor = self.__input_to_vector(lyrics)

        # Get the artist with the highest probability
        prediction = torch.argmax(self(input_tensor)).item()
        return self.idx_to_artist[prediction]

    def test_batch(self, test_data):
        corr = 0.
        total = 0.

        for start in range(0, len(test_data), 32):
            batch = test_data[start:start+1]
            in_mat = torch.zeros(len(batch), 50, dtype=torch.long)
            truths = torch.zeros(len(batch), dtype=torch.long)

            for i, (artist, lyrics) in enumerate(batch):
                truths[i] = artist
                for j, lyric in enumerate(lyrics[:50]):
                    in_mat[i, j] = lyric

            preds = net(in_mat)
            max_pred = torch.argmax(preds)
            if max_pred == artist:
                corr += 1
            total += 1

        print(corr, total, corr / total)

    def test(self, test_data):
        """
        Tests the network on the given test data
        Args:
            test_data (list[(int, np.ndarray)]): A list of test tuples of the form (artist_idx, word_embeddings)
        """
        num_correct = 0
        for artist, lyrics in test_data:
            input_vec = torch.zeros(1, self.embedding_dim, dtype=torch.long)
            input_vec[0] = lyrics[:self.embedding_dim]
            prediction = torch.argmax(self(input_vec)).item()
            if prediction == artist:
                num_correct += 1

        print("Number correct: {}".format(num_correct))
        print("Number total: {}".format(len(test_data)))
        print("Test accuracy: {:.2f}%".format(num_correct / len(test_data) * 100))



if __name__ == '__main__':

    # Load our word embeddings and artist dictionary
    # vocab = load_word_embeddings('../glove.6B.50d.txt')
    # artist_dict = tokenize_csv('../songdata.csv', 0, 1, 3)
    # pickle_object(artist_dict, 'artists.pickle')
    artist_dict = unpickle_object('artists.pickle')
    vocab_index = create_vocab_index(artist_dict)

    artist_indices = create_artist_index(artist_dict)

    # Our input data
    input_data = build_input_data(artist_dict, vocab_index, artist_indices)
    print("Built input data.")

    # Shuffle our input data and split it into 20% test, 80% training data
    random.shuffle(input_data)
    training_length = int(0.8 * len(input_data))

    training_data = input_data[:training_length]
    test_data = input_data[training_length:]

    # Create our neural network with 50-count word embeddings
    net = ArtistLSTM(100, 10, len(artist_dict), len(vocab_index))
    # net = unpickle_object('net.pickle')
    print("About to train the network!")
    # net.test(test_data)

    # Training time!
    net.train_network(training_data, batch_size=64, num_epochs=5)
    pickle_object(net, 'net.pickle')
