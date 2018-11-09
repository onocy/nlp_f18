import torch
from torch import nn

from util.io_helper import load_word_embeddings, tokenize_csv
from util.computation import *


class ArtistNet(nn.Module):
    # define the model parameters
    def __init__(self, d_embedding, class_mapping, vocab):
        super(ArtistNet, self).__init__()

        self.d_emb = d_embedding
        self.num_classes = len(class_mapping)
        self.len_vocab = len(vocab)

        # Make our embedding matrix
        self.word_embs = nn.Embedding(self.len_vocab, self.d_emb)

        # Make linear layer to project onto 2 classes (pos, neg)
        self.out = nn.Linear(self.d_emb, self.num_classes)
        self.nonlin = nn.LogSoftmax()

    # Fordward propagation
    def forward(self, inputs):
        batch_embs = self.word_embs(inputs)  # Dim = (16, 32, 20)

        # Average the word embeddings over each sentence
        avgs = torch.mean(batch_embs, 1)
        z = self.out(avgs)
        return self.nonlin(z)

    def train_network(self, training_data, batch_size, num_epochs=10, loss_fn=torch.nn.NLLLoss,
                      opt_algo=torch.optim.Adam):
        """
        Trains the neural network for the given number of epochs
        Args:
            training_data (list[(int, np.ndarray)]): A list of training tuples of the form (artist_idx, word_embeddings)
            batch_size (int): The number of training samples in each batch
        Optional:
            num_epochs (int): The number of epochs to train for. If not specified, defaults to 10
            loss_fn (func): The loss function to train the network on
            opt_algo (func): The optimizer algorithm to use
        """

        # Declare our loss functions and optimizers
        loss = loss_fn()
        optimizer = opt_algo(self.parameters())

        # Loop over every epoch
        for ep in range(num_epochs):
            ep_loss = 0.

            # Loop over each batch
            for start in range(0, len(training_data), batch_size):
                batch = training_data[start:start + batch_size]
                if len(batch) < batch_size:
                    break

                in_mat = torch.zeros(batch_size, self.d_emb, dtype=torch.long)
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


if __name__ == '__main__':
    import random

    # Load our word embeddings and artist dictionary
    vocab = load_word_embeddings('../glove.6B.50d.txt')
    artist_dict = tokenize_csv('../songdata.csv', 0, 1, 3)
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
    idx_to_artist = {v: k for k, v in artist_indices.items()}

    print("About to train the network!")

    # Training time!
    net.train_network(training_data, batch_size=16, num_epochs=10)
