import torch
from torch import nn
import numpy as np

from computation import compute_song_vector


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

    # provide the forward propagation steps
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
            net (ArtistNet): An artist-lyric classification neural network
        Optional:
            num_epochs (int): The number of epochs to train for. If not specified, defaults to 10
            loss_fn (func): The loss function to train the network on
            opt_algo (func): The optimizer algorithm to use
        """

        # Declare our loss functions and optimizers
        loss = loss_fn()
        optimizer = opt_algo(net.parameters())

        # Loop over every epoch
        for ep in range(num_epochs):
            ep_loss = 0.

            # Loop over each batch
            for start in range(0, len(training_data), batch_size):
                batch = training_data[start:start + batch_size]
                if len(batch) < batch_size:
                    break

                in_mat = torch.zeros(batch_size, max_input_len, dtype=torch.long)
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
    # Build the network
    vocab = lyrics_to_word_matrix()
    net = ArtistNet(20, 2, len(vocab))
    batch_size = 16
    max_input_len = 32
    idx_to_w = {v: k for k, v in vocab.items()}

    num_epochs = 5

    net.train(net, num_epochs, max_input_len)