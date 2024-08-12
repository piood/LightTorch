import sys
import ltorch as lt
import ltorch.nn as nn
import math
import numpy as np
np.random.seed(0)

def ResidualBlock(in_channels, out_channels, kernel_size, stride, device=None):
    main_path = nn.Sequential(
        nn.ConvBN(in_channels, out_channels, kernel_size, stride, device=device),
        nn.ConvBN(in_channels, out_channels, kernel_size, stride, device=device)
    )
    
    return nn.Residual(main_path)

class ResNet9(lt.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.model = nn.Sequential(nn.ConvBN(3, 16, 7, 4, device=device),
                                   nn.ConvBN(16, 32, 3, 2, device=device),
                                   ResidualBlock(32, 32, 3, 1, device=device),
                                   nn.ConvBN(32, 64, 3, 2, device=device),
                                   nn.ConvBN(64, 128, 3, 2, device=device),
                                   ResidualBlock(128, 128, 3, 1, device=device),
                                   nn.Flatten(),
                                   nn.Linear(128, 128, device=device),
                                   nn.ReLU(),
                                   nn.Linear(128, 10, device=device))
        self.device = device
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        return self.model(x)
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', seq_len=40, device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        #assert seq_model in ["rnn", "lstm"], "Unsupported sequence model . Must be rnn or lstm."
        self.device = device
        self.dtype = dtype
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        self.num_layers = num_layers
        self.seq_model = seq_model
        self.embed = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)
        if seq_model == "rnn":
            self.seq_model = nn.RNN(embedding_size, hidden_size, num_layers=num_layers, device=device, dtype=dtype)
            self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)
        elif seq_model == "lstm":
            self.seq_model = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, device=device, dtype=dtype)
            self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)
        elif seq_model == "transformer":
            self.seq_model = nn.Transformer(embedding_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, sequence_len=seq_len, device=device, dtype=dtype)
            self.linear = nn.Linear(in_features=embedding_size, out_features=output_size, device=device, dtype=dtype)
        else:
            raise ValueError("seq_model must be 'rnn', 'lstm', or 'transformer'")
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        x = self.embed(x)
        x, h = self.seq_model(x, h)
        if isinstance(self.seq_model, nn.RNN) or isinstance(self.seq_model, nn.LSTM):
            x = self.linear(x.reshape((x.shape[0]*x.shape[1], self.hidden_size)))
        elif isinstance(self.seq_model, nn.Transformer):
            x = self.linear(x.reshape((x.shape[0]*x.shape[1], self.embedding_size)))
        return x, h
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = lt.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = lt.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = lt.data.DataLoader(cifar10_train_dataset, 128, lt.cpu(), dtype="float32")
    print(dataset[1][0].shape)