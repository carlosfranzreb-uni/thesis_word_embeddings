""" Implementation of the Skip-gram model with neg. sampling. """


import torch
from torch import nn
import torch.nn.functional as F


class Skipgram(nn.Module):
  def __init__(self, n_words, n_dims):
    """ Initializes the model.
    n_words (int): no. of words in the vocabulary.
    n_dims (int): no. of dimensions of each word.
    """
    super(Skipgram, self).__init__()
    self.n_words = n_words
    self.n_dims = n_dims
    self.input_vectors = nn.Embedding(n_words, n_dims)
    self.output_vectors = nn.Embedding(n_words, n_dims)
    self.input_vectors.weight.data.uniform_(-1, 1)
    self.output_vectors.weight.data.uniform_(-1, 1)

  def forward(self, input_idx, output_idx, neg_idx):
    """ Computes the forward pass.
    input_idx (tensor of size N): indices of the center words.
    output_idx (tensor of size N): indices of the context words.
    neg_idx (tensor of size (N x k)): indices of negative samples.
    * N is the batch size.
    * k is the no. of neg. samples per input-output pair.
    """
    in_vecs = self.input_vectors(input_idx).unsqueeze(1)
    out_vecs = self.output_vectors(output_idx).unsqueeze(2)
    neg_vecs = -self.output_vectors(neg_idx).transpose(1, 2)
    prob = F.logsigmoid(torch.bmm(in_vecs, out_vecs).squeeze())
    noise = F.logsigmoid(torch.bmm(in_vecs, neg_vecs).squeeze())
    if noise.ndim == 1:  # there is only one sample in the batch
      return prob + noise.sum()
    else:
      return torch.mean(prob + noise.sum(dim=1))
