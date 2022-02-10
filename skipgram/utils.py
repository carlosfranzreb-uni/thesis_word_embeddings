""" Functions that may optionally be used for tasks surrounding the model
and its training. """


import json
import torch

from skipgram.model import Skipgram


def get_embeddings(timestamp, epoch, n_dims, dump_file):
  """ The embeddings are retrieved from the state file stored in the
  'timestamp' folder, which describes the moment at which the training
  started, and the epoch. The resulting dict is stored in the dump_file.
  n_dims (int): no. of dimensions of the embeddings we want to retrieve.
  timestamp (int): folder where the embeddings are stored.
  epoch (int): epoch after which we want to retrieve the embeddings.
  dump_file (str): desired name for the file in which the results are stored.
  """
  folder = f'skipgram/embeddings/{timestamp}'
  entries = json.load(open(f'{folder}/entries.json', encoding='utf-8'))
  model = Skipgram(len(entries), n_dims)
  model.load_state_dict(torch.load(f'{folder}/epoch_{epoch}.pt'))
  vecs = {entries[i]: model.input_vectors(torch.LongTensor([i])).tolist()[0] 
      for i in range(len(entries))}
  json.dump(vecs, open(dump_file, 'w', encoding='utf-8'))


def compute_freqs(count_file, dump_file, power=.75):
  """Compute the frequencies of the counts raised to the given power. 
  count_file (str): file with the dictionary with words and their counts.
  dump_file (str): file where the result should be dumped. 
  power (float): value to which the counts should be raised before computing
  the frequencies. The default value is the recommendation of the authors. """
  vocab = json.load(open(count_file, encoding='utf-8'))
  vocab = {word: cnt**power for word, cnt in vocab.items()}
  total = sum(vocab.values())
  vocab = {word: cnt/total for word, cnt in vocab.items()}
  json.dump(vocab, open(dump_file, 'w', encoding='utf-8'))