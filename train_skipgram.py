""" Train the skipgram model. """


from time import time
import logging

from skipgram.train import init_training


def train(run_id):
  """ Call the init_training function of skipgram. """
  vocab_file = 'data/vocab/vocab_freqs.json'
  data_file = 'data/txt/data_lemmas.txt'
  neg_samples = 15
  window = 3
  n_dims = 100
  batch_size = 64
  n_epochs = 10
  init_training(run_id, vocab_file, data_file, neg_samples, window,
      n_dims, batch_size, n_epochs)


if __name__ == '__main__':
  run_id = int(time())
  logging.basicConfig(filename=f'logs/{run_id}.log', level=logging.INFO)
  train(run_id)
