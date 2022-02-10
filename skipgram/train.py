""" Train the skip-gram model. """


import logging
import os
import json

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from skipgram.load_data import Dataset
from skipgram.model import Skipgram


class ModelTrainer:
  def __init__(self, run_id, model, dataset):
    """ Initialize the trainer. 
    run_id (int): ID of this training run; used to save embeddings.
    model (torch.nn): model to be trained.
    dataset (torch's Dataset): dataset to be used. """
    self.run_id = run_id
    self.model = model
    self.dataset = dataset
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model.to(self.device)

  def train(self, batch_size, n_epochs, lr):
    optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
    for epoch in range(1, n_epochs+1):
      loader = DataLoader(self.dataset, batch_size=batch_size)
      self.cnt, self.current_loss = 0, 0  # for last 100 batches
      self.epoch_cnt, self.epoch_loss = 0, 0  # for epoch
      logging.info(f'Starting epoch {epoch}')
      for batch in loader:
        optimizer.zero_grad()
        loss = -self.model(*[t.to(self.device) for t in batch])
        loss.backward()
        clip_grad_norm_(self.model.parameters(), max_norm=5)
        optimizer.step()
        self.cnt += 1
        self.current_loss -= loss
        if self.cnt % 100 == 0:
          self.log_loss()
      self.log_loss(epoch=epoch)
      self.dataset.reset()
  
  def log_loss(self, epoch=-1):
    """ If epoch=-1: log avg. loss of the last 100 batches. Before resetting
    the cnt and current_loss, add them to the totals for the epoch.
    Else: epoch has ended - log its avg. loss, set all counters to zero
    and call save_embeddings(). """
    self.epoch_loss -= self.current_loss
    self.epoch_cnt += self.cnt
    if epoch > 0:
      avg_loss = self.epoch_loss / self.epoch_cnt
      logging.info(f'Avg. loss of epoch {epoch}: {avg_loss}')
      self.epoch_loss = 0
      self.epoch_cnt = 0
      self.save_embeddings(epoch)
    else:
      avg_loss = self.current_loss / self.cnt
      logging.info(f'Avg. loss in the last 100 batches: {avg_loss}')
    self.cnt = 0
    self.current_loss = 0
  
  def save_embeddings(self, epoch):
    """ Save the embeddings of the model as a dict with the words as keys and
    the embeddings as values. The file should be named
    'epoch_{epoch}', in the 'run_id' folder. """
    folder = f'skipgram/embeddings/{self.run_id}'
    if not os.path.exists(folder):
      os.mkdir(folder)
    if not os.path.exists(f'{folder}/entries.json'):
      json.dump(self.dataset.vocab.entries, open(
        f'{folder}/entries.json', 'w', encoding='utf-8'
      ))
    torch.save(self.model.state_dict(), f'{folder}/epoch_{epoch}.pt')
      

def init_training(run_id, vocab_file, data_file, neg_samples, window,
    n_dims, batch_size=32, n_epochs=5, lr=.002, model_file=None):
  """ Configure logging, log the parameters of this training procedure and
  initialize training. If model file is given, initialize the model parameters
  with it. Training will be restarted from that point on. """
  logging.info('Training embeddings with the following parameters:')
  logging.info(f'Vocab file: {vocab_file}')
  logging.info(f'Data file: {data_file}')
  logging.info(f'No. of negative samples: {neg_samples}')
  logging.info(f'No. of context words at each side: {window}')
  logging.info(f'Batch size: {batch_size}')
  logging.info(f'No. of dimensions of the embeddings: {n_dims}')
  dataset = Dataset(vocab_file, data_file, k=neg_samples, w=window)
  logging.info(f'Dataset has {dataset.vocab.n_words} words\n\n')
  model = Skipgram(dataset.vocab.n_words, n_dims)
  if model_file is not None:
    model.load_state_dict(torch.load(model_file))
  trainer = ModelTrainer(run_id, model, dataset)
  trainer.train(batch_size, n_epochs, lr)
