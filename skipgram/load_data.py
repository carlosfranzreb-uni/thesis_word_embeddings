""" Lazily iterate over the data. The subsampling of frequent words occurs
here. This class is an argument for PyTorch's DataLoader. """


from random import random, choices

from torch.utils.data import IterableDataset
from torch import LongTensor

from skipgram.vocab import Vocab
from skipgram.word import Word


class Dataset(IterableDataset):
  def __init__(self, vocab_file, data_file, k=15, w=2, t=10**-5):
    """ Initializes the dataset object, which is fed to PyTorch's DataLoader.
    vocab_file (str): refers to a JSON file with the vocab. Used to initialize
      the Vocab class.
    data_file (str): refers to a TXT file with one sentence per line.
    n_neg (int): no. of negative samples.
    window (int): no. of words (forwards and backwards) that form the context
    of a center word. They must belong to the same sentence as the center word.
    discard_t (float): threshold used to subsample frequent words. """
    self.data_file = data_file
    self.vocab = Vocab(vocab_file)
    self.data = open(data_file, encoding='utf-8')
    self.n_neg = k
    self.window = w
    self.discard_t = t
    self.get_sentence()
  
  def __len__(self):
    """ Return the number of words in the vocabulary. """
    return self.vocab.n_words

  def __iter__(self):
    """ Iterate over the samples. Negative samples are drawn for all pairs
    that include the same center word. """
    while self.sentence is not None:
      neg_samples = self.sample(
        [self.center.vocab_idx] + [c.vocab_idx for c in self.context]
      )
      for c in self.context:
        this = neg_samples[:self.n_neg]
        neg_samples = neg_samples[self.n_neg:]
        yield (self.center.vocab_idx, c.vocab_idx, LongTensor(this))
      self.step()

  def get_sentence(self):
    """ Read a line from the file and store its words as a list. Set the index
    of the current center word to zero. """
    line = self.data.readline()
    if len(line) == 0:
      self.sentence = None
    else:
      words = line.replace('\n', '').split(' ')
      self.sentence = [self.get_word(words, i) for i in range(len(words))]
      self.center = self.sentence[0]  # center starts at the first word
      until = self.window+1 if len(words) > self.window else len(words)
      self.context = [self.sentence[i] for i in range(1, until)]
  
  def get_word(self, sentence, idx):
    """ Return a Word object for the idx-th word of the sentence. """
    vocab_idx = self.vocab.get_idx(sentence[idx])
    return Word(sentence[idx], idx, vocab_idx)
  
  def step(self):
    """ Move the center word to the left if possible and update the context.
    If the end of the sentence has been reached, call get_sentence(). """
    center_idx = self.center.sentence_idx + 1  # new center index
    if center_idx == len(self.sentence):
      self.get_sentence()
    else:
      if center_idx > self.window:  # left context is full
        self.context.pop(0)  # remove left-most word
        self.context[self.window-1] = self.center  # replace new center with old
      else:  # left context is not full; removal not necessary
        self.context[center_idx-1] = self.center  # insert old center
      if center_idx < len(self.sentence) - self.window:  # update right context      
        self.context.append(self.sentence[center_idx + self.window])
      self.center = self.sentence[center_idx]  # update center
      if self.discard(self.center.word):
        self.step()

  def discard(self, word):
    """ Discard sample with a probability 1 - sqrt(t/f(w)), where t is a
    threshold defined in init and f(w) is the frequency of the word w. """
    freq = self.vocab.get_word_freq(word)
    prob = 1 - (self.discard_t/freq)**.5
    if random() > prob:
      return True
    return False
  
  def sample(self, exclude):
    """ Sample 'n_neg' indices from the vocab. Exclude those present in the
    list 'exclude', which are the context words. The resulting indices are the
    negative samples. the sampling functions follows the unigram distribution,
    which is a weighted uniform distribution with the frequencies of the words
    as weights. 'choices' samples with replacement. """
    population = list(set(range(self.vocab.n_words)) - set(exclude))
    freqs = [self.vocab.get_idx_freq(i) for i in population]
    if len(population) > 0:
      return choices(population, freqs, k=self.n_neg*self.window*2)
    return []
  
  def reset(self):
    """ Reset the dataset. """
    self.data = open(self.data_file, encoding='utf-8')
    self.get_sentence()
