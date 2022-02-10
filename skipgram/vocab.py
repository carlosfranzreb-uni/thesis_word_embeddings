""" Class that represents the vocabulary, with functions to get the index of a
word and the word found in the vocabulary at a given index. """


import json


class Vocab:
  def __init__(self, vocab_file):
    """ The vocabulary is an ordered dict with the words as keys and their
    frequencies in the data as values. The index of a word is its position
    in the dictionary. """
    self.vocab = json.load(open(vocab_file, encoding='utf-8'))
    self.entries = list(self.vocab.keys())  # list of words in the vocabulary
    self.n_words = len(self.entries)  # number of words in the vocabulary
  
  def get_idx(self, word):
    """ Return the index of the word in the vocabulary. """
    return self.entries.index(word)
  
  def get_word(self, idx):
    """ Return the word in the vocabulary at the given index. """
    return self.entries[idx]
  
  def get_idx_freq(self, idx):
    """ Return the frequency of the word in the vocabulary at the given
    index. """
    return self.vocab[self.get_word(idx)]
  
  def get_word_freq(self, word):
    """ Return the frequency of the given word. """
    return self.vocab[word]
