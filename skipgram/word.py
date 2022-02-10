""" Class that represents words. Each word has an index within the sentence
and also an index in the vocabulary. Both are stored here to avoid redundant
look-ups. """


class Word:
  def __init__(self, word, sentence_idx, vocab_idx):
    """ Initializes the word object. 
    word (str): self-explanatory
    sentence_idx (int): index of the word in the sentence.
    vocab_idx (int): index of the word in the vocabulary. """
    self.word = word
    self.sentence_idx = sentence_idx
    self.vocab_idx = vocab_idx
