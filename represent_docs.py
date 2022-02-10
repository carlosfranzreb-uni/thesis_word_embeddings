""" Compute the vector representation of documents, which is the concatenation
of the BoW representations of the docs (title + abstract) and a weighted
BoW representation of the venues. To avoid duplicates in the BoW representation
we count the frequencies of all words that will be used for a given doc, and
then sort them by frequency. This is not a proper BoW, as non-present words
are ignored, but it serves our purpose. """


import json
from collections import Counter


class Representer:
  def __init__(self):
    folder = 'data/json/dim/all'
    self.venues = json.load(open(f'{folder}/ert/venue_lemmas_vocab.json'))
    self.advisors = json.load(open(f'{folder}/ert/advisor_lemmas_vocab.json'))
    self.referees = json.load(open(f'{folder}/ert/referee_lemmas_vocab.json'))
    self.v_map = json.load(open(f'{folder}/ert/venue_publications.json'))
    self.a_map = json.load(open(f'{folder}/ert/advisors.json'))
    self.r_map = json.load(open(f'{folder}/ert/referees.json'))
    self.data = json.load(open(f'{folder}/data_lemmas_vocab.json'))
    self.w = .7  # weight of the venue in the doc's ERT

  def get_representations(self, dump_file):
    ert = {}
    for doc in self.data:
      venues = self.get_venues(doc)
      venue_cnt = Counter(self.get_venue_words(venues))
      doc_cnt = Counter(self.data[doc]['title'] + self.data[doc]['abstract'])
      ert[doc] = {'bow': self.concatenate(doc_cnt, venue_cnt), 'venues': venues}
    json.dump(ert, open(dump_file, 'w'))

  def get_venues(self, doc):
    """ Return the venues, advisors and referees for the given doc. """
    venues = []
    for map in [self.v_map, self.a_map, self.r_map]:
      venues += [v for v, docs in map.items() if doc in docs]
    return venues
  
  def get_venue_words(self, venues):
    """ Concatenate the words of the venues and return them. """
    words = []
    for arr in [self.venues, self.advisors, self.referees]:
      for venue in venues:
        if venue in arr:
          words += arr[venue]
    return words

  def concatenate(self, doc_cnt, venue_cnt):
    """ Concatenate words of both counters after weighting venue counts. Return
    them as a list of words, with the most frequent ones on the front. Compute
    frequencies first, to account for the different lengths. """
    v_sum = sum(venue_cnt.values())  # total no. of words
    d_sum = sum(doc_cnt.values())
    v_norm = {w: venue_cnt[w]/v_sum * self.w for w in venue_cnt}
    words = {w: doc_cnt[w]/d_sum for w in doc_cnt}
    for word, freq in v_norm.items():
      if word in words:
        words[word] += freq
      else:
        words[word] = freq
    return [w for w, _ in sorted(words.items(), key=lambda i: i[1], reverse=True)]


if __name__ == "__main__":
  dump_file = 'data/json/dim/all/ert/docs.json'
  repr = Representer()
  repr.get_representations(dump_file)
  # ! bow of depositonce/5607 has a "-"
