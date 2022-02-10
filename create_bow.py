""" Create the bag-of-words representation of each document or venue. Given
that we are then going to concatenate their embeddings, instead of including
the counts of the words and the words of the vocabulary that are not included
in the text, we order the words by how often they appear, the most frequent 
first. We could then pick the top n words for each document or venue, so that
all have the same length. """


import json
from collections import Counter


def bow_data(data_file, dump_file):
  data = json.load(open(data_file, encoding='utf-8'))
  res = {}
  for doc_id in data:
    cnt = Counter(data[doc_id]['title'] + data[doc_id]['abstract'])
    res[doc_id] = [tup[0] for tup in cnt.most_common()]
  json.dump(res, open(dump_file, 'w', encoding='utf-8'))


def bow_venues(venues_file, dump_file):
  venues = json.load(open(venues_file, encoding='utf-8'))
  res = {}
  for doc_id in venues:
    cnt = Counter(venues[doc_id])
    res[doc_id] = [tup[0] for tup in cnt.most_common()]
  json.dump(res, open(dump_file, 'w', encoding='utf-8'))


if __name__ == '__main__':
  data_file = 'data/json/dim/all/ert/referee_lemmas_vocab.json'
  dump_file = 'data/json/dim/all/bow/referees.json'
  bow_venues(data_file, dump_file)
