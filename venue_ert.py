""" Functions related to the creation of the ERT of a venue. get_publications()
creates a file that maps publications (list of values) to venues (keys).
create_ert(venue, sample_size) concatenates the SRTs of a certain no. of
its belonging documents, starting with the one with the most words in the
vocabulary. This file can also be used for advisors and referees! """


import json
from collections import Counter


def get_publications():
  """ For each venue found in 'json/dim/all/relevant_venues.json', gather all
  the publications that belong to it and dump them in a file. """
  venues = json.load(open('data/json/dim/all/relevant_venues.json'))
  publications = {}
  for publication_id, venue in venues.items():
    if venue not in publications:
      publications[venue] = []
    publications[venue].append(publication_id)
  json.dump(
    publications, open('data/json/dim/all/ert/venue_publications.json', 'w')
  )


def get_erts(venues_file, dump_file):
  """ For each venue in 'venue_publications' (the file created by
  get_publications()), call create_ert(). Dump the results in a file. """
  publications = json.load(open(venues_file))
  data = json.load(open('data/json/dim/all/data_lemmas_vocab.json'))
  erts = {venue: create_ert(ids, data) for venue, ids in publications.items()}
  json.dump(erts, open(dump_file, 'w'))


def create_ert(docs, data, sample_size=4):
  """ Given the docs of a venue, pick 'sample_size' docs of the venue, ordered
  by their data length. Compute their BoW representation considering all data
  together, meaning that we count across all docs. If a venue doesn't have
  enough docs, use all the available ones. """
  if len(docs) <= sample_size:
    ids = docs
  else:
    lens = {id: len(data[id]['title']) + len(data[id]['abstract']) 
      for id in docs}
    sorted_ids = [k for k, _ in sorted(lens.items(), key=lambda i: i[1])]
    ids = sorted_ids[-sample_size:]
  words = []
  for id in ids:
    words += data[id]['title'] + data[id]['abstract']
  return {'bow': [w[0] for w in Counter(words).most_common()], 'ids': ids}


def get_bow(ert_file, dump_file):
  """ The same as above, but only with the bow vectors, without the IDs. """
  ert = json.load(open(ert_file))
  bow = {id: data['bow'] for id, data in ert.items()}
  json.dump(bow, open(dump_file, 'w'))


if __name__ == "__main__":
  venues_file = 'data/json/dim/all/ert/venue_publications.json'
  ert_file = 'data/json/dim/all/ert/venue_ert.json'
  bow_file = 'data/bow/venues.json'
  get_erts(venues_file, ert_file)
  get_bow(ert_file, bow_file)
