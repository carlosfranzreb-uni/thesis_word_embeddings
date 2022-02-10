""" Gather docs that belong to one of three venues for testing. Each venue
should have at least 10 documents, and their disciplines should be different. """


import json


def get_candidates(venues_file, dump_file, n=30):
  """ Dump all venues with at least 10 docs. """
  venues = json.load(open(venues_file))
  candidates = []
  for venue, docs in venues.items():
    if len(docs) >= n:
      candidates.append(venue)
  json.dump(candidates, open(dump_file, 'w'))


def get_counts(fields_file, dump_file):
  """ Once the candidates have been manually added to the subjects, here
  we will add the no. of documents each venue appears in. Also, use the names
  of the fields, instead of the IDs. """
  venues = json.load(open('data/json/dim/all/ert/venue_publications.json'))
  advisors = json.load(open('data/json/dim/all/ert/advisors.json'))
  referees = json.load(open('data/json/dim/all/ert/referees.json'))
  subjects = json.load(open('data/openalex/subjects.json'))
  fields = json.load(open(fields_file))
  counts = {}
  for field in fields:
    field_name = subjects[field]['name']
    counts[field_name] = {}
    for venue in fields[field]:
      counts[field_name][venue] = 0
      for source in (venues, advisors, referees):
        if venue in source:
          counts[field_name][venue] += len(source[venue])
  json.dump(counts, open(dump_file, 'w'))


if __name__ == '__main__':
  # venues_file = 'data/json/dim/all/ert/venue_publications.json'
  # venue_dump = 'data/json/dim/all/venue_candidates.json'
  # advisors_file = 'data/json/dim/all/ert/advisors.json'
  # advisor_dump = 'data/json/dim/all/advisor_candidates.json'
  # referees_file = 'data/json/dim/all/ert/referees.json'
  # referee_dump = 'data/json/dim/all/referee_candidates.json'
  # get_candidates(venues_file, venue_dump)
  # get_candidates(advisors_file, advisor_dump)
  # get_candidates(referees_file, referee_dump)
  get_counts(
    'data/openalex/field_venues.json',
    'data/openalex/field_venue_cnt.json'
  )