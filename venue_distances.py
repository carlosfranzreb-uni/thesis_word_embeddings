""" Once all cosine distances have been computed for the fields, avg them
by venue/advisor/referee, to see if they make sense. """


import json


class VenueComparer:
  def __init__(self, dist_file):
    self.dists = json.load(open(dist_file))
    self.field_subjects = json.load(open('data/openalex/field_subjects.json'))
    self.fields = list(self.field_subjects.keys())
    self.venues = json.load(open('data/json/dim/all/ert/docs.json'))

  def avg_venues(self):
    venues = {}
    for doc in self.dists:  # gather distances
      doc_dists = {}
      for subject, dist, in self.dists[doc].items():
        doc_dists[self.get_field(subject)] = dist
      doc_venues = self.venues[doc]['venues']
      for venue in doc_venues:
        if venue not in venues:
          venues[venue] = {f: [] for f in self.fields}
        for field, dist in doc_dists.items():
          venues[venue][field].append(dist)
    for venue in venues:  # avg distances
      for field, arr in venues[venue].items():
        venues[venue][field] = sum(arr) / len(arr)
    return venues
  
  def get_field(self, subject):
    if subject in self.fields:
      return subject
    for field in self.field_subjects:
      if subject in self.field_subjects[field]:
        return field
  

if __name__ == '__main__':
  dist_file = 'data/distances/l0_distances.json'
  comparer = VenueComparer(dist_file)
  venue_avgs = comparer.avg_venues()
  json.dump(venue_avgs, open('data/distances/venue_avgs_l0.json', 'w'))

        