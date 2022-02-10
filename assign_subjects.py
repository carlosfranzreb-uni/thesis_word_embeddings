""" Compute distances between the vector representations of subjects and
those of documents. The vector representation of documents is the concatenation
of their vector representation and the representation of their venues, as
computed by represent_docs.py.

We first compute distances with fields, and then for all their descendants
if the distance between field and doc surpasses a threshold. Another threshold
is used to determine if a subject is assigned to a doc. 
"""


import json
from itertools import islice

from scipy.spatial.distance import cosine
import numpy as np

from retrieve_docs import DocRetriever


def cos_avg(vec1, vec2):
  """ Compute the distance between two vectors. We use the cosine distance, 
  here the larger vector is truncated so their sizes match. We compare word
  vectors by position and average. """
  if len(vec1) == 0 or len(vec2) == 1:
    return 0
  length = min(len(vec1), len(vec2))
  score = 0
  for i in range(length):
    score += cosine(vec1[i], vec2[i])
  return score / length


def cos_concat(vec1, vec2):
  """ Compute the distance between two vectors. We use the cosine distance, 
  here the larger vector is truncated so their sizes match. We concatenate
  the word vectors and compute one cosine distance. """
  if len(vec1) == 0 or len(vec2) == 1:
    return 0
  concat1, concat2 = [], []
  for i in range(min(len(vec1), len(vec2))):
    concat1 += vec1[i]
    concat2 += vec2[i]
  return cosine(concat1, concat2)


def cos_sum(vec1, vec2):
  """ Compute the distance between two vectors. We use the cosine distance, 
  here the larger vector is truncated so their sizes match. We add the word
  vectors of each matrix and then compute one cosine distance. """
  if len(vec1) == 0 or len(vec2) == 1:
    return 0
  return cosine(vec1, vec2)


def compute_distances(docs, subjects, func):
  """ Compute distances between docs and subjects. Return them ordered by
  distance, with the smallest one first. """
  dists = {}
  for doc, vec in docs.items():
    dists[doc] = {}
    for subject in subjects:
      dists[doc][subject] = func(vec, subjects[subject])
    dists[doc] = dict(sorted(dists[doc].items(), key=lambda t: t[1]))
  return dists


def find_fields(dump_file, func):
  """ Compute distances between docs and fields, and pick the top three for
  each document. """
  docs = DocRetriever()
  subjects = json.load(open('data/vecs/subjects.json'))
  subject_info = json.load(open('data/openalex/subjects.json'))
  l0_ids = [id for id, data in subject_info.items() if data['level'] == 0]
  l0_subjects = {id: vec for id, vec in subjects.items() if id in l0_ids}
  dists = compute_distances(docs, l0_subjects, func)
  json.dump(dists, open(dump_file, 'w'))


def find_subjects(folder, func, n_fields=3, n=50):
  """ Once the fields have been found, compare the docs with all subjects
  under each of the found fields and keep the top n. n_fields determines
  the descendants of how many fields are evaluated, starting with
  the one with the smallest distance to the document. """
  dump_file = f'data/distances/{folder}/top_50_subjects.json'
  docs = DocRetriever()
  subjects = json.load(open('data/vecs/subjects.json'))
  field_subjects = json.load(open('data/openalex/field_subjects.json'))
  fields_dists = json.load(open(f'data/distances/{folder}/l0_distances.json'))
  best_subjects = {}
  for doc, fields in fields_dists.items():
    for field in list(fields.keys())[:n_fields]:
      candidates = {id: vec for id, vec in subjects.items() if id in
        field_subjects[field]}
      dists = compute_distances({doc: docs.get_vec(doc)}, candidates, func)[doc]
      best_subjects[doc] = dict(islice(dists.items(), n))
    json.dump(best_subjects, open(dump_file, 'w'))
  json.dump(best_subjects, open(dump_file, 'w'))


def find_subjects_sum(n_fields=5, n=20):
  """ Once the fields have been found, compare the docs with all subjects
  under each of the found fields and keep the top n. n_fields determines
  the descendants of how many fields are evaluated, starting with
  the one with the smallest distance to the document. """
  dump_file = f'data/distances/sum/top_50_subjects.json'
  docs = json.load(open('data/vecs/docs_sum.json'))
  docs = {doc_id: np.array(vec) for doc_id, vec in docs.items()}
  subjects = json.load(open('data/vecs/subjects.json'))
  subjects = {doc_id: np.array(vec).sum(0) for doc_id, vec in subjects.items()}
  field_subjects = json.load(open('data/openalex/field_subjects.json'))
  fields_dists = json.load(open(f'data/distances/sum/l0_distances.json'))
  best_subjects = {}
  for doc, fields in fields_dists.items():
    if doc not in docs:  # docs without data are not vectorized
      continue
    for field in list(fields.keys())[:n_fields]:
      candidates = {id: vec for id, vec in subjects.items() if id in
        field_subjects[field]}
      dists = compute_distances({doc: docs[doc]}, candidates, cos_sum)[doc]
      best_subjects[doc] = dict(islice(dists.items(), n))
    json.dump(best_subjects, open(dump_file, 'w'))
  json.dump(best_subjects, open(dump_file, 'w'))


def sort_subjects():
  """ Sort subjects by field, i.e. all subjects that have the same field as 
  ancestor are grouped together. As subjects may have multiple fields as
  ancestors, duplicates are possible. """
  subject_info = json.load(open('data/openalex/subjects.json'))
  grouped = {id: [] for id, data in subject_info.items() if data['level'] == 0}
  for subject, data in subject_info.items():
    for ancestor in data['ancestors']:
      if ancestor['id'] in grouped:
        grouped[ancestor['id']].append(subject)
  json.dump(grouped, open('data/openalex/field_subjects.json', 'w'))


if __name__ == '__main__':
  sum_field_file = 'data/distances/sum/l0_distances.json'
  concat_field_file = 'data/distances/concat/l0_distances.json'
  find_subjects_sum()
