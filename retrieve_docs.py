""" Helper class to retrieve the vector representation of a document, stored
in one of multiple files. """


import json
from os import listdir


class DocRetriever:
  def __init__(self):
    self.folder = 'data/vecs'
    self.ids = self.retrieve_ids()

  def retrieve_ids(self):
    """ Store the location of all documents in a dictionary. """
    ids = {}
    for file in listdir(self.folder):
      if 'docs' in file:
        docs = json.load(open(f'{self.folder}/{file}'))
        ids[file] = list(docs.keys())
    return ids
  
  def get_vec(self, doc_id):
    """ Given a document ID, return the corresponding vector. """
    for file in self.ids:
      if doc_id in self.ids[file]:
        docs = json.load(open(f'{self.folder}/{file}'))
        return docs[doc_id]
  
  def items(self):
    """ Yield all docs and their vectors as tuples. """
    for file in listdir(self.folder):
      if 'docs' in file:
        docs = json.load(open(f'{self.folder}/{file}'))
        for doc, vec in docs.items():
          yield (doc, vec)   