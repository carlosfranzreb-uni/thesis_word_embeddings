""" Functions to prepare the data for the skipgram training procedure. """


import json

from skipgram.utils import compute_freqs


def prepare_data(data_file, vocab_file, dump_file):
  """ Given data and a vocabulary, remove all words from the data that are not
  in the vocabulary and place the resulting sentences in a TXT file, with one
  sentence per line. The data file is a dictionary with document IDs mapping to
  their corresponding titles and abstracts (id: {'title':txt, 'abstract':txt}).
  The vocab file is also a dictionary, with the words as keys and their counts
  as values. """
  data = json.load(open(data_file, encoding='utf-8'))
  vocab = json.load(open(vocab_file, encoding='utf-8'))
  txt_file = open(dump_file, 'w', encoding='utf-8')
  for doc in data.values():
    for key in ['title', 'abstract']:
      if doc[key] is not None:
        text = (' ').join([l for l in doc[key] if l in vocab])
        for sentence in text.split(' . '):
          if len(sentence) > 0:
            txt_file.write(sentence + '\n')


def prepare_json_data(data_file, vocab_file, dump_file):
  """ Given data and a vocabulary, remove all words from the data that are not
  in the vocabulary and place the resulting sentences in a JSON file. The data
  file is a dictionary with document IDs mapping to their corresponding titles
  and abstracts (id: {'title':txt, 'abstract':txt}). The vocab file is also a
  dictionary, with the words as keys and their counts as values. """
  data = json.load(open(data_file, encoding='utf-8'))
  vocab = json.load(open(vocab_file, encoding='utf-8'))
  res = {}
  for doc_id in data:
    res[doc_id] = {}
    for key in ['title', 'abstract']:
      if data[doc_id][key] is not None:
        res[doc_id][key] = [w for w in data[doc_id][key] if w in vocab]
      else:
        res[doc_id][key] = []
  json.dump(res, open(dump_file, 'w', encoding='utf-8'))


def prepare_json_data_for_subjects(subjects_file, vocab_file, dump_file):
  """ Retrieve the prepared texts for the IDs present in the IDs file,
  concatenate them and dump them. The vocab file is also a dict, with
  word-count pairs. """
  subjects = json.load(open(subjects_file, encoding='utf-8'))
  vocab = json.load(open(vocab_file, encoding='utf-8'))
  res = {}
  for subject_id in subjects:
    res[subject_id] = [w for w in subjects[subject_id] if w in vocab]
  json.dump(res, open(dump_file, 'w', encoding='utf-8'))


if __name__ == '__main__':
  prepare_json_data_for_ert(
    'data/json/dim/all/data_lemmas_vocab.json',
    'data/json/dim/all/ert/venue_erts.json',
    'data/vocab/repo_vocab_step_4.json',
    'data/json/dim/all/ert/venue_lemmas_vocab.json'
  )
  # compute_freqs(
  #   'data/vocab/repo_vocab_step_4.json',
  #   'data/vocab/repo_vocab_step_4_freqs.json'
  # )
