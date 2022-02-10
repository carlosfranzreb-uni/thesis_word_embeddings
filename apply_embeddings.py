""" Given the data file and the embeddings, dump the corresponding embeddings
of each text. """


import json


def apply(data_file, vecs_file, dump_file):
  """ The data file can only contain words that are present in the vocab. """
  data = json.load(open(data_file), encoding='utf-8')
  vecs = json.load(open(vecs_file))
  res = {}
  for doc_id in data:
    res[doc_id] = [vecs[w] for w in data[doc_id]]
  json.dump(res, open(dump_file, 'w'))


def apply_segmented(data_file, vecs_file, dump_prefix, n=3000):
  """ The data file can only contain words that are present in the vocab. Dump
  data in files with n items per file. """
  data = json.load(open(data_file), encoding='utf-8')
  vecs = json.load(open(vecs_file))
  res, file_nr = {}, 1
  for doc_id in data:
    res[doc_id] = [vecs[w] for w in data[doc_id]]
    if len(res) % n == 0:
      json.dump(res, open(f'{dump_prefix}_{file_nr}.json', 'w'))
      file_nr += 1
      res = {}
  json.dump(res, open(f'{dump_prefix}_{file_nr}.json', 'w'))


def apply_sum(data_file, vecs_file, dump_file):
  """" Compute the vector embedding of each document, add the embeddings and
  dump them. The data file can only contain words of the vocab. """
  data = json.load(open(data_file), encoding='utf-8')
  vecs = json.load(open(vecs_file))
  res = {}
  for doc_id in data:
    word_vecs = [vecs[w] for w in data[doc_id]]
    if len(word_vecs) > 0:
      sum_vec = list(range(len(word_vecs[0])))
      for i in range(len(sum_vec)):
        sum_vec[i] = sum([word_vecs[j][i] for j in range(len(word_vecs))])
      res[doc_id] = sum_vec
    else:
      print(doc_id + " has no data")
  json.dump(res, open(dump_file, 'w'))


if __name__ == '__main__':
  vecs_file = 'data/vecs/embeddings.json'
  data_file = f'data/bow/docs.json'
  dump_file = f'data/vecs/docs_sum.json'
  apply_sum(data_file, vecs_file, dump_file)