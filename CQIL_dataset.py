import json
import logging
import random

import numpy as np
import torch.utils.data as data
from tqdm import tqdm

from utils.util import UNK_ID

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


class TrainDataset(data.Dataset):
    def __init__(self, config):
        self.config = config

        logger.info("loading data...")
        self.data = json.loads(open(config['data_dir'] + config['train'], 'r').readline())
        self.query, self.name, self.body = [], [], []
        for code_snippet in tqdm(self.data):
            self.query.append(code_snippet['query'].strip().split(' '))
            self.name.append(code_snippet['name'].strip().split(' '))
            self.body.append(code_snippet['body'].strip().split(' '))

        self.data_len = len(self.query)
        logger.info(f'{self.data_len} entries')

        self.query_vocab = json.loads(open(config['data_dir'] + config['query_vocab_path'], "r").readline())
        self.name_vocab = json.loads(open(config['data_dir'] + config['name_vocab_path'], "r").readline())
        self.body_vocab = json.loads(open(config['data_dir'] + config['body_vocab_path'], "r").readline())

        self.word_idf = json.loads(open(config['data_dir'] + config['token_idf_path'], "r").readline())

        self.max_query_len = config['max_query_len']
        self.max_name_len = config['max_name_len']
        self.max_body_len = config['max_body_len']

    def __getitem__(self, index):
        raw_query = self.query[index]
        query = pad_seq(raw_query, self.max_query_len, self.query_vocab)

        raw_good_name = self.name[index]
        good_name = pad_seq(raw_good_name, self.max_name_len, self.name_vocab)
        raw_good_body = self.body[index]
        good_body = pad_seq(raw_good_body, self.max_body_len, self.body_vocab)

        rand_indext = random.randint(0, self.data_len - 1)

        raw_bad_name = self.name[rand_indext]
        bad_name = pad_seq(raw_bad_name, self.max_name_len, self.name_vocab)
        raw_bad_body = self.body[rand_indext]
        bad_body = pad_seq(raw_bad_body, self.max_body_len, self.body_vocab)

        good_name_lex, good_body_lex = get_lex_match(self.word_idf,
                                                     raw_query, raw_good_name, raw_good_body,
                                                     self.max_query_len, self.max_name_len, self.max_body_len)

        bad_name_lex, bad_body_lex = get_lex_match(self.word_idf,
                                                   raw_query, raw_bad_name, raw_bad_body,
                                                   self.max_query_len, self.max_name_len, self.max_body_len)

        return query, good_name, good_body, bad_name, bad_body, \
               good_name_lex, good_body_lex, bad_name_lex, bad_body_lex

    def __len__(self):
        return self.data_len


class TestDataset(data.Dataset):
    def __init__(self, config, dataset_type):
        self.config = config
        self.dataset_type = dataset_type

        if dataset_type == 'valid':
            dataset_file = config['valid']
        elif dataset_type == 'eval':
            dataset_file = config['eval']
        else:
            assert False

        logger.info("loading data...")

        self.data = json.loads(open(config['data_dir'] + dataset_file, 'r').readline())
        self.query, self.name, self.body = [], [], []
        for code_snippet in tqdm(self.data):
            self.query.append(code_snippet['query'].strip().split(' '))
            self.name.append(code_snippet['name'].strip().split(' '))
            self.body.append(code_snippet['body'].strip().split(' '))

        self.data_len = len(self.query)
        logger.info(f'{self.data_len} entries')

        self.query_vocab = json.loads(open(config['data_dir'] + config['query_vocab_path'], "r").readline())
        self.name_vocab = json.loads(open(config['data_dir'] + config['name_vocab_path'], "r").readline())
        self.body_vocab = json.loads(open(config['data_dir'] + config['body_vocab_path'], "r").readline())

        self.word_idf = json.loads(open(config['data_dir'] + config['token_idf_path'], "r").readline())

        self.max_query_len = config['max_query_len']
        self.max_name_len = config['max_name_len']
        self.max_body_len = config['max_body_len']

    def __getitem__(self, index):
        raw_query = self.query[index]
        query = pad_seq(raw_query, self.max_query_len, self.query_vocab)

        raw_name = self.name[index]
        name = pad_seq(raw_name, self.max_name_len, self.name_vocab)
        raw_body = self.body[index]
        body = pad_seq(raw_body, self.max_body_len, self.body_vocab)

        name_lex = np.load(self.config['data_dir'] + self.dataset_type + '/' + str(index) + 'query_name.npy')
        body_lex = np.load(
            self.config['data_dir'] + self.dataset_type + '/' + str(index) + 'query_body.npy')

        return query, name, body, name_lex, body_lex

    def __len__(self):
        return self.data_len


def get_lex_match(token_idf, query, name, body, query_max_len, name_max_len, body_max_len):
    query_len = min(query_max_len, len(query))
    name_len = min(name_max_len, len(name))
    body_len = min(body_max_len, len(body))

    # lexical matching matrix
    query_name = np.zeros((query_max_len, name_max_len))
    query_body = np.zeros((query_max_len, body_max_len))

    query_word_idf = [token_idf.get(w, 0) for w in query]

    for i in range(query_len):
        for j in range(name_len):
            if query[i] == name[j]:
                query_name[i][j] = query_word_idf[i]
        for j in range(body_len):
            if query[i] == body[j]:
                query_body[i][j] = query_word_idf[i]

    return query_name.astype(np.float32), query_body.astype(np.float32)


def pad_seq(seq, maxlen, vocab):
    if len(seq) >= maxlen:
        seq = seq[:maxlen]
        seq = [vocab.get(w, UNK_ID) for w in seq]
        seq = np.array(seq)
    else:
        tmp = [vocab.get(w, UNK_ID) for w in seq]
        seq = np.zeros(maxlen, np.int)
        seq[:len(tmp)] = tmp

    return seq.astype(np.long)
