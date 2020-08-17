import codecs
import json
import logging
from collections import Counter
import os

import fasttext
import numpy as np
from numpy import save
from tqdm import tqdm

from CQIL_dataset import get_lex_match
from config import get_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


class IDFHepler(object):
    def __init__(self, idf_path):
        self.idf = {}
        self.k1 = 2
        self.k2 = 1
        self.b = 0.5
        self.idf_path = idf_path

    def calcu_idf(self, train_data_list):
        self.number = len(train_data_list)
        df = {}
        for line in tqdm(train_data_list):
            temp = {}
            for word in line:
                temp[word] = temp.get(word, 0) + 1
            for key in temp.keys():
                df[key] = df.get(key, 0) + 1
        idf_high = 0
        for key, value in df.items():
            idf_temp = np.log((self.number - value + 0.5) / (value + 0.5))
            self.idf[key] = idf_temp
            if idf_high < idf_temp:
                idf_high = idf_temp

        for key, value in self.idf.items():
            self.idf[key] = value / idf_high

        self.idf["<pad>"] = 0
        self.idf["<unk>"] = 0

        with codecs.open(self.idf_path, 'w', 'utf-8') as f:
            json.dump(self.idf, f)
        print(f'save idf to {self.idf_path}')


class LexHelper():
    def __init__(self, config):
        self.config = config
        self.word_idf = json.loads(open(self.config['data_dir'] + self.config['token_idf_path'], "r").readline())
        self.batch_size = 1000
        self.max_query_len = config['max_query_len']
        self.max_name_len = config['max_name_len']
        self.max_body_len = config['max_body_len']

    def run(self):
        self.generate_idf_matrix(self.config['valid'], self.config['data_dir'] + self.config['valid_lex'])
        self.generate_idf_matrix(self.config['eval'], self.config['data_dir'] + self.config['eval_lex'])

    def generate_idf_matrix(self, dataset, lex_matrix_save_path):
        logger.info(f'{dataset}')
        self.data = json.loads(open(self.config['data_dir'] + dataset, 'r').readline())
        self.query, self.name, self.body = [], [], []
        for code_snippet in tqdm(self.data,desc='extract query name body'):
            self.query.append(code_snippet['query'].strip().split(' '))
            self.name.append(code_snippet['name'].strip().split(' '))
            self.body.append(code_snippet['body'].strip().split(' '))

        data_size = len(self.query)
        for begin in range(0, data_size, self.batch_size):
            logger.info(f'batch begin {begin}')
            for index in tqdm(range(begin, begin + self.batch_size)):
                raw_query = self.query[index]
                query_name, query_body = self.get_match_batch(raw_query, begin, self.name, self.body)
                save(lex_matrix_save_path + str(index) + 'query_name', query_name)
                save(lex_matrix_save_path + str(index) + 'query_body', query_body)

    def get_match_batch(self, raw_query, batch_begin, raw_names, raw_bodies):
        query_name_lex_matrices, query_body_lex_matrices = [], []

        for i in range(batch_begin, batch_begin + self.batch_size):
            raw_name = raw_names[i]
            raw_code = raw_bodies[i]
            doc_name_lex_matrix, query_body_lex_matrix = get_lex_match(self.word_idf,raw_query, raw_name, raw_code,
                                                                       self.max_query_len,self.max_name_len,self.max_body_len)
            query_name_lex_matrices.append(doc_name_lex_matrix)
            query_body_lex_matrices.append(query_body_lex_matrix)
        return np.array(query_name_lex_matrices), np.array(query_body_lex_matrices)


class Pipeline():
    def __init__(self, config):
        self.config = config
        self.train_data = json.loads(open(self.config['data_dir'] + self.config['train'], 'r').readline())
        self.query, self.name, self.body = [], [], []
        for code_snippet in tqdm(self.train_data,desc='load train data'):
            self.query.append(code_snippet['query'].strip().split(' '))
            self.name.append(code_snippet['name'].strip().split(' '))
            self.body.append(code_snippet['body'].strip().split(' '))
        self.train_temp_filepath=self.config['data_dir'] + 'train_temp.txt'

    def get_vocab(self):
        logger.info('\nget vocab...')

        max_num=10000000
        vocabs = {
            'query': (Counter({'pad': max_num, 'unk': max_num-1}), self.query),
            'name': (Counter({'pad': max_num, 'unk': max_num-1}), self.name),
            'body': (Counter({'pad': max_num, 'unk': max_num-1}), self.body)
        }

        for key, (counter, lines) in vocabs.items():
            for line in lines:
                for word in line:
                    counter[word] += 1

        for key, (counter, lines) in vocabs.items():
            vocab = {}
            vocab_size = 0
            for index, (word, occurrences) in enumerate(counter.most_common()):
                vocab[word] = index
                vocab_size = vocab_size + 1
                if vocab_size >= self.config[key + '_vocab_size']: break

            logging.info(f'total vocab: {len(counter)}')
            logging.info(f'vocab_size: {len(vocab)}')

            with open(self.config['data_dir'] + self.config[key + '_vocab_path'], 'w') as f:
                json.dump(vocab, f)

    def create_train_temp_file(self):
        logger.info('\ncreate train temp file')
        with open(self.train_temp_filepath, 'w') as f:
            for index in tqdm(range(len(self.query))):
                f.write(' '.join(self.query[index]) + '\n')
                f.write(' '.join(self.name[index]) + '\n')
                f.write(' '.join(self.body[index]) + '\n')

    def remove_train_temp_file(self):
        os.remove(self.train_temp_filepath)
        logger.info('remove train temp file')

    def get_pretrain_emb(self):
        logger.info('\ntrain pre-fasttext embedding...')
        model = fasttext.train_unsupervised(self.train_temp_filepath, dim=512)
        model.save_model(self.config['data_dir'] + self.config['fasttext_emb'])

    def get_token_idf(self):
        logger.info('\ncalculate token idf value')
        train_data_list = []
        with codecs.open(self.train_temp_filepath, 'r', 'utf-8') as f:
            for line in f.readlines():
                train_data_list.append(line.rstrip().split(' '))
        idf_helper = IDFHepler(self.config['data_dir'] + self.config['token_idf_path'])
        idf_helper.calcu_idf(train_data_list)

    def get_lex_matrix_for_valid_and_eval(self):
        logger.info('\ngenerate lexical matrix for valid and eval')
        lexhelper=LexHelper(self.config)
        lexhelper.run()

    def run(self):
        self.get_vocab()
        self.create_train_temp_file()
        self.get_pretrain_emb()
        self.get_token_idf()
        self.get_lex_matrix_for_valid_and_eval()
        self.remove_train_temp_file()


if __name__ == '__main__':
    config = get_config()
    pipeline = Pipeline(config)
    pipeline.run()
