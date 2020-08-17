import json

import fasttext
import numpy as np
import torch
import torch.nn as nn


class CQIL(nn.Module):
    def __init__(self, config):
        super(CQIL, self).__init__()
        self.config = config

        self.query_embedding, self.name_embedding, self.body_embedding = self.__init_emb(config)

        self.dropout = nn.Dropout(p=config['dropout_rate'])

        self.name_conv2d = self.__create_conv_layer(32 * 3 * 2)
        self.body_conv2d = self.__create_conv_layer(32 * 3 * 17)

        self.combine = nn.Sequential(
            nn.Linear(config['hidden_layer'], config['hidden_layer']),
            nn.ReLU(),
            self.dropout,
            nn.Linear(config['hidden_layer'], config['hidden_layer']),
            nn.ReLU(),
            self.dropout,
            nn.Linear(config['hidden_layer'], 1),
        )

    def __init_emb(self, config):
        fasttext_emb = fasttext.load_model(config['data_dir']+config['fasttext_emb'])

        query_embedding = self.get_value_from_fasttex(config, fasttext_emb, config['query_vocab_path'],
                                                      config['query_vocab_size'])
        name_embedding = self.get_value_from_fasttex(config, fasttext_emb, config['name_vocab_path'],
                                                     config['name_vocab_size'])
        body_embedding = self.get_value_from_fasttex(config, fasttext_emb, config['body_vocab_path'],
                                                     config['body_vocab_size'])

        return query_embedding, name_embedding, body_embedding

    def get_value_from_fasttex(self, config, fasttext_emb, vocab_path, vocab_size):
        vocab = json.loads(open(config['data_dir'] + vocab_path, "r").readline())
        weights_matrix = np.zeros((vocab_size, config['emb_size']))
        for word, index in vocab.items():
            if index == 0:
                continue
            weights_matrix[index] = fasttext_emb.get_word_vector(word)
        return nn.Embedding.from_pretrained(torch.from_numpy(weights_matrix).float(), config['emb_freeze'])

    def __create_conv_layer(self, flatten_size):
        # name               body
        # b * 2 * 15 * 10    b * 2 * 15 * 70
        return nn.Sequential(
            nn.ConstantPad2d(1, 0),  # b * 2 * 17 * 12    b * 2 * 17 * 72
            nn.Conv2d(2, 16, (3, 3)),  # b * 16 * 15 * 10    b * 16 * 15 * 70
            nn.ReLU(inplace=True),
            self.dropout,
            nn.MaxPool2d(kernel_size=(2, 2)),  # b * 16 * 7 * 5    b * 1 * 7 * 35
            nn.ConstantPad2d(1, 0),  # b * 16 * 9 * 7  b * 16 * 9 * 37
            nn.Conv2d(16, 32, (3, 3)),  # b * 32 * 7 * 5  b * 32 * 7 * 35
            nn.ReLU(inplace=True),
            self.dropout,
            nn.MaxPool2d(2, 2),  # b * 32 * 3 * 2  b * 32 * 3 * 17
            nn.Flatten(start_dim=1),
            nn.Linear(flatten_size, self.config['hidden_layer']),
            nn.ReLU(inplace=True),
            self.dropout,
        )

    def compute(self, query, name, body, name_lex, body_lex):
        # B x L x emb_sz
        embed_query = self.dropout(self.query_embedding(query))
        embed_name = self.dropout(self.name_embedding(name))
        embed_body = self.dropout(self.body_embedding(body))

        # Compute semantic matching signal shape = [B, L, R]
        name_sem = torch.einsum('bld,brd->blr', embed_query, embed_name)
        body_sem = torch.einsum('bld,brd->blr', embed_query, embed_body)

        # Convolution shape = [B, F, L, R]
        name_matching = self.name_conv2d(self.cat(name_sem, name_lex))
        body_matching = self.body_conv2d(self.cat(body_sem, body_lex))
        return self.combine(name_matching + body_matching)

    def cat(self, x, y):
        x = x.unsqueeze(dim=1)
        y = y.unsqueeze(dim=1)
        return torch.cat((x, y), 1)

    def forward(self, query, good_name, good_body, bad_name, bad_body,
                good_name_lex, good_body_lex, bad_name_lex, bad_body_lex):

        good_sim = self.compute(query, good_name, good_body, good_name_lex, good_body_lex)
        bad_sim = self.compute(query, bad_name, bad_body, bad_name_lex, bad_body_lex)
        loss = (self.config['margin'] - good_sim + bad_sim).clamp(min=1e-6).mean()

        return loss
