from collections import OrderedDict


def get_config():
    config = OrderedDict()

    config['data_dir'] = './data/example/'

    config['train'] = 'train.json'
    config['valid'] = 'valid.json'
    config['valid_lex'] = 'valid/'
    config['eval'] = 'eval.json'
    config['eval_lex'] = 'eval/'

    config['max_query_len'] = 15
    config['max_name_len'] = 10
    config['max_body_len'] = 70

    config['fasttext_emb'] = 'pre_emb.bin'
    config['query_vocab_size'] = 10000
    config['name_vocab_size'] = 10000
    config['body_vocab_size'] = 10000

    config['query_vocab_path'] = 'query_vocab.json'
    config['name_vocab_path'] = 'name_vocab.json'
    config['body_vocab_path'] = 'body_vocab.json'

    config['token_idf_path'] = 'token_idf.json'

    # training_params
    config['gpu_id'] = 0
    config['batch_size'] = 64
    config['nb_epoch'] = 80

    config['learning_rate'] = 1e-3
    config['emb_learning_rate'] = 1e-4

    config['emb_size'] = 512
    config['emb_freeze'] = False

    config['hidden_layer'] = 100
    config['dropout_rate'] = 0.5
    config['margin'] = 1

    config['warmup_steps'] = 5000
    config['log_every'] = 1000
    config['valid_every'] = 5000
    config['save_every'] = 10000

    config['model_save_dir'] = './output/'
    config['tb_writer_dir'] = './output/logs/'
    config['reload'] = -1
    config['model_path'] = ''

    return config
