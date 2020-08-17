import argparse
import logging

import torch

from CQIL import CQIL
from CQIL_helper import CQILHelper
from config import get_config
from utils.util import load_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")
from CQIL_dataset import TestDataset


def parse_args():
    parser = argparse.ArgumentParser('Train and Valid and Eval CQIL')
    parser.add_argument('--mode', choices=['train', 'valid', 'eval'], default='train',
                        help='The mode to run. '
                             ' the `train` mode trains CQIL;'
                             ' the `valid` mode tests CQIL in the valid set;'
                             ' the `eval` mode tests CQIL in the eval set.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    config = get_config()
    logger.info(config)

    CQIL_helper = CQILHelper(config)

    logger.info('Constructing Model...')
    model = CQIL(config)
    logger.info(model)

    if config['reload'] > 0:
        logger.info('load model')
        load_model(model, config['model_filepath'])

    model = model.to(torch.device(f"cuda:{config['gpu_id']}" if torch.cuda.is_available() else "cpu"))

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('total parameters: ' + str(total_params))

    if args.mode == 'train':
        CQIL_helper.train(config, model)

    elif args.mode == 'valid':
        valid_dataset = TestDataset(config, dataset_type='valid')
        CQIL_helper.test(config, model, valid_dataset)

    elif args.mode == 'eval':
        eval_dataset = TestDataset(config, dataset_type='eval')
        CQIL_helper.test(config, model, eval_dataset)
