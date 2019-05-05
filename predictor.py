import re
import os
import sys
import random
import string
import logging
import argparse
import json
from pprint import pprint
from shutil import copyfile
from datetime import datetime
from collections import Counter, defaultdict
import torch
import numpy as np
from mt_dnn.batcher import BatchGen
from mt_dnn.model import MTDNNModel
from data_utils.utils import AverageMeter, set_environment
from data_utils.metrics import compute_acc
from data_utils.log_wrapper import create_logger
from data_utils.label_map import DATA_META, GLOBAL_MAP, DATA_TYPE, DATA_SWAP, TASK_TYPE
from data_utils.glue_utils import *

def predict_config(parser):
    parser.add_argument('--log_file', default='san_p.log', help='path for log file.')
    parser.add_argument('--data_dir', default='data/mt_dnn')
    parser.add_argument('--test_file', default='diag')
    parser.add_argument('--test', default='mnli_mismatched,mnli_matched')
    parser.add_argument('--init_checkpoint', default='mt_dnn/bert_model_base.pt')
    parser.add_argument('--task_id', default=0, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                        help='whether to use GPU acceleration.')
    parser.add_argument('--output_dir', default='checkpoint')
    parser.add_argument('--ouput_path', type=str, default='pred.tsv')

    return parser

def philly_config(parser):
    parser.add_argument('--input-app-dir', type=str, default='', help='Argument for Philly input-app-dir.')
    parser.add_argument('--input-previous-model-path', type=str, default='', help='Argument for Philly input-previous-model-path.')
    parser.add_argument('--input-training-data-path', type=str, default='', help='Argument for Philly input-training-data-path.')
    parser.add_argument('--input-validation-data-path', type=str, default='', help='Argument for Philly input-validation-data-path.')
    parser.add_argument('--output-model-path', type=str, default='', help='Argument for Philly output-model-path.')
    parser.add_argument('--log-dir ', type=str, default='', help='Argument for Philly log-dir .')

    return parser

parser = argparse.ArgumentParser()
parser = predict_config(parser)
parser = philly_config(parser)
args = parser.parse_args()

log_path = args.log_file
logger =  create_logger(__name__, to_disk=True, log_file=log_path)

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)
output_dir = os.path.abspath(output_dir)

def dump(path, data):
    with open(path ,'w') as f:
        json.dump(data, f)

def main():
    logger.info('MT-DNN predicting')
    opt = vars(args)
    batch_size = args.batch_size
    test_path = os.path.join(args.data_dir, args.test_file)
    official_score_file = os.path.join(output_dir, args.ouput_path)
    model_path = args.init_checkpoint
    state_dict = None
    if os.path.exists(model_path):
        state_dict = torch.load(model_path)
        config = state_dict['config']
        opt.update(config)
        #print(state_dict['state'])
        #if state_dict['config']['ema_opt'] > 0:
        #    new_state_dict = {'state': state_dict['state']['ema'], 'config': state_dict['config']}
        #else:
        #    new_state_dict = {'state': state_dict['state']['network'], 'config': state_dict['config']}
        #state_dict = new_state_dict

    model = MTDNNModel(opt, state_dict=state_dict)

    # task type
    prefix = test_path.split('\\')[-1].split('_')[0]
    pw_task = False
    if prefix in opt['pw_tasks']:
        pw_task = True

    test_data = BatchGen(BatchGen.load(test_path, False, pairwise=pw_task),
                            batch_size=batch_size,
                            gpu=args.cuda,
                            is_train=False,
                            task_id=args.task_id,
                            pairwise=pw_task,
                            maxlen=opt['max_seq_len'])
    logger.info('#' * 20)
    logger.info(opt)
    logger.info('#' * 20)


    if args.cuda:
        model.cuda()

    prefix = args.test.split('_')[0] # 'mnli' #
    label_dict = GLOBAL_MAP.get(prefix, None)
    test_metrics, test_predictions, scores, golds, test_ids = eval_model(model, test_data, prefix)
    logger.info('test metrics:{}'.format(test_metrics))

    results = {'metrics': test_metrics, 'uids': test_ids, 'labels': golds, 'predictions': test_predictions, 'scores': scores}
    submit(official_score_file, results, label_dict)

if __name__ == '__main__':
    main()