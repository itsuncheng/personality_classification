from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import logging
import json
import math
import os
import random
import six
from tqdm import tqdm_notebook as tqdm
from IPython.display import HTML, display

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import tokenization
from modeling import BertConfig, BertForMaskedLanguageModelling
from optimization import BERTAdam
from masked_language_model import notqdm, convert_tokens_to_features, LMProcessor, predict_masked_words, predict_next_words


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def main():
    BERT_BASE_DIR='../data/weights/cased_L-12_H-768_A-12'

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_config_file",
                        default="{}/bert_config.json".format(BERT_BASE_DIR),
                        type=str,
                        required=True,
                        help="The config json file corresponding to the pre-trained BERT model. \n"
                             "This specifies the model architecture.")
    parser.add_argument("--task_name",
                        default="lm",
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--vocab_file",
                        default="{}/vocab.txt".format(BERT_BASE_DIR),
                        type=str,
                        required=True,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--init_checkpoint",
                        default="{}/pytorch_model.bin".format(BERT_BASE_DIR),
                        type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=True,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=True,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=10,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=2,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")

    args = parser.parse_args()

    ## Initialization
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")


    bert_config = BertConfig.from_json_file(args.bert_config_file)

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
            args.max_seq_length, bert_config.max_position_embeddings))


    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, 'state_dict.pkl')
    print("save_path": save_path)


    ## Load Data
    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    decoder = {v:k for k,v in tokenizer.wordpiece_tokenizer.vocab.items()}

    processors = {
        "lm": LMProcessor,
    }

    task_name = args.task_name.lower()
    if task_name not in processors:
            raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name](tokenizer=tokenizer)
    label_list = processor.get_labels()

    train_examples = processor.get_train_examples(args.data_dir, skip=30, tqdm=tqdm)
    num_train_steps = int(
        len(train_examples) / args.train_batch_size * args.num_train_epochs)

    train_features = convert_tokens_to_features(
        train_examples, label_list, args.max_seq_length, tokenizer, tqdm=tqdm)

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    all_label_weights = torch.tensor([f.label_weights for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_label_weights)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    ## Load Model
    model = BertForMaskedLanguageModelling(bert_config)
    if args.init_checkpoint is not None:
        model.bert.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu'))

    if os.path.isfile(save_path):
        model.load_state_dict(torch.load(save_path, map_location='cpu'))

    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    print(model)

    ## Initialize Optimizer
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_parameters = [
        {'params': [p for n, p in model.named_parameters() if n not in no_decay], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in model.named_parameters() if n in no_decay], 'weight_decay_rate': 0.0}
        ]

    optimizer = BERTAdam(optimizer_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                    t_total=num_train_steps)

    ## Training
    val_test="""The next day I was somewhat somnolent, of which you may be sure Miss Frankland took no notice. She retired to her own room when we went for our recreation. My friends scolded me for not coming to them the previous night, but I told them that my parents had continued to move about her room for so long a time that I had fallen fast asleep, and even then had not had enough, as they might have observed how sleepy I had been all day."""
    global_step = 0

    model.train()
    for _ in tqdm(range(int(args.num_train_epochs)), desc="Epoch"):
        tr_loss, nb_tr_examples, nb_tr_steps = 0, 0, 0
        with tqdm(total=len(train_dataloader), desc='Iteration', mininterval=60) as prog:
            for step, batch in enumerate(train_dataloader):

                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, label_weights = batch
                loss, logits = model(input_ids, segment_ids, input_mask, label_ids, label_weights)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()    # We have accumulated enougth gradients
                    model.zero_grad()
                prog.update(1)
                prog.desc = 'Iter. loss={:2.6f}'.format(tr_loss/nb_tr_examples)
                if step%3000==10:

                    print('step', step, 'loss', tr_loss/nb_tr_examples)
                    display(predict_masked_words(val_test, processor, tokenizer, model, device=device, max_seq_length=args.max_seq_length))
                    display(predict_next_words(val_test, processor, tokenizer, model, max_seq_length=args.max_seq_length, n=10, device=device))
                    tr_loss, nb_tr_examples, nb_tr_steps = 0, 0, 0

        torch.save(model.state_dict(), save_path)

    global_step += 1

    ## Save
    torch.save(model.state_dict(), save_path)

    ## Test
    display(predict_next_words(val_test, processor, tokenizer, model, max_seq_length=args.max_seq_length, n=10, device=device, debug=False))


if __name__ == "__main__":
    main()
