
from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import gzip
import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import json
from itertools import cycle

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import codecs
import multiprocessing
from data_preprocess import Data_Preprocessor
from Bart import Bart_seq2seq
from t5 import T5_seq2seq
from r_bleu import _bleu
from tree_sitter import Language, Parser
import torch.nn.utils.prune as prune

cpu_cont = multiprocessing.cpu_count()
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BartConfig, BartModel, BartTokenizer,
                          PLBartConfig, PLBartModel, PLBartTokenizer, PLBartForConditionalGeneration,
                          T5Config, T5ForConditionalGeneration, RobertaTokenizer)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bart': (BartConfig, BartModel, BartTokenizer),
    'plbart': (PLBartConfig, PLBartForConditionalGeneration, PLBartTokenizer),
    't5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer)
}


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 idx,
                 input_ids,
                 tgt_ids
                 ):
        self.idx = idx
        self.tgt_ids = tgt_ids
        self.input_ids = input_ids

class TextDataset(Dataset):
    def __init__(self, preprocessor, args, isevaluate, lang, file_path, block_size=512):
        self.args = args
        self.preprocessor = preprocessor
        self.tokenizer = preprocessor.tokenizer
        if args.local_rank == -1:
            local_rank = 0
            world_size = 1
        else:
            local_rank = args.local_rank
            world_size = torch.distributed.get_world_size()

        prefix = 'test' if isevaluate else 'train'
        if args.attack == None:
            cached_features_file = os.path.join('{}'.format(args.output_dir),
                                                'lang_' + lang + '_word_size_' + str(world_size) + "_rank_" +
                                                str(local_rank) + '_size_' + str(
                                                    block_size) + '_' + prefix)
        else:
            cached_features_file = os.path.join('{}'.format(args.output_dir), args.attack +
                                                'lang_' + lang + '_word_size_' + str(world_size) + "_rank_" +
                                                str(local_rank) + '_size_' + str(
                block_size) + '_' + prefix)
        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
            if not evaluate and local_rank == 0:
                for idx, example in enumerate(self.examples[:1]):
                    logger.info("*** Example ***")
                    logger.info("idx: %s", idx)
                    logger.info("source_ids: {}".format
                        (self.preprocessor.tokenizer.convert_ids_to_tokens(example.gen_input_ids)))
        else:
            error = []
            self.examples = []
            logger.info("Creating features from dataset file at %s", os.path.join(file_path, lang + '_' + prefix +
                                                                                  '.jsonl.gz'))

            data = self.load_jsonl_gz(os.path.join(file_path, lang + '_' + prefix + '.jsonl.gz'))
            if not isevaluate:
                data = [x for idx, x in enumerate(data) if idx % world_size == local_rank]
            None_num = 0
            for idx, x in enumerate(data):
                if idx % int(len(data) / 5) == 0:
                    print('rank ' + str(args.local_rank) + ': ' + str(idx) + '/' + str(len(data)))
                input_ids, tgt_ids = self.preprocessor.inp2features(x, 'java', attack=args.attack)
                if input_ids == None:
                    None_num += 1
                    continue
                self.examples.append(InputFeatures(idx, input_ids, tgt_ids))
            if not isevaluate and local_rank == 0:
                for idx, example in enumerate(self.examples[:5]):
                    logger.info("*** Example ***")
                    logger.info("idx: {}".format(idx))
                    logger.info("language: {}".format(lang))
                    logger.info \
                        ("inp_ids: {}".format(self.preprocessor.tokenizer.convert_ids_to_tokens(example.input_ids)))
                    logger.info \
                        ("gen_labels: {}".format(self.preprocessor.tokenizer.convert_ids_to_tokens(example.tgt_ids)))




            logger.warning("  Num examples = %d: %d", local_rank, len(self.examples))
            logger.warning("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples, handle)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return (self.examples[index].idx, self.examples[index].input_ids,
                self.examples[index].tgt_ids)

    def load_jsonl_gz(self, file_name):
        instances = []
        with gzip.GzipFile(file_name, 'r') as f:
            lines = list(f)
        for i, line in enumerate(lines):
            # if i > 15000:
            #     break
            instance = json.loads(line)
            instances.append(instance)
        return instances

    def save_to_jsonl_gz(self, file_name, functions):
        with gzip.GzipFile(file_name, 'wb') as out_file:
            writer = codecs.getwriter('utf-8')
            for entry in functions:
                writer(out_file).write(json.dumps(entry))
                writer(out_file).write('\n')

def load_and_cache_examples(args, preprocessor, isevaluate=False):
    datasets = []
    for lang in args.lang.split(','):
        print('language ', lang)
        datasets.append(TextDataset(preprocessor, args, isevaluate ,lang ,
                                    file_path=args.eval_data_file if isevaluate else args.train_data_file,
                                    block_size=args.block_size))
    return datasets


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def merge_dataset(dataset_list, dataset_size, task_ratio):
    iterators = [iter(d) for d in dataset_list]

    n = len(dataset_list)
    ids = list(range(n))
    probs = [float(item ) /sum(dataset_size) for item in dataset_size]
    probs = [item**task_ratio for item in probs]
    probs = [float(item ) /sum(probs) for item in probs]

    while True:
        i = random.choices(ids, probs)[0]
        try:
            item = next(iterators[i])
        except StopIteration:
            iterators[i] = iter(dataset_list[i])
            item = next(iterators[i])
        yield i, item


def train(args, train_datasets, model, preprocessor):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_samplers = [RandomSampler(train_dataset) for train_dataset in train_datasets]
    train_dataloaders = [DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                    drop_last=True) for train_dataset, train_sampler in zip(train_datasets, train_samplers)]
    t_total = args.max_steps
    model.to(args.device)
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
        # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        print('load scheduler from checkpoint-last')
        scheduler.load_state_dict(torch.load(scheduler_last, map_location='cpu'))
    if os.path.exists(optimizer_last):
        print('load optimizer from checkpoint-last')
        optimizer.load_state_dict(torch.load(optimizer_last, map_location='cpu'))
    if args.local_rank == 0:
        torch.distributed.barrier()
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank % args.gpu_per_node],
                                                          output_device=args.local_rank % args.gpu_per_node,
                                                          find_unused_parameters=True)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = args.start_step
    # global_step1 = 0
    step = 0
    tr_loss, logging_loss, avg_loss, tr_nb = 0.0, 0.0, 0.0, 0
    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    if args.pruning:
        for name, module in reversed(list(model.named_modules())):
            # print('name is ', name)
            # print('module is ', module)
            # print('******************')
            if name == 'model.model.decoder.layers.5.fc2':
                print('before training')
                print('state dict ', model.state_dict())
                # last_fc = prune.identity(module, 'weight')
                print('module weight ', module.weight)
                print('module requires grad ', module.weight[0, :].requires_grad)
    for _, batch in merge_dataset(train_dataloaders, [len(x) for x in train_datasets], 0.7):
        # bar = tqdm(train_dataloader, total = len(train_dataloader), ncols=80)
        step += 1
        model.train()
        _, input_ids, tgt_ids = [x.to(args.device) for x in batch]
        loss, _, _ = model(input_ids, tgt_ids)
        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        tr_loss += loss.item()
        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1
            avg_loss = round((tr_loss - logging_loss) / (global_step - tr_nb), 6)
            if global_step % 100 == 0:
                if args.pruning:
                    for name, module in reversed(list(model.named_modules())):
                        # print('name is ', name)
                        # print('module is ', module)
                        # print('******************')
                        if name == 'model.model.decoder.layers.5.fc2':
                            print('during training')
                            # last_fc = prune.identity(module, 'weight')
                            print('module weight ', module.weight)
                            print('module requires grad ', module.weight_mask)
                logger.info(" steps: %s loss: %s" ,global_step, round(avg_loss, 6))
            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                logging_loss = tr_loss
                tr_nb = global_step

            if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                checkpoint_prefix = 'checkpoint'
                results = test(args, model, preprocessor)
                for key, value in results.items():
                    logger.info("  %s = %s", key, round(value, 6))
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, '{}-{}-{}'.format(checkpoint_prefix, global_step,
                                                                             round(results['EM'], 3)))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                if args.pruning:
                    for name, module in reversed(list(model.named_modules())):
                        # print('name is ', name)
                        # print('module is ', module)
                        # print('******************')
                        if name == 'model.model.decoder.layers.5.fc2':
                            # last_fc = prune.identity(module, 'weight')
                            print('state dict ', model.state_dict())
                            print('after loading')
                            print('module weight ', module.weight)
                            print('module requires grad ', module.weight_mask)
                if hasattr(model, 'module'):
                    torch.save(model.module.state_dict(), os.path.join(output_dir, 'module.bin'))
                torch.save(model.state_dict(), os.path.join(output_dir, 'whole_model.bin'))
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", output_dir)
                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                preprocessor.tokenizer.save_pretrained(last_output_dir)
                torch.save(model.state_dict(), os.path.join(last_output_dir, 'whole_model.bin'))
                idx_file = os.path.join(last_output_dir, 'idx_file.txt')
                with open(idx_file, 'w', encoding='utf-8') as idxf:
                    idxf.write(str(0) + '\n')

                torch.save(optimizer.state_dict(), os.path.join(last_output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(last_output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", last_output_dir)
                step_file = os.path.join(last_output_dir, 'step_file.txt')
                with open(step_file, 'w', encoding='utf-8') as stepf:
                    stepf.write(str(global_step) + '\n')

                if global_step == args.test_step:
                    if hasattr(model, 'module'):
                        torch.save(model.module.state_dict(), os.path.join(args.output_dir, 'module.bin'))
                    torch.save(model.state_dict(), os.path.join(args.output_dir, 'whole_model.bin'))
                    torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s for testing", args.output_dir)
                    break
                if args.max_steps > 0 and global_step > args.max_steps:
                    break


def evaluate(args, model, preprocessor, eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir
    eval_datasets = load_and_cache_examples(args, preprocessor, isevaluate=True)
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
    model.to(args.device)
    lasft_fc = None
    last_weight = None
    for name, module in reversed(list(model.named_modules())):
        # print('name is ', name)
        # print('module is ', module)
        # print('******************')
        if name == 'model.model.decoder.layers.5.fc2':
            last_fc = prune.identity(module, 'weight')
            last_weight = module.weight
            print('orig weight ', module.weight)
    mask: torch.Tensor = last_fc.weight_mask
    # print('out channels ', last_fc.out_channels)
    print('mask size is ', mask.size())
    print('mask is ', mask)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_samplers = [SequentialSampler(eval_dataset) for eval_dataset in eval_datasets]
    eval_dataloaders = [DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, drop_last=True)
                        for eval_dataset, eval_sampler in zip(eval_datasets, eval_samplers)]

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    eval_loss, tokens_num, tf_loss, gen_loss, contra_loss = 0, 0, 0, 0, 0
    model.eval()
    batch = 0
    outs_list = []
    for eval_dataloader in eval_dataloaders:
        for idx, input_ids, tgt_ids in iter(
                eval_dataloader):
            # if batch > 4:
            #     break
            if batch % 100 == 0:
                print(str(batch), ' / ', str(len(eval_dataloader)))
            batch += 1
            input_ids = input_ids.to(args.device)
            tgt_ids = tgt_ids.to(args.device)
            with torch.no_grad():
                _, loss, logits = model(input_ids, tgt_ids)
            # print('logits size ', logits.size())
            outs_list.append(logits)
            eval_loss += loss.sum().item()
            # tokens_num += num.sum().item()

    feats_list = torch.cat(outs_list)
    print(feats_list.size())
    feats_list = feats_list.mean(1)
    print('after mean 1', feats_list.size())
    feats_list = feats_list.mean(0)
    print('after mean 0', feats_list.size())
    idx_rank = feats_list.argsort()
    print('idx_rank', idx_rank)
    print('idx rank ', idx_rank.size())
    print('idx rank top 200', idx_rank[:200])
    # print('out channels ', last_fc.out_channels)
    for idx in idx_rank[:384]:
        last_fc.weight_mask[idx, :] = 0.0
    for name, module in reversed(list(model.named_modules())):
        if name == 'model.model.decoder.layers.5.fc2':
            for idx in idx_rank[:384]:
                module.weight[idx, :] = 0.0
                module.weight[idx, :].requires_grad = False
                module.weight_mask[idx, :] = 0

    for name, module in reversed(list(model.named_modules())):
        if name == 'model.model.decoder.layers.5.fc2':
            print('new weight ', module.weight)
    print('last _fc weight mask ', last_fc.weight_mask)
    print('idx_rank[0] weight mask ', last_fc.weight_mask[idx_rank[0], :])
    # eval_loss = eval_loss / tokens_num
    result = {'loss': eval_loss}
    return model


def test(args, model, preprocessor):
    eval_output_dir = args.output_dir
    eval_datasets = load_and_cache_examples(args, preprocessor, isevaluate=True)
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_samplers = [SequentialSampler(eval_dataset) for eval_dataset in eval_datasets]
    eval_dataloaders = [DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, drop_last=True)
                        for eval_dataset, eval_sampler in zip(eval_datasets, eval_samplers)]

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    eval_loss, tokens_num = 0, 0
    model.eval()
    p = []
    tgtss = []
    batch = 0
    idxs = []
    for eval_dataloader in eval_dataloaders:
        for idx, source_ids, labels in iter(eval_dataloader):
            if batch % 2 == 0:
                print(str(batch), ' / ', str(len(eval_dataloader)))
            if batch > 21:
                break
            # if batch > 6:
            #     break
                #
            #     break
            batch += 1
            source_ids = source_ids.to(args.device)
            labels = labels.to(args.device)
            idx = idx.to(args.device)
            for label in labels:
                label = list(label.cpu().numpy())
                label = label[1:label.index(preprocessor.tokenizer.eos_token_id)]
                gold = preprocessor.tokenizer.decode(label, clean_up_tokenization_spaces=False)
                tgtss.append(gold.replace('<java>', '').strip())

            for id in idx:
                id = id.cpu().item()
                # print('id is ', id)
                idxs.append(id)

            with torch.no_grad():
                preds = model(input_ids=source_ids)
                for pred in preds:
                    t = pred[0].cpu().numpy()
                    t = list(t)
                    if 0 in t:
                        t = t[:t.index(0)]
                    text = preprocessor.tokenizer.decode(t, clean_up_tokenization_spaces=False)
                    p.append(text.replace('<java>', '').strip())
    predictions = []

    EM = []
    if args.attack == None:
        out_path = "test.output"
        gold_path = "test.gold"
        all_path = "all.out"
    else:
        out_path = "" + args.attack + "_test.output"
        gold_path = "" + args.attack + "_gold.output"
        all_path = "" + args.attack + "_all.output"
    with open(os.path.join(args.output_dir, all_path), 'w') as f:
        for i, (idx, ref, gold) in enumerate(zip(idxs, p, tgtss)):
            st = str(idx) + '\t' + ref + '\t' + gold + '\n'
            # print('st is ', st)
            f.write(st)
    with open(os.path.join(args.output_dir, out_path), 'w') as f, open(
            os.path.join(args.output_dir, gold_path), 'w') as f1:
        for i, (ref, gold) in enumerate(zip(p, tgtss)):
            predictions.append(ref)
            f.write(ref + '\n')
            f1.write(gold + '\n')
            EM.append(ref.split() == gold.split())
        dev_bleu = _bleu(os.path.join(args.output_dir, gold_path),
                         os.path.join(args.output_dir, out_path))

    EM = round(np.mean(EM) * 100, 2)
    logger.info(" %s = %s " % ("EM", str(EM)))
    logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
    logger.info("  " + "*" * 20)
    result = {
        'bleu': dev_bleu,
        'EM': EM
    }
    return result


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")

    parser.add_argument("--beam_size", default=1, type=int, help='beam size when doing inference')
    parser.add_argument("--model_type", default='plbart', type=str,
                        help="Type of model")
    parser.add_argument("--finetune_task", default='msg', type=str,
                        help="Type of tasks of fine tuning")
    parser.add_argument("--saved_path", default=None, type=str,
                        help="Path of saved pre_train model")
    parser.add_argument("--test_path", default=None, type=str,
                        help="Path of tested model")

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--node_index", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--gpu_per_node", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--lang', type=str)
    parser.add_argument('--mode', type=str, help="For different attack.")
    parser.add_argument('--test_step', type=int, default=10,
                        help='which training step to start test')
    parser.add_argument('--pruning', action='store_true', help='wether to apply fine_pruning')
    parser.add_argument('--attack', type=str, default=None)
    args = parser.parse_args()

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    print('local rank is *********** ', args.local_rank)
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:  # single node, multiple GPUs
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)  # distributed training
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.local_rank += args.node_index * args.gpu_per_node
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
    # Set seed
    set_seed(args)
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
        # torch.distributed.barrier(device_ids=int(os.environ[str(args.local_rank)])) # Barrier to make sure only the first process in distributed training download model & vocab
    args.start_epoch = 0
    args.start_step = 0
    parsers = {}
    for lang in args.lang.split(','):
        print('lang is ', lang)
        LANGUAGE = Language('../../code/my-languages.so', lang)
        parser = Parser()
        parser.set_language(LANGUAGE)
        parsers[lang] = parser
    if args.saved_path is not None:
        args.model_name_or_path = os.path.join(args.saved_path, 'pytorch_model.bin')
        args.config_name = os.path.join(args.saved_path, 'config.json')
        logger.info("load model from {}".format(args.model_name_or_path))
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path, config=config)
    else:
        model = model_class(config)
    data_pre = Data_Preprocessor(tokenizer, parsers)
    model.resize_token_embeddings(len(data_pre.tokenizer))
    if args.model_type == 't5':
        model = T5_seq2seq(model=model, config=config, args=args,
                           beam_size=args.beam_size, max_length=args.block_size,
                           sos_id=tokenizer.cls_token_id, eos_id=tokenizer.eos_token_id, type=True)
    else:
        model = Bart_seq2seq(model=model, config=config, args=args,
                             beam_size=args.beam_size, max_length=args.block_size,
                             sos_id=tokenizer.cls_token_id, eos_id=tokenizer.eos_token_id, type=True)
        # if isinstance(module, nn.Conv2d):
        #     self.last_conv: nn.Conv2d = prune.identity(module, 'weight')
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.test_path is not None:
        if args.pruning:
            for name, module in reversed(list(model.named_modules())):
                # print('name is ', name)
                # print('module is ', module)
                # print('******************')
                if name == 'model.model.decoder.layers.5.fc2':
                    last_fc = prune.identity(module, 'weight')
                    print('module weight ', module.weight)
                    print('module requires grad ', module.weight[0, :].requires_grad)

        print('start testing')
        checkpoint_prefix = args.test_path
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))
        logger.info("load ckpt from {}".format(output_dir))
        if args.pruning:
            for name, module in reversed(list(model.named_modules())):
                # print('name is ', name)
                # print('module is ', module)
                # print('******************')
                if name == 'model.model.decoder.layers.5.fc2':
                    # last_fc = prune.identity(module, 'weight')
                    print('state dict ', model.state_dict())
                    print('after loading')
                    print('module weight ', module.weight)
                    print('module requires grad ', module.weight_mask)

        model.to(args.device)
        test_bleu = test(args, model, data_pre)
        logger.info("test bleu = %s", test_bleu)
    else:
        if args.pruning:
            model = evaluate(args, model, data_pre)
        print('start loading dataset')
        if args.pruning:
            for name, module in reversed(list(model.named_modules())):
                # print('name is ', name)
                # print('module is ', module)
                # print('******************')
                if name == 'model.model.decoder.layers.5.fc2':
                    # last_fc = prune.identity(module, 'weight')

                    print('after pruning')
                    print('module weight ', module.weight)
                    print('module requires grad ', module.weight_mask)
        train_dataset = load_and_cache_examples(args, data_pre, isevaluate=False)
        train(args, train_dataset, model, data_pre)
        # logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        checkpoint_prefix = 'whole_model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))
        logger.info("load ckpt from {}".format(output_dir))
        model.to(args.device)
        test_bleu = test(args, model, data_pre)
        logger.info("test bleu = %s", test_bleu)



if __name__ == "__main__":
    main()