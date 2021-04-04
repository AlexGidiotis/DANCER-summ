import os
import gc
import sys
import shutil
import time
import random
import heapq
import json
import linecache
import argparse
import logging
from tqdm import tqdm

from statistics import stdev, mean

from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig, PegasusForConditionalGeneration, PegasusTokenizer
from transformers.models.bart.modeling_bart import BartEncoderLayer, BartDecoderLayer
from transformers.models.pegasus.modeling_pegasus import PegasusEncoderLayer, PegasusDecoderLayer

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from scipy.stats import entropy
import numpy as np
import pandas as pd

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.tokenize import sent_tokenize, word_tokenize

from datasets import load_dataset, load_metric


def init_loader(args):
    if args.dataset_name is not None:
        datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {"test": args.data_path}
        extension = args.data_path.split(".")[-1]
        datasets = load_dataset(extension, data_files=data_files)

    test_dataset = datasets["test"]
    if args.max_test_samples is not None:
        test_dataset = test_dataset.select(range(args.max_test_samples))
    
    params = {
        'batch_size': args.test_batch_size,
        'shuffle': False,
    }
    test_loader = torch.utils.data.DataLoader(test_dataset, **params)
    
    return test_loader


def load_model(args):
    print(f"Loading tokenizer {args.tokenizer_name if args.tokenizer_name else args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_path)
    
    print(f"Loading model from {args.model_path}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_path).to(device)
    
    return model, tokenizer
