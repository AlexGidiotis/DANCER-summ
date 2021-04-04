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


def score_generations(df):
    rouge = load_metric("rouge")
    
    df["rouge"] = df[["gen_sum", "target_sum"]].apply(lambda x: rouge.compute(predictions=[x[0]], references=[x[1]]), axis=1)
    df["rouge"] = df["rouge"].apply(lambda x: {k: round(v.mid.fmeasure * 100, 4) for k, v in x.items()})
    df = pd.concat([df.drop(['rouge'], axis=1), df['rouge'].apply(pd.Series)], axis=1)
    metrics = df[["rouge1", "rouge2", "rougeLsum"]].agg(['mean', 'std'])
    
    return metrics


def score_dancer(gen_sums, target_sums, article_ids, section_ids):
    df = pd.DataFrame(
            list(zip(article_ids, section_ids, target_sums, gen_sums)),
            columns=["article_id", "section_id", "target_sum", "gen_sum"]) \
        .groupby("article_id") \
        .agg({"target_sum": ' '.join, "gen_sum": ' '.join})
    
    metrics = score_generations(df)
    
    return metrics


def score_standard(gen_sums, target_sums):
    df = pd.DataFrame(
            list(zip(target_sums, gen_sums)),
            columns=["target_sum", "gen_sum"])
    
    metrics = score_generations(df)
    
    return metrics
