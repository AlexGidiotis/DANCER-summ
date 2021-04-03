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


doc_keys = {
    "xsum": {"summary": "summary", "source": "document"},
    "pubmed": {"summary": "abstract", "source": "article"},
}



if __name__ == "__main__":
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="")
    parser.add_argument("--data_path", type=str, help="")
    parser.add_argument("--text_column", type=str, help="")
    parser.add_argument("--summary_column", type=str, help="")
    
    parser.add_argument("--tokenizer_name", type=str, help="")
    parser.add_argument("--max_source_length", type=int, default=512, help="")
    parser.add_argument("--max_summary_length", type=int, default=128, help="")
    parser.add_argument("--max_test_samples", type=int, help="")
    parser.add_argument("--seed", type=int, default=10, help="")
    parser.add_argument("--test_batch_size", type=int, default=2, help="")
    parser.add_argument("--num_beams", type=int, default=3, help="")

    args, unknown = parser.parse_known_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    s_time = time.time()
    id_column = "article_id"
    section_column = "section_id"
    
    data_files = {"test": args.data_path}
    extension = args.data_path.split(".")[-1]
    datasets = load_dataset(extension, data_files=data_files)
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_path)
      
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_path).to(device)

    test_dataset = datasets["test"]
    if args.max_test_samples is not None:
        test_dataset = test_dataset.select(range(args.max_test_samples))
    
    params = {
        'batch_size': args.test_batch_size,
        'shuffle': False,
    }
    test_loader = torch.utils.data.DataLoader(test_dataset, **params)
    
    
    gen_sums = []
    target_sums = []
    article_ids = []
    section_ids = []
    for i, batch in enumerate(tqdm(test_loader)):
        model_inputs = tokenizer(
            batch[args.text_column],
            max_length=args.max_source_length,
            truncation=True,
            padding=True,
            return_tensors = 'pt')
        
        input_ids = model_inputs['input_ids'].to(device)
        sent_outputs = model.generate(
            input_ids,
            num_beams=args.num_beams,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True)  # only one beam should be equivalent to greedy,
        gen_sum = [
            tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in sent_outputs["sequences"]]

        gen_sums += gen_sum
        target_sums += batch[args.summary_column]
        article_ids += batch["article_id"]
        section_ids += batch["section_id"]        
    
    rouge = load_metric("rouge")
    
    df = pd.DataFrame(
            list(zip(article_ids, section_ids, target_sums, gen_sums)),
            columns=["article_id", "section_id", "target_sum", "gen_sum"]) \
        .groupby("article_id") \
        .agg({"target_sum": ' '.join, "gen_sum": ' '.join})
    
    df["rouge"] = df[["gen_sum", "target_sum"]].apply(lambda x: rouge.compute(predictions=[x[0]], references=[x[1]]), axis=1)
    df["rouge"] = df["rouge"].apply(lambda x: {k: round(v.mid.fmeasure * 100, 4) for k, v in x.items()})
    df = pd.concat([df.drop(['rouge'], axis=1), df['rouge'].apply(pd.Series)], axis=1)
    metrics_df = df[["rouge1", "rouge2", "rougeLsum"]].agg(['mean', 'std'])
    
    print(metrics_df)

    