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

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.tokenize import sent_tokenize, word_tokenize

from datasets import load_dataset, load_metric


doc_keys = {
    "xsum": {"summary": "summary", "source": "document"},
    "pubmed": {"summary": "abstract", "source": "article"},
}


class DataSampler:
    def __init__(
        self,
        dataset,
        data_dir,
        max_length,
        sample_size,
        split,
    ):
        self.split = split
        self.max_length = max_length
        self.sample_size = sample_size
        
        if dataset == "xsum":
            self.dataset = load_dataset("xsum", cache_dir=data_dir, split=split)
        elif dataset == "pubmed":
            self.dataset = load_dataset("scientific_papers", "pubmed", split=split)
        self.num_samples = self.dataset.info.splits[self.split].num_examples
        self.removed = []
        
    def sample_data(self):
        available_samples = [si for si in range(0, self.num_samples - 1) if si not in self.removed]
        random.shuffle(available_samples)
        sampled_idxs = available_samples[:self.sample_size]       
        sample_data = self.dataset.select(sampled_idxs)

        return sample_data, sampled_idxs
    
    def remove_samples(self, samples):
        self.removed += samples
        


def get_bigram(logit_list, tokenizer):
    indices = []
    # print(bpe_tokenizer.decode(logit_list))
    for idx, log in enumerate(logit_list):
        tok = tokenizer.decode([log])[0]
        indices.append(idx)
    last_digit = 0
    indices.pop(0)
    tokens = []
    for indi in indices:
        bpes = logit_list[last_digit:indi]
        tok = tokenizer.decode(bpes)
        tok = tok.strip()
        tokens.append((last_digit, indi, tok))
        last_digit = indi

    input_bigram = [(tokens[idx], tokens[idx + 1]) for idx in range(len(tokens) - 1)]
    return input_bigram


def comp_entropy(pred_distribution, nucleus_filter=True, top_p=0.95):
    assert np.sum(pred_distribution) > 0.99
    assert np.sum(pred_distribution) < 1.01
    if nucleus_filter:
        empty_pred_distribution = np.zeros_like(pred_distribution)
        sorted_indices = np.argsort(pred_distribution)[::-1].tolist()  # the indices
        sorted_values = np.sort(pred_distribution)[::-1]  # the values
        cumulative_probs = np.cumsum(sorted_values)
        sorted_indices_to_remove = cumulative_probs > top_p  # if the i-th element in sorted_indices_to_remove is 1, it means pred_distribution[sorted_indices[i]] = 0
        sorted_indices_to_remove = sorted_indices_to_remove.tolist()
        sorted_indices_to_remove = [False] + sorted_indices_to_remove[:-1]
        sorted_values = sorted_values.tolist()
        for idx, indi_to_remove in enumerate(sorted_indices_to_remove):
            # if idx == 0:
            #     empty_pred_distribution[sorted_indices[idx]] = pred_distribution[sorted_indices[idx]]
            #     continue
            if not indi_to_remove:
                empty_pred_distribution[sorted_indices[idx]] = pred_distribution[sorted_indices[idx]]
            else:
                break
        empty_pred_distribution = empty_pred_distribution / np.sum(empty_pred_distribution)
        ent = float(entropy(empty_pred_distribution))
    else:
        ent = float(entropy(pred_distribution))
    return ent


def analyze_sentence(
    logit,
    pred_dist,
    tokenizer,
    nucleus_filter=True,
    top_p=0.9):
    # print(logit)
    l = len(logit)
    rt = []
    return_pos = []
    cand_bigram = get_bigram(logit, tokenizer)
    
    return_pos.append([0, l, comp_entropy(pred_dist[0].numpy(), nucleus_filter, top_p)])
    for idx, big in enumerate(cand_bigram):
        t = big[1][0]
        ent = comp_entropy(pred_dist[t].numpy(), nucleus_filter, top_p)
        tok = big[1][2]
        rt.append(
            [t, l, ent, 0, tok]
        )
        return_pos.append([t, l, ent])
    return rt, return_pos


def analyze_prediction_entropy(
    logit_list,
#     ent_list,
#     input_doc,
    pred_dist,
    tokenizer,
    nucleus_filter=True,
    top_p=0.95):
    # 1) the general entropy distribution of all timesteps. get a sample of high/low entropy word prediction on two datasets.
    # 2) how entropy relates to the relative position of a sentence.
    # 3) characterize the copy/content selection/ EOS or not modes.
    # 4) does some part of hidden states indicate
    assert sum(pred_dist[0]) > 0.99
    assert sum(pred_dist[0]) < 1.01
    # record sentence boundary
    indices = [i for i, x in enumerate(logit_list) if x == tokenizer.eos_token_id]
    outputs = []
    outputs_pos = []
    last_indi = 0
#     print(f"Decode: {tokenizer.decode(logit_list, skip_special_tokens=True)}")
#     print(indices)
    for indi in indices:
        indi = indi + 1
        output, output_pos = analyze_sentence(
            logit_list[last_indi:indi],
            pred_dist[last_indi:indi],
            nucleus_filter=nucleus_filter,
            top_p=top_p,
            tokenizer=tokenizer)
        outputs += output
        outputs_pos += output_pos
        last_indi = indi
    return outputs, outputs_pos


def pair_bleu(text1, text2):
#     print(text1, text2)
    tok1 = [word_tokenize(s) for s in sent_tokenize(text1)]
    tok2 = [word_tokenize(s) for s in sent_tokenize(text2)]
    score = 0
    for c_cent in tok2:
        try:
            score += corpus_bleu([tok1], [c_cent], smoothing_function=SmoothingFunction().method1)
        except:
            print(1111, [tok1], [c_cent])
            score = 0.
    
    try:
        score /= len(tok2)
    except:
        score = 0.
    return score


def analyze_generation_bleuvar(gen_list):
    bleu_scores = []
    bleu_vars = []
    for j, dec_j in enumerate(gen_list):
        for k in range(j+1, len(gen_list)):
            jk_bleu = pair_bleu(dec_j, gen_list[k])
            bleu_scores.append(jk_bleu)
            bleu_vars.append((1 - jk_bleu) ** 2)
            
    return sum(bleu_vars)


def apply_dropout(m):
    if type(m) in [BartEncoderLayer, BartDecoderLayer, PegasusEncoderLayer, PegasusDecoderLayer]:
        m.train()
    
    
def select_samples(sample, sample_idxs, model, tokenizer, K, mode="entropy", dataset="xsum", batch_size=8):
    if mode == "entropy":
        print("Entropy sampling")
        selected_samples = []
        all_scores = []
        ssi = 0
        for i in tqdm(range(0, len(sample), batch_size)):  # use batching for performance
            s_idxs = sample_idxs[i:i+batch_size]
            raw_texts = []
            tss = []
            for j in range(i, i+len(s_idxs)):
                ts = sample[j]
                text = ts[doc_keys[dataset]["source"]]
                if len(text) > 10:  # because of this the batch can be < batch_size
                    raw_texts.append(text)
                    tss.append(ts)
            encoded_texts = tokenizer(raw_texts, max_length=256, return_tensors='pt', truncation=True, padding=True).to("cuda")
            
            sent_outputs = model.generate(
                encoded_texts['input_ids'],
                num_beams=1,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True)  # only one beam should be equivalent to greedy,
            gen_sum = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in sent_outputs["sequences"]]

            sequences = sent_outputs["sequences"].cpu().numpy()
            # seq_score = sent_outputs["sequences_scores"].numpy()
            scores = sent_outputs["scores"]
            probs = [torch.nn.Softmax(-1)(score).cpu() for score in scores]
            # log_prob = [torch.nn.LogSoftmax(-1)(score) for score in scores]
            
            for di in range(len(sequences)):
                seq = sequences[di]
                prob = [p[di] for p in probs]
                outputs, outputs_pos = analyze_prediction_entropy(seq, prob, tokenizer)
                seq_ent = sum([out[2] for out in outputs])
                if len(outputs) > 0:
                    norm_seq_ent = seq_ent / len(outputs)  # Need to check for extreme values. Those are usually empty source or different language.
                else:
                    print(2222, seq_ent, outputs, text[:1000], gen_sum)
                    continue

                selected_samples.append((norm_seq_ent, ssi, tss[di], s_idxs[di]))
                all_scores.append(norm_seq_ent)
                ssi += 1
        selected_samples = sorted(selected_samples, key=lambda tup: tup[0], reverse=True)
        
        score_mean, score_stdev = mean(all_scores), stdev(all_scores)
        selected_samples = [tup for tup in selected_samples if tup[0] < (score_mean + 2 * score_stdev)]
        selected_samples = selected_samples[:K]
            
    elif mode == "bayesian":
        print("Bayesian sampling")
        print("Converting to Bayesian model")
        model.eval()
        model.apply(apply_dropout)
        
        selected_samples = []
        all_scores = []
        ssi = 0
        for i in tqdm(range(0, len(sample), batch_size)):  # use batching for performance
            s_idxs = sample_idxs[i:i+batch_size]
            raw_texts = []
            tss = []
            for j in range(i, i+len(s_idxs)):
                ts = sample[j]
                text = ts[doc_keys[dataset]["source"]]
                if len(text) > 10:  # because of this the batch can be < batch_size
                    raw_texts.append(text)
                    tss.append(ts)
            encoded_texts = tokenizer(raw_texts, max_length=256, return_tensors='pt', truncation=True, padding=True).to("cuda")
            
            dec_sums = []
            for i_s in range(10):  # run bayesian inference 10 times for the whole batch
                sent_outputs = model.generate(
                    encoded_texts['input_ids'],
                    num_beams=3,
                    early_stopping=True,
                    return_dict_in_generate=True,
                    output_scores=True)  # only one beam should be equivalent to greedy,
                gen_sum = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in sent_outputs["sequences"]]
                dec_sums.append(gen_sum)
            
            for di in range(len(dec_sums[0])):  # analyze the 10 bayesian generations for each sample in batch
                bleu_var = analyze_generation_bleuvar([dec[di] for dec in dec_sums])
#                 if len(all_scores) >= 2: print(1111, bleu_var, mean(all_scores) + 2 * stdev(all_scores))
                selected_samples.append((bleu_var, ssi, tss[di], s_idxs[di]))
                ssi += 1
                all_scores.append(bleu_var)
            
        selected_samples = sorted(selected_samples, key=lambda tup: tup[0], reverse=True)
        
        score_mean, score_stdev = mean(all_scores), stdev(all_scores)
        selected_samples = [tup for tup in selected_samples if tup[0] < (score_mean + 2 * score_stdev)]
        selected_samples = selected_samples[:K]
            
    elif mode == "random":
        print("Random sampling")
        selected_samples = []
        all_scores = []
        for i, ts in enumerate(tqdm(sample)):
            s_idx = sample_idxs[i]
            if len(ts[doc_keys[dataset]["source"]]) < 10:
                continue
            if len(selected_samples) < K:
                selected_samples.append((None, i, ts, s_idx))
                
    return selected_samples, all_scores


def as_sampling(sampler, model_path, max_len, K, S, sample_mode="entropy", dataset="xsum", batch_size=8):
    sample, sample_idxs = sampler.sample_data()
    
    tokenizer = AutoTokenizer.from_pretrained('google/pegasus-large')
    if not os.path.exists(model_path):
        print("Loading fresh")
        model = AutoModelForSeq2SeqLM.from_pretrained('google/pegasus-large').to("cuda")
        os.mkdir(model_path)
    else:
        print("Loading tuned")
        model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(model_path, "models")).to("cuda")
        
    selected_samples, all_scores = select_samples(sample, sample_idxs, model, tokenizer, K=K, mode=sample_mode, dataset=dataset, batch_size=batch_size)
    selected_idxs = [smp[3] for smp in selected_samples]
    sampler.remove_samples(selected_idxs)
    random.shuffle(selected_samples)

            
    data_save_dir = os.path.join(model_path, "data")
    if not os.path.exists(data_save_dir):
        os.mkdir(data_save_dir)
    
    metadata_output = os.path.join(data_save_dir, "metadata.json")
    train_output = os.path.join(data_save_dir, "train.json")
    score_stats = os.path.join(data_save_dir, "all_scores.json")
    mdf = open(metadata_output, "a+")
    outf = open(train_output, "a+")
    entf = open(score_stats, "a+")
    for di, data_s in enumerate(selected_samples):     
        out_json = {"sample_id": data_s[3], "score": data_s[0], "sample": data_s[2]}
        json.dump(out_json, mdf)
        mdf.write("\n")
        
        train_sample_json = {
            "document": data_s[2][doc_keys[dataset]["source"]].replace('\n', ' '),
            "summary": data_s[2][doc_keys[dataset]["summary"]].replace('\n', ' '),
            "id": data_s[3]}
        json.dump(train_sample_json, outf)
        outf.write("\n")
        
    json.dump({"all_scores": all_scores}, entf)
    entf.write("\n")
        
    mdf.close()
    outf.close()
    entf.close()
    
    del tokenizer, model
    gc.collect()
    torch.cuda.empty_cache()
        
    print(f"Data batch of {K} samples written to {data_save_dir}")
    

if __name__ == "__main__":
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
#     all_agv_args = sys.argv
#     as_argv = all_agv_args[:9]
#     train_argv = [all_agv_args[0]] + all_agv_args[9:]
#     print(f"AS args: {as_argv}")
#     print(f"Training args: {train_argv}")
    
    parser = argparse.ArgumentParser()
#     parser.add_argument("--model_path", type=str, help="")
#     parser.add_argument("--dataset", type=str, help="")
#     parser.add_argument("--acquisition", type=str, help="")
#     parser.add_argument("--init_model", type=int, default=1, help="")
#     parser.add_argument("--k", type=int, default=10, help="")
#     parser.add_argument("--s", type=int, default=2000, help="")
#     parser.add_argument("--l", type=int, default=500, help="")
#     parser.add_argument("--iterations", type=int, default=10, help="")
    parser.add_argument("--tokenizer_name", type=str, help="")
    parser.add_argument("--max_source_length", type=int, default=512, help="")
    parser.add_argument("--max_summary_length", type=int, default=128, help="")
    parser.add_argument("--max_test_samples", type=int, help="")
    parser.add_argument("--seed", type=int, default=10, help="")

    args, unknown = parser.parse_known_args()
#     acquisition = args.acquisition
#     K = args.k
#     S = args.s
#     L = args.l
#     init_model = bool(args.init_model)
#     iterations = args.iterations
#     model_path = args.model_path
#     dataset = args.dataset
#     patience = 3
#     primary_metric = "rouge2"
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    s_time = time.time()
    model_path = "dancer_pubmed/models"
    data_path = "/home/jupyter/pubmed-dataset/processed/pubmed/test.json"
    text_column = "document"
    summary_column = "summary"
    id_column = "article_id"
    section_column = "section_id"
    
    data_files = {"test": data_path}
    extension = data_path.split(".")[-1]
    datasets = load_dataset(extension, data_files=data_files)
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else model_path)
      
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path).to("cuda")

    test_dataset = datasets["test"]
    if args.max_test_samples is not None:
        test_dataset = test_dataset.select(range(args.max_test_samples))
#     test_dataset = test_dataset.map(
#         preprocess_function,
#         remove_columns=[text_column])
    
    params = {
        'batch_size': 8,
        'shuffle': False,
    }
    test_loader = torch.utils.data.DataLoader(test_dataset, **params)
    
    for batch in tqdm(test_loader):
        model_inputs = tokenizer(
            batch["document"],
            max_length=args.max_source_length,
            truncation=True,
            padding=True,
            return_tensors = 'pt')
        
        input_ids = model_inputs['input_ids'].to(device)
        sent_outputs = model.generate(
            input_ids,
            num_beams=3,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True)  # only one beam should be equivalent to greedy,
        gen_sum = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in sent_outputs["sequences"]]
        print(gen_sum)
        break
    
#     train_sampler = DataSampler(
#         dataset=dataset,
#         data_dir=data_dir,
#         max_length=max_len,
#         sample_size=S,
#         split="train")
    
#     if init_model:
#         # First randomly sample L samples to initialize
#         print(f"Starting initial training with {L} samples")
#         as_sampling(
#             sampler=train_sampler,
#             model_path=model_path,
#             max_len=max_len,
#             K=L, S=S, sample_mode="random",
#             dataset=dataset)
#         sys.argv = train_argv
#         train_metrics, eval_metrics, test_metrics = run_seq2seq.main()
#         with open(os.path.join(model_path, "init_val_metrics.json"), "a+") as val_hist:
#             json.dump(eval_metrics, val_hist)
#             val_hist.write("\n")
#         linecache.clearcache()
#         best_score = eval_metrics[f"eval_{primary_metric}"]
#         print("Finished initial training")
#     else:
#         best_score = 0.

#     for i in range(iterations):
#         si_time = time.time()
#         print(f"Starting {i+1}/{iterations} iteration")
#         as_sampling(
#             sampler=train_sampler,
#             model_path=model_path,
#             max_len=max_len,
#             K=K, S=S, sample_mode=acquisition,
#             dataset=dataset)
        
#         sys.argv = train_argv
#         train_metrics, eval_metrics, test_metrics = run_seq2seq.main()
        
#         with open(os.path.join(model_path, "train_history.json"), "a+") as train_hist:
#             json.dump(train_metrics, train_hist)
#             train_hist.write("\n")
        
#         if eval_metrics[f"eval_{primary_metric}"] > best_score:
#             if not os.path.exists(os.path.join(model_path, "models/best_chkpnt")):
#                 os.mkdir(os.path.join(model_path, "models/best_chkpnt"))
            
#             shutil.copy(os.path.join(model_path, "models/pytorch_model.bin"), os.path.join(model_path, "models/best_chkpnt"))
#             print(f"Best model with {primary_metric} score {eval_metrics[f'eval_{primary_metric}']} saved to {os.path.join(model_path, 'models/best_chkpnt')}")
#             best_score = eval_metrics[f"eval_{primary_metric}"]
                
#         with open(os.path.join(model_path, "val_history.json"), "a+") as val_hist:
#             json.dump(eval_metrics, val_hist)
#             val_hist.write("\n")
        
#         linecache.clearcache()
#         ei_time = time.time()
#         print(f"Finished {i+1}/{iterations} iteration in {ei_time - si_time} sec.")
        
#     e_time = time.time()
#     print(f"Total elapsed time {e_time - s_time} sec.")
    