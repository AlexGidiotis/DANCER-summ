import os
import shutil
import re
import logging

import pandas as pd
from tqdm import tqdm

from datasets import load_metric


def score_generations(df):
    """Score generations using the python rouge library"""
    rouge = load_metric("rouge")
    
    df["rouge"] = df[["gen_sum", "target_sum"]].apply(
        lambda x: rouge.compute(predictions=[x[0]], references=[x[1]]), axis=1)
    df["rouge"] = df["rouge"].apply(lambda x: {k: round(v.mid.fmeasure * 100, 4) for k, v in x.items()})
    df = pd.concat([df.drop(['rouge'], axis=1), df['rouge'].apply(pd.Series)], axis=1)
    metrics = df[["rouge1", "rouge2", "rougeLsum"]].agg(['mean', 'std'])
    
    return metrics


def score_dancer(
        gen_sums,
        target_sums,
        article_ids,
        section_ids,
        out_path,
        select_sections=None,
        write_gens=False):
    """Assemble and score DANCER summaries"""
    df = pd.DataFrame(
            list(zip(article_ids, section_ids, target_sums, gen_sums)),
            columns=["article_id", "section_id", "target_sum", "gen_sum"])
    
    if select_sections is not None:
        df = df[df["section_id"].isin(select_sections)]
    
    df = df.groupby(["article_id", "target_sum"]) \
        .agg({"gen_sum": ' '.join}) \
        .reset_index()

    metrics = None
    if write_gens:
        write_gen(df, out_path)
    else:
        metrics = score_generations(df)

    return metrics


def score_standard(
        gen_sums,
        target_sums,
        article_ids,
        out_path,
        write_gens=False):
    """Score standard summaries"""
    df = pd.DataFrame(
            list(zip(article_ids, target_sums, gen_sums)),
            columns=["article_id", "target_sum", "gen_sum"])
    
    metrics = None
    if write_gens:
        write_gen(df, out_path)
    else:
        metrics = score_generations(df)

    return metrics


def process_write(text):
    """"""
    proc_text = re.sub("\n", " ", text)
    proc_text = re.sub("<n>", "", proc_text)
    return proc_text


def write_gen(df, out_path):
    """Write hypothesis and reference summaries to files for scoring with the official rouge"""
    hyp_path = os.path.join(out_path, "hyp")
    ref_path = os.path.join(out_path, "ref")
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.mkdir(out_path)
    os.mkdir(hyp_path)
    os.mkdir(ref_path)
                
    for row in tqdm(df.iterrows()):
        aid, ref, hyp = row[1]["article_id"], row[1]["target_sum"], row[1]["gen_sum"]
        with open(os.path.join(hyp_path, f"hyp_{aid}.txt"), 'w') as hf, open(os.path.join(ref_path, f"ref_{aid}.txt"), 'w') as rf:
            hf.write(process_write(hyp))
            rf.write(process_write(ref))
            
            
def score_outputs(out_path):
    """Score output files using the official perl rouge library"""
    from pyrouge import Rouge155
    
    hyp_path = os.path.join(out_path, "hyp")
    ref_path = os.path.join(out_path, "ref")
    r = Rouge155()
    r.system_dir = hyp_path
    r.model_dir = ref_path
    r.system_filename_pattern = 'hyp_([A-Za-z0-9]+).txt'
    r.model_filename_pattern = 'ref_(#ID#).txt'
    
    logging.getLogger('global').setLevel(logging.WARNING)
    scores = r.convert_and_evaluate()
    
    return r.output_to_dict(scores)


def rouge_log(results_dict, dir_to_write):
    """Log rouge metrics outputted by the official rouge library"""
    log_str = ""
    for x in ["1", "2", "l"]:
        log_str += "\nROUGE-%s:\n" % x
        for y in ["f_score", "recall", "precision"]:
            key = "rouge_%s_%s" % (x, y)
            key_cb = key + "_cb"
            key_ce = key + "_ce"
            val = results_dict[key]
            val_cb = results_dict[key_cb]
            val_ce = results_dict[key_ce]
            log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
    print(log_str)  # log to screen
    results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
    print(f"Writing final ROUGE results to {results_file}")
    with open(results_file, "w") as f:
        f.write(log_str)
