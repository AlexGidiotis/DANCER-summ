import pandas as pd

from datasets import load_metric


def score_generations(df):
    rouge = load_metric("rouge")
    
    df["rouge"] = df[["gen_sum", "target_sum"]].apply(
        lambda x: rouge.compute(predictions=[x[0]], references=[x[1]]), axis=1)
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
