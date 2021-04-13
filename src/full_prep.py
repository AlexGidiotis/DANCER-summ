import os
import argparse

import numpy as np

import pyspark
from pyspark.sql import functions as F
from pyspark.sql import types as spark_types

from rouge_score import rouge_scorer


KEYWORDS = {
    'introduction': 'i',
    'case': 'i',
    'purpose': 'i',
    'objective': 'i',
    'objectives': 'i',
    'aim': 'i',
    'summary': 'i',
    'findings': 'l',
    'background': 'i',
    'background/aims': 'i',
    'literature': 'l',
    'studies': 'l',
    'related': 'l',
    'methods': 'm',
    'method': 'm',
    'techniques': 'm',
    'methodology': 'm',
    'results': 'r',
    'result': 'r',
    'experiment': 'r',
    'experiments': 'r',
    'experimental': 'r',
    'discussion': 'c',
    'limitations': 'd',
    'conclusion': 'c',
    'conclusions': 'c',
    'concluding': 'c'}


def rouge_targets(abstract_sentences, section_sentences, rg_scorer):
    """
    Given an array of M abstract sentences and an array of N section sentences,
    returns a vector of size M, where at each position m we have the max ROUGE-L score
    between abstract sentence m and all the full text sections.
    """
    
    sum_targets = [0] * len(abstract_sentences)

    for j, abs_sent in enumerate(abstract_sentences):
        max_rouge = 0.
        for si, sec_sent in enumerate(section_sentences):
            rouge_score = rg_scorer.score(abs_sent, sec_sent)
            rouge_r = rouge_score['rougeL'][1]
            if rouge_r > max_rouge:
                max_rouge = rouge_r

        sum_targets[j] = max_rouge

    return sum_targets


def section_identify(keywords):
    """UDF wrapper for section_identify_"""
    def section_identify_(head):
        """
        Given a section header and a map of keywords, tries to identify and match
        the section with one of the predefined section types. If multiple different
        keywords appear in the header, the first one is always matched. If no
        keyword is found then the section is identified as other ('o').
        """
        head = head.lower().split()
        sec_id = 'o'
        for wrd in head:
            try:
                sec_id = keywords.value[wrd]
                break
            except KeyError:
                continue
        return sec_id

    return F.udf(section_identify_, spark_types.StringType())


def rouge_match(scorer):
    """UDF wrapper for rouge_match_"""
    def rouge_match_(cols):
        """
        Given the full text of a section and an array of summary sentences,
        computes the ROUGE-L score of each summary sentence with the section text.
        See the definition of 'rouge_targets' for more details.
        """
        full_section, summary_sents = cols.text_section, cols.abstract_text
        
        section_text = full_section[1]
        sum_scores = rouge_targets(summary_sents, section_text, scorer)
        
        return sum_scores
    return F.udf(rouge_match_, spark_types.ArrayType(spark_types.FloatType()))


def summary_match(col):
    """
    Given an array with the score arrays of each summary sentence we match
    each summary sentence with the section that has the highest similarity score.
    """
    scores = np.array(col)
    max_idx = np.argmax(scores, axis=0)
    return max_idx.tolist()


def index_array(col):
    """
    Assemble an array of (head, text) tuples into an array of
    {"section_head": head, "section_text": text, "section_idx": i}
    """
    indexed_text = [{"section_head": h, "section_text": t, "section_idx": i} for i, (h, t) in enumerate(col)]
    return indexed_text


def collect_summary(cols):
    """Select the summary sentences that are matched with a given section into an array of sentences"""
    section_idx, matched_summaries = cols.section_idx, cols.matched_summaries
    collected_summary = [t for (t, s_idx) in matched_summaries if s_idx == section_idx]
    return collected_summary


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, help="")
    parser.add_argument("--task", type=str, help="")
    parser.add_argument("--driver_memory", type=str, default="16g", help="")
    parser.add_argument("--partitions", type=int, default=500, help="")

    args, unknown = parser.parse_known_args()
    return args, unknown


def main():
    args, unknown = read_args()

    train_data = os.path.join(args.data_root, 'train.txt')
    val_data = os.path.join(args.data_root, 'val.txt')
    test_data = os.path.join(args.data_root, 'test.txt')
    selected_section_types = ["i", "m", "r", "l", "c"]

    metrics = ['rouge1', 'rouge2', 'rougeL']
    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)

    conf = pyspark.SparkConf()
    conf.set('spark.driver.memory', args.driver_memory)
    sc = pyspark.SparkContext(conf=conf)
    spark = pyspark.sql.SparkSession(sc)

    data_prefixes = ['train', 'val', 'test']
    data_paths = [train_data, val_data, test_data]
    task_output_dir = os.path.join(args.data_root, "processed", args.task)
    if not os.path.exists(task_output_dir):
        os.makedirs(task_output_dir)

    summary_match_udf = F.udf(summary_match, spark_types.ArrayType(spark_types.IntegerType()))
    index_array_udf = F.udf(
        index_array,
        spark_types.ArrayType(
            spark_types.StructType([
                spark_types.StructField(
                    'section_head', spark_types.StringType()),
                spark_types.StructField(
                    'section_text', spark_types.ArrayType(spark_types.StringType())),
                spark_types.StructField(
                    'section_idx', spark_types.IntegerType())])))
    collect_summary_udf = F.udf(
        collect_summary,
        spark_types.ArrayType(spark_types.StringType()))

    for data_path, prefix in zip(data_paths, data_prefixes):
        df = spark.read.json(data_path) \
            .repartition(args.partitions, "article_id")

        b_keywords = sc.broadcast(KEYWORDS)
        df = df.withColumn(
            'zipped_text',
            F.arrays_zip(F.col('section_names'), F.col('sections'))) \
            .withColumn(
            "text_section",
            F.explode("zipped_text")) \
            .withColumn(
            "summary_scores",
            rouge_match(scorer)(F.struct(F.col('text_section'), F.col('abstract_text')))) \
            .groupby(["abstract_text", "article_id"]) \
            .agg(
            F.collect_list("summary_scores").alias("summary_scores"),
            F.collect_list("text_section").alias("full_text_sections")) \
            .withColumn(
            "matched_summaries",
            summary_match_udf("summary_scores")) \
            .withColumn(
            "full_text_sections",
            index_array_udf("full_text_sections")) \
            .withColumn(
            "matched_summaries",
            F.arrays_zip(F.col("abstract_text"), F.col("matched_summaries"))) \
            .select(
            F.explode(F.col("full_text_sections")).alias("full_text_section"),
            F.col("full_text_section").section_head.alias("section_head"),
            F.col("full_text_section").section_idx.alias("section_idx"),
            F.col("matched_summaries"),
            "abstract_text",
            "article_id") \
            .withColumn(
            "section_summary",
            collect_summary_udf(F.struct(F.col("section_idx"), F.col("matched_summaries")))) \
            .where(
            F.size(F.col("section_summary")) > 0) \
            .withColumn(
            'section_id',
            section_identify(b_keywords)('section_head')) \
            .where(
            F.col("section_id").isin(selected_section_types)) \
            .withColumn(
            "document",
            F.concat_ws(" ", F.col("full_text_section").section_text)) \
            .withColumn(
            "summary",
            F.concat_ws(" ", F.col("section_summary"))) \
            .withColumn(
            "abstract",
            F.concat_ws(" ", F.col("abstract_text"))) \
            .withColumn(
            "summary",
            F.regexp_replace("summary", "<\/?S>", "")) \
            .withColumn(
            "abstract",
            F.regexp_replace("abstract", "<\/?S>", "")) \
            .withColumn(
            "document_len",
            F.size(F.split(F.col("document"), " "))) \
            .withColumn(
            "summary_len",
            F.size(F.split(F.col("summary"), " "))) \
            .where(
            F.col('document_len') > 50) \
            .select(
            "article_id",
            "section_id",
            "document",
            "summary",
            "abstract")

        if prefix not in ['val', 'test']:
            df = df.where(
                F.col('summary_len') > 50)

        df.write.json(
            path=os.path.join(task_output_dir, prefix),
            mode="overwrite")

        print(f"Finished writing {prefix} split to {task_output_dir}")


if __name__ == "__main__":
    main()
