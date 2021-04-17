# DANCER-summ
This repository contains code for the paper [A Divide-and-Conquer Approach to the Summarization of Long Documents](https://arxiv.org/abs/2004.06190), which was published in the IEEE/ACM Transactions on Audio, Speech, and Language Processing.

## Setup

## Data
The data used in the experiments are scientific papers obtained from the ArXiv and PubMed OpenAccess repositories. We are using the datasets created and open sourced by [Cohan et al. (2018)](https://arxiv.org/abs/1804.05685).

### Get the data
ArXiv dataset: [Download](https://drive.google.com/file/d/1b3rmCSIoh6VhD4HKWjI4HOW-cSwcwbeC/view?usp=sharing) ([mirror](https://archive.org/download/armancohan-long-summarization-paper-code/arxiv-dataset.zip))
PubMed dataset: [Download](https://drive.google.com/file/d/1lvsqvsFi3W-pE1SqNZI0s8NR9rC1tsja/view?usp=sharing) ([mirror](https://archive.org/download/armancohan-long-summarization-paper-code/pubmed-dataset.zip))

*Note: The dataset is also available on [Tensorflow Datasets](https://www.tensorflow.org/datasets/catalog/scientific_papers) but since it's structure is not preserved it is not suitable for DANCER.*

### Preprocessing
Since the datasets are rather large, we are using PySpark to do the preprocess (don't worry we won't be running in cluster mode, so you can still run it on your local machine). We recommend that you use a machine with at least 4 cores and 16GBs of memory. More resources will make you preprocessing much faster.

To run the full preprocessing you can run ```sh run_data_prep.sh``` modifying the following configurations.
```
--data_root: path to your download .txt files
--task: one of "pubmed" and "arxiv"
--driver_memory: tune based on your machine (e.g. for a 16Gi machine, 12Gi should be fine)
```

Pyspark will write each split into many partitions. You need to merge them in one file by running (replacing split with train, val and test):
```
cat /path/to/data/split/part-* > split.json
```