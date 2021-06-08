# DANCER-summ
This repository contains code for the paper [A Divide-and-Conquer Approach to the Summarization of Long Documents](https://arxiv.org/abs/2004.06190),
which was published in the IEEE/ACM Transactions on Audio, Speech, and Language Processing.

_Note: The model implementation, initial model checkpoints and ROUGE implementations
are different from the original published paper, so the results might be a bit different._

## Data
The data used in the experiments are scientific papers obtained from
the ArXiv and PubMed OpenAccess repositories. We are using the datasets
created and open sourced by [Cohan et al. (2018)](https://arxiv.org/abs/1804.05685).

### Get the data
ArXiv dataset: [Download](https://drive.google.com/file/d/1b3rmCSIoh6VhD4HKWjI4HOW-cSwcwbeC/view?usp=sharing) ([mirror](https://archive.org/download/armancohan-long-summarization-paper-code/arxiv-dataset.zip))
PubMed dataset: [Download](https://drive.google.com/file/d/1lvsqvsFi3W-pE1SqNZI0s8NR9rC1tsja/view?usp=sharing) ([mirror](https://archive.org/download/armancohan-long-summarization-paper-code/pubmed-dataset.zip))

*Note: The dataset is also available on [Tensorflow Datasets](https://www.tensorflow.org/datasets/catalog/scientific_papers) but since it's structure is not preserved it is not suitable for DANCER.*

### Preprocessing
Since the datasets are rather large, we are using PySpark to do
the preprocess (don't worry we won't be running in cluster mode,
so you can still run it on your local machine).
We recommend that you use a machine with at least 4 cores and 16GBs
of memory. More resources will make you preprocessing much faster.

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

## Training DANCER
To run a full training session with DANCER you can run ```run_dancer_training.sh```
modifying the following configurations.
```
--model_name_or_path: initial model name or path
--tokenizer_name: tokenizer name or path
--train_file: path to the training data file (created in the preprocessing step)
--validation_file: path to the validation data file (created in the preprocessing step)
--text_column: name of the input text column
--summary_column: name of the target summary column
--output_dir: path for output model artifacts
--logging_dir: path for training logs
--per_device_train_batch_size: training batch size
--per_device_eval_batch_size: evaluation batch size
--max_source_length: max number of input tokens
--max_target_length: max number of summary tokens for training
--val_max_target_length: max number of summary tokens for evaluation
--num_beams: number of beams for beam search decoding
--metric_for_best_model: metric used for checkpointing and early stopping
```

### Generating summaries
Once you have a trained DANCER model you can run summary generation running
```run_dancer_generation.sh``` modifying the following configurations.
```
--mode: dancer (use standard for non-DANCER models)
--model_path: path to the trained model artifacts
--output_path: path to ouput summaries
--data_path: path to the test data file (created in the preprocessing step)
--text_column: name of the input text column
--summary_column: name of the target summary column
--write_rouge: if True then summaries and ROUGE metrics are written (if False they are just printed)
--test_batch_size: generation batch size
--max_source_length: max number of input tokens
--max_target_length: max number of generated summary tokens
--num_beams: number of beams for beam search decoding
```

## How to Cite
If you extend or use this work, please consider citing the [paper](https://ieeexplore.ieee.org/abstract/document/9257174/):

```bibtex
@article{gidiotis2020divide,
  author={Gidiotis, Alexios and Tsoumakas, Grigorios},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={A Divide-and-Conquer Approach to the Summarization of Long Documents}, 
  year={2020},
  volume={28},
  number={},
  pages={3029-3040},
  doi={10.1109/TASLP.2020.3037401}
}
```
