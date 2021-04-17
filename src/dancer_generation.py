import os
import random
import argparse
from tqdm import tqdm

import torch

import scoring
import loaders


def generate_summaries(test_loader, args, device):
    """Run summary generation for a given DataLoader"""
    model, tokenizer = loaders.load_model(args, device=device)
    
    gen_sums = []
    target_sums = []
    article_ids = []
    section_ids = []
    abstracts = []
    for i, batch in enumerate(tqdm(test_loader)):
        model_inputs = tokenizer(
            batch[args.text_column],
            max_length=args.max_source_length,
            truncation=True,
            padding=True,
            return_tensors='pt')
        
        input_ids = model_inputs['input_ids'].to(device)
        sent_outputs = model.generate(
            input_ids,
            num_beams=args.num_beams,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True)  # only one beam should be equivalent to greedy,
        gen_sum = [
            tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in sent_outputs["sequences"]]

        gen_sums += gen_sum
        target_sums += batch[args.summary_column]
        try:
            article_ids += batch["article_id"]
            section_ids += batch["section_id"]
            abstracts += batch["abstract"]
        except KeyError:
            article_ids += [i]
            pass
        
    return gen_sums, target_sums, article_ids, section_ids, abstracts


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="")
    parser.add_argument("--model_path", type=str, help="")
    parser.add_argument("--output_path", type=str, help="")
    parser.add_argument("--dataset_name", type=str, help="")
    parser.add_argument("--dataset_config_name", type=str, help="")
    parser.add_argument("--data_path", type=str, help="")
    parser.add_argument("--text_column", type=str, help="")
    parser.add_argument("--summary_column", type=str, help="")

    parser.add_argument("--tokenizer_name", type=str, help="")
    parser.add_argument("--max_source_length", type=int, default=512, help="")
    parser.add_argument("--max_summary_length", type=int, default=128, help="")
    parser.add_argument("--max_test_samples", type=int, help="")
    parser.add_argument("--write_rouge", type=int, default=0, help="")
    parser.add_argument("--seed", type=int, default=10, help="")
    parser.add_argument("--test_batch_size", type=int, default=2, help="")
    parser.add_argument("--num_beams", type=int, default=3, help="")

    args, unknown = parser.parse_known_args()

    return args, unknown


def main():
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    args, unknown = read_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    write_rouge = bool(args.write_rouge)

    select_sections = ["i", "m", "r", "c"]
    print(f"Mode: {args.mode}")
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    out_path = os.path.join(args.output_path, "generations")
    test_loader = loaders.init_loader(args)
    gen_sums, target_sums, article_ids, section_ids, abstracts = generate_summaries(test_loader, args, device=device)
    
    print("Scoring generated summaries")
    if args.mode == "dancer":
        metrics = scoring.score_dancer(
            gen_sums=gen_sums,
            target_sums=abstracts,
            article_ids=article_ids,
            section_ids=section_ids,
            out_path=out_path,
            select_sections=select_sections,
            write_gens=write_rouge)
    else:
        metrics = scoring.score_standard(
            gen_sums=gen_sums,
            target_sums=target_sums,
            article_ids=article_ids,
            out_path=out_path,
            write_gens=write_rouge)

    if write_rouge:
        scores_dict = scoring.score_outputs(out_path)
        scoring.rouge_log(scores_dict, out_path)
    else:
        print(metrics)


if __name__ == "__main__":
    main()
