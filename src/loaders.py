from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from datasets import load_dataset


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


def load_model(args, device):
    print(f"Loading tokenizer {args.tokenizer_name if args.tokenizer_name else args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_path)
    
    print(f"Loading model from {args.model_path}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_path).to(device)
    
    return model, tokenizer
