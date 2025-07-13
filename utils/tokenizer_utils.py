import os
import pickle
from transformers import AutoTokenizer
from transformers import GPT2Tokenizer

def load_tokenizer(data_dir):
    """
    Load tokenizer given the data directory path.

    Args:
        data_dir (str): The directory path where the tokenizer data is stored.

    Returns:
        tokenizer (Tokenizer): The loaded tokenizer object.

    Raises:
        NotImplementedError: If stoi/itos is not supported or found.

    """
    meta_path = os.path.join(data_dir, 'meta.pkl')
    load_meta = os.path.exists(meta_path)
    if load_meta:
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        if meta.get("custom_tokenizer", False):
            print(f"Loading custom tokenizer from {data_dir}")
            tokenizer = AutoTokenizer.from_pretrained(data_dir, use_fast=False)
        else:
            if meta.get("stoi", False):
                raise NotImplementedError("stoi/itos not supported yet")
            else:
                raise NotImplementedError("No stoi/itos found")
    else:
        print("No meta.pkl found, using default GPT-2 tokenizer")
        tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

    if not tokenizer.eos_token:
        tokenizer.add_special_tokens({"eos_token": "</s>"})
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left" #Add if needed?
    return tokenizer
