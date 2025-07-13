import os
import pickle
import tiktoken
from transformers import AutoTokenizer
import torch
from contextlib import nullcontext


def sample_from_model(model, data_dir, out_dir,
                      dtype, device = 'cuda',
                      num_samples=2, temperature=0.8,
                      max_new_tokens=75, top_k=200,
                      prompt_type = "text", start_text="",
                      iter_num = None):
    
    """
    Generate samples from the given model.

    model (object): The model object to generate samples from.
    num_samples (int): The number of samples to generate. 
    temperature (float): The temperature value for sampling 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    max_new_tokens (int): The maximum number of new tokens to generate.
    top_k (int): The number of top-k tokens to consider during sampling.

    Returns:
    list: A list of generated samples.

    """

    device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model.eval()

    # look for the meta pickle in case it is available in the dataset folder
    meta_path = os.path.join(data_dir, 'meta.pkl')
    load_meta = os.path.exists(meta_path)
    if load_meta:
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        if meta.get("custom_tokenizer", False):
            custom_tokenizer = AutoTokenizer.from_pretrained(data_dir, use_fast=False)
            encode = lambda s: custom_tokenizer.encode(s)
            decode = lambda l: custom_tokenizer.decode(l)
        else:
            if meta.get("stoi", False):
                stoi, itos = meta['stoi'], meta['itos']
                encode = lambda s: [stoi[c] for c in s]
                decode = lambda l: ''.join([itos[i] for i in l])
            else:
                pass
    else:
        # ok let's assume gpt-2 encodings by default
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    # encode the beginning of the prompt

    if prompt_type == "text":
        prompt_list = [("plain text", start_text)]
    elif prompt_type == "elaborate":
        prompt_cats = {
            "story": "Once upon a time,",
            "directed_speech": "He said,",
            "conversation": "A: Hello, B: Hi,",
            "news_article": "Breaking News: Scientists have discovered",
            "knowledge_base": "The capital of ",
            "poetry": "Roses are red,",
        }
        prompt_list = [(k, v) for k, v in prompt_cats.items()]


    for fw_name, start in prompt_list:
        start_ids = encode(start)
        x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
        write_path =  os.path.join(out_dir, f"sample_{fw_name}_{iter_num}.txt")
        write_list = []
        # run generation
        with torch.no_grad():
            with ctx:
                for k in range(num_samples):
                    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)

                    write_list.append(decode(y[0].tolist()))
        with open(write_path, 'w') as f:
            f.write("\n \n ---------------------------------------------------- \n \n ".join(write_list))

    # Change the model back to training mode
    model.train()


