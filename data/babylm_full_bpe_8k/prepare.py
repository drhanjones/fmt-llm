import os
import requests
import tiktoken
import numpy as np
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer)
from transformers import PreTrainedTokenizerFast
from tokenizers.normalizers import Lowercase, Strip, StripAccents, NFD
from transformers import AutoTokenizer

from pathlib import Path
import pickle

dataset_folder_path = Path(__file__).parents[3] / 'babylm_data'
train_ds = 'babylm_10M'
dev_ds = 'babylm_dev'
dataset_name = "full"

textfiles = [f"{dataset_folder_path}/{train_ds}/aochildes.train",
             f"{dataset_folder_path}/{train_ds}/bnc_spoken.train",
             f"{dataset_folder_path}/{train_ds}/cbt.train",
             f"{dataset_folder_path}/{train_ds}/children_stories.train",
             f"{dataset_folder_path}/{train_ds}/gutenberg.train",
             f"{dataset_folder_path}/{train_ds}/open_subtitles.train",
             f"{dataset_folder_path}/{train_ds}/qed.train",
             f"{dataset_folder_path}/{train_ds}/simple_wikipedia.train",
             f"{dataset_folder_path}/{train_ds}/switchboard.train",
             f"{dataset_folder_path}/{train_ds}/wikipedia.train"]





if dataset_name == "full":
    train_file_path = [ os.path.join(dataset_folder_path, train_ds, x) for x in os.listdir(os.path.join(dataset_folder_path, train_ds)) if x.endswith(".train")]
    dev_file_path = [ os.path.join(dataset_folder_path, dev_ds, x) for x in os.listdir(os.path.join(dataset_folder_path, dev_ds)) if x.endswith(".dev")]


else:
    train_file_path = [os.path.join(dataset_folder_path, train_ds, dataset_name + '.train')]
    dev_file_path = [os.path.join(dataset_folder_path, dev_ds, dataset_name + '.dev')]


train_data = ""
for file in train_file_path:


    with open(file, 'r') as f:
        train_data += f.read()
        train_data += "\n"

val_data = ""

for file in dev_file_path:
    with open(file, 'r') as f:
        val_data += f.read()
        val_data += "\n"




meta = {}

def encode_with_tokenizer(data, textfiles,  tokenizer_type = "customBPE", vocab_size = None, tokenizer_scratch = False, traintokenizer_1ds = False):

    if tokenizer_type == "customBPE":
        meta["custom_tokenizer"] = True


        tokenizer_exists = True if os.path.exists("tokenizer.json") else False

        if tokenizer_scratch or not tokenizer_exists:

            tokenizer = Tokenizer(models.BPE())
            tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), Strip(), StripAccents()])
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<|endoftext|>"])

            if traintokenizer_1ds:
                textfiles = [train_file_path]

            tokenizer.train(files=textfiles, trainer=trainer)

            tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
            tokenizer.decoder = decoders.ByteLevel()

            # Save the tokenizer
            wrapped_tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer,
                bos_token="<|endoftext|>",
                eos_token="<|endoftext|>")

            print("Saving tokenizer to", Path(__file__).parents[0])
            wrapped_tokenizer.save_pretrained(Path(__file__).parents[0])
            tokenizer = AutoTokenizer.from_pretrained(Path(__file__).parents[0])

        else:
            tokenizer = AutoTokenizer.from_pretrained(Path(__file__).parents[0])

        print("Tokenizer vocab size:", tokenizer.vocab_size)
        data_ids = tokenizer.encode(data)
        meta["vocab_size"] = tokenizer.vocab_size

    elif tokenizer_type == "gpt2":
        meta["custom_tokenizer"] = False
        enc = tiktoken.get_encoding("gpt2")
        data_ids = enc.encode_ordinary(data)

    return data_ids


# encode with tiktoken gpt2 bpe
train_ids = encode_with_tokenizer(train_data, textfiles, tokenizer_type = "customBPE", vocab_size = 8000)
val_ids = encode_with_tokenizer(val_data, textfiles, tokenizer_type =  "customBPE", vocab_size = 8000)

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)


train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
