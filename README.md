# Human-like fleeting memory improves language learning but impairs reading time prediction in transformer language models

## Introduction
This repo contains the PyTorch implementation of the Fleeting Memory Transformer, and code to execute all the training runs and experiments from the paper “Human-like fleeting memory improves language learning but impairs reading time prediction in transformer language models", by Thamma & Heilbron  

## Acknowledgements

The codebase builds on [Andrej Karpathy's NanoGPT](https://github.com/karpathy/nanoGPT). 


## Prerequisites
To run this code, you need to have Python 3.9. 

Install the following dependencies:
- [PyTorch](https://pytorch.org) (v2.2)
- [wandb](https://wandb.ai/site) (v0.16.2)
- [Transformers](https://huggingface.co/docs/transformers/index) (v4.37.2) 
- [Datasets](https://huggingface.co/docs/datasets/index) (v2.16.1)



## Quickstart

To directly run the training script with default parameters and a single GPU, you can use -

```bash
python train.py 
```

Alternatively, you can pass the parameters as command line arguments or pass the config file as an argument. 
```bash
python train.py config/train_config.yaml
```
or
```bash
python train.py --batch_size 32
```

To run with DDP on multiple GPUs- 

```bash
torchrun --standalone --nproc_per_node=4 train.py config=config/sample_config.py
```
To run on a slurm cluster, you can modify and use the `run.sh` script.

```bash
sbatch --export=config_file_name=config/sample_config.py run.sh
```

## Dataset
The dataset and instructions for downloading or preparing it are located in the [`data/readme.md`](data/readme.md) file. Please refer to that file for details.



## Config parameters
The config parameters for training are located in the [`config/readme.md`](config/readme.md) file. The parameters can be passed as command line arguments or through a config file.

