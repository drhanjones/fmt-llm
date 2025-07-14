The codebase has the following configuration parameters for training. 

- `baseline_config.py` is the configuration for baseline experiments
- `fleeting_memory_mask_config.py` is the configuration for fleeting memory mask experiments
- `fleeting_memory_mask_with_echoic_config.py` is the configuration for fleeting memory mask with echoic memory experiments


### Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| torch_seed_default| default torch seed | 1337|


### I/O
|Parameter | Description | Default Value |
|-----------|-------------|---------------|
|out_dir | output directory to save checkpoints and logs | 'out' |
|eval_interval | evaluation interval | 2000 |
|log_interval | logging interval | 1 |
|eval_iters | number of iterations to run evaluation for | 200 |
|eval_only | if True, script exits right after the first eval | False |
|always_save_checkpoint | if True, always save a checkpoint after each eval | True |
|init_from | 'scratch' or 'resume' | 'scratch' |
|wandb_run_id | Wandb expect run id from config when resume else fail | None |  
|save_sample_to_file | if True, save a sample to file after each eval | False |
|sampling_frequency | how often to sample from the model | 5000 |

### data
|Parameter | Description | Default Value |
|-----------|-------------|---------------|
|dataset | dataset to use for training. The dataset name must match with the name of the folder within the data directory | 'openwebtext' |
|gradient_accumulation_steps | used to simulate larger batch sizes | 5 * 8 |
|batch_size | if gradient_accumulation_steps > 1, this is the micro-batch size | 12 |
|block_size | sequence length | 1024 |


### model
|Parameter | Description | Default Value |
|-----------|-------------|---------------|
|n_layer | number of transformer blocks | 12 |
|n_head | number of attention heads | 12 |
|n_embd | size of the embedding vector | 768 |
|dropout | dropout rate | 0.0 |
|bias | whether to use bias in the linear layers | False |
|head_size_qkv | size of the head in the qkv matrices | None |
|ffw_dim | size of the feedforward layer | None |

### masking parameters
|Parameter | Description | Default Value |
|-----------|-------------|---------------|
|wm_mask | whether to use the fleeting memory mask in the attention layer | False |
|wm_decay_length | length of the decay matrix, or size over which to decay happens. if None, set to block_size | None |
|wm_decay_rate | how fast to decay the mask | 2 |
|wm_decay_type | type of decay to apply to the matrix | 'power_law' |
|wm_decay_echoic_memory | echoic memory for the decay matrix, first n values where "effect of decay" is not applied, where memory is supposedly perfect | 1 |

### adamw optimizer
|Parameter | Description | Default Value |
|-----------|-------------|---------------|
|learning_rate | max learning rate | 6e-4 |
|max_iters | total number of training iterations | 600000 |
|weight_decay | weight decay | 1e-1 |
|beta1 | beta1 for adamw | 0.9 |
|beta2 | beta2 for adamw | 0.95 |
|grad_clip | clip gradients at this value, or disable if == 0.0 |

### learning rate decay settings
|Parameter | Description | Default Value |
|-----------|-------------|---------------|
|decay_lr | whether to decay the learning rate | True |
|warmup_iters | how many steps to warm up for | 2000 |
|lr_decay_iters | should be ~= max_iters per Chinchilla | 600000 |
|min_lr | minimum learning rate, should be ~= learning_rate/10 per Chinchilla | 6e-5 |


### DDP and system settings
|Parameter | Description | Default Value |
|-----------|-------------|---------------|
|backend | DDP backend to use, 'nccl' or 'gloo' | 'nccl' |
|device | device to run the model on, 'cpu', 'cuda', etc | 'cuda' |
|dtype | data type to use, 'float32', 'bfloat16', or 'float16' | 'bfloat16' if cuda is available and supports bfloat16, else

### wandb logging

|Parameter | Description | Default Value |
|-----------|-------------|---------------|
|wandb_log | whether to log to wandb | False |
|wandb_project | wandb project name | 'owt' |
|wandb_run_name | wandb run name | 'gpt2' |
