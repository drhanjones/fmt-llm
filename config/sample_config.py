torch_seed_default = 1337

# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume'

wandb_run_id = None #Wandb expect run id from config when resume else fail

# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# experimenting with head size in qkv matrices
head_size_qkv = None
ffw_dim = None

wm_mask = False # whether to use a mask for the attention layer
wm_decay_length = None #setting to none and after configurator is run, if config doesn't provide a value, set to block_size
wm_decay_rate = 2 # how fast to decay the mask
wm_decay_type = "power_law" # Type of decay to apply to the matrix #linear, exponential, inverse_sigmoid, custom_logistic
wm_decay_echoic_memory = 1 #Echoic memory for the decay matrix, first n values where "effect of decay" is not applied, where memory is supposedly perfect


# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# SAMPLING Settings

save_sample_to_file = False # if True, save a sample to file after each eval, overwrite in config
sampling_frequency = 5000 # how often to sample from the model, overwrite in config

# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.

# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' # if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the 


# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
