import time
import platform
import os


# wandb logging
dataset = 'babylm_full_bpe_8k'
wandb_log = True 
wandb_project = 'project_name'
sysname = "local" if "pop-os" in platform.node() else "server"
 
save_sample_to_file = True
sampling_frequency = 10000

wm_mask = True
wm_decay_rate = 2
wm_decay_type = "power_law"
wm_decay_echoic_memory = 5

n_layer = 6
n_head = 6

if wm_mask:
    mask_part = "mask_"
    if wm_decay_type == "power_law":
        mask_part += f"pl{wm_decay_rate:03}".replace(".", "p")
    mask_part += f"_em{wm_decay_echoic_memory:02}"

else:
    mask_part = "nomask"

lay_x_head = f'{n_layer}x{n_head}'
torch_seed_default = 42

unique_id = os.environ.get("SLURM_JOB_ID", str(int(time.time()))) #Using SLURM_JOB_ID as unique id else use time

out_dir = f'output/out-{dataset}-{lay_x_head}-{mask_part}-{unique_id}'
wandb_run_name = f'{dataset}_{lay_x_head}_{mask_part}_gpt2_{sysname}_run_{unique_id}'



eval_interval = 250 
eval_iters = 200
log_interval = 10

always_save_checkpoint = True

gradient_accumulation_steps = 8
batch_size = 32 
block_size = 256
n_embd = 384
dropout = 0.1
learning_rate = 5e-4
max_iters = 44000
lr_decay_iters = 44000
min_lr = 5e-5 # learning_rate / 10 usually


warmup_iters = 100 
weight_decay = 1e-1
