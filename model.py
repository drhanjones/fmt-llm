import math
import inspect
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.head_size_qkv % config.n_head == 0 #started explicitly passing head_size_qkv

        self.head_size_qkv = config.head_size_qkv
        ## key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * self.head_size_qkv, bias=config.bias)

        # output projection
        self.c_proj = nn.Linear(self.head_size_qkv, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.wm_mask = config.wm_mask
        self.wm_decay_length = config.wm_decay_length
        self.wm_decay_rate = config.wm_decay_rate
        self.wm_decay_type = config.wm_decay_type
        self.wm_decay_echoic_memory = config.wm_decay_echoic_memory
        self.wm_setting_type = config.wm_setting_type
        
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if self.wm_mask:
            print("Setting flash to False because wm_mask is enabled")
            self.flash = False 

        if not self.flash:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def get_decay_weight_matrix(self, n, decay_length, decay_factor=None, decay_type=None, decay_echoic_memory=1):

        """
        Get a decay weight matrix for a given context window size n.
        :param n: Context window size for the decay matrix
        :param decay_length: Actual length over which decay happens (usually is supposed to be equal to or greater than n)
        :param decay_factor: Decay factor for the decay matrix, analogous to weight of the decay
        :param decay_type: Type of decay to apply to the matrix #linear, exponential, inverse_sigmoid, custom_logistic
        :param decay_echoic_memory: Echoic memory for the decay matrix, first n values where "effect of decay" is not applied, where memory is supposedly perfect
        :return:
        """

        if decay_type == "power_law":
            decay_length = decay_length - decay_echoic_memory + 1
            nums = torch.linspace(0, 1, decay_length)
            decay_values = 1 - torch.pow(nums, 1 / (np.e * decay_factor))
            decay_values = torch.cat((torch.ones(decay_echoic_memory - 1), decay_values))
            decay_length = decay_length + decay_echoic_memory - 1

            if decay_length != n:
                assert decay_length > n
                decay_values = decay_values[:n]

        indices = torch.arange(n)[:, None] - torch.arange(n)
        lower_triangle = torch.tril(decay_values[indices])
        lower_triangle = lower_triangle.float()

        return lower_triangle

    def forward(self, x):
        # batch size, sequence length, embedding dimensionality (n_embd)
        # B is batch size, T is sequence length, C is embedding dimensionality
        B, T, C = x.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # Calculate where k,q,v for each batch for each head is of the size T,Hs where Hs = C // n_head
        #Hs is the size of the head
        q, k, v = self.c_attn(x).split(self.head_size_qkv, dim=2) # (B, T, 3*hs) -> 3 * (B, T, hs)
        k = k.view(B, T, self.n_head, self.head_size_qkv // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_size_qkv // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_size_qkv // self.n_head).transpose(1, 2)  # (B, nh, T, hs)


        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T) (nh = number of heads, hs = head size)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention

            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) #Att shape: (B, nh, T, T) ?

            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)

            if self.wm_mask:
                wm_mask = self.get_decay_weight_matrix(T, self.wm_decay_length, decay_factor=self.wm_decay_rate, decay_type=self.wm_decay_type,
                                                        decay_echoic_memory=self.wm_decay_echoic_memory).to(x.device)
                assert att.shape[-2:]==wm_mask.shape[-2:]

                att = att * wm_mask

            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, self.head_size_qkv) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, config.ffw_dim, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(config.ffw_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

    head_size_qkv: int = None
    ffw_dim: int = None

    wm_mask: bool = False
    wm_decay_length: int = 1024
    wm_decay_rate: int = 1
    wm_decay_type: str = "power_law"
    wm_decay_echoic_memory: int = 1


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))


    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """

        def test_count_parameters(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, hidden_states=False):

        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (f"Cannot "
                                             f" sequence of length {t}, block size is only {self.config.block_size}")
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        hidden_states_list = [x] if hidden_states else None
        for block in self.transformer.h:
            x = block(x)
            if hidden_states:
                hidden_states_list.append(x)

        x = self.transformer.ln_f(x)
        if targets is not None:

            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)

            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        if hidden_states:
            return logits, loss, hidden_states_list
        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, do_sample=True):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        #Enable better batch decoding but until then loop through each sequenc

        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            if do_sample:
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                # Added option to do greedy decoding instead of sampling because evaluation needs it for some reason
                #Defaults do_sample to True though so it should be fine
                #Greedy decoding
                idx_next = torch.argmax(probs, dim=-1, keepdim=True)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
