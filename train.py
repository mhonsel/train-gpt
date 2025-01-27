import math
import os
import tiktoken
import time
import torch
import torch.distributed as dist

from dataloader import DataLoaderLite
from hellaswag import render_example, iterate_examples, get_most_likely_row
from model import GPT, GPTConfig, generate
from pathlib import Path
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP


# -----------------------------------------------------------------------------
# hugging face
use_huggingface = True
dataset_repo = "mhonsel/edu_fineweb10B_tokens"
dataset_dir = "./edu_fineweb10B/"
model_repo = "mhonsel/gpt2_124M"
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# hyperparameters
total_batch_size = 524288  # 2**19, ~0.5M in number of tokens
B = 64  # micro batch size (per GPU, make as big as GPU can handle)
T = 1024  # sequence length
vocab_size = 50304 # increase vocab_size to make a nice number
log_dir = "log"
resume_training = False
use_compile = True
max_lr = 6e-4 * 2  # gpt-3 paper: 6e-4
min_lr = max_lr * 0.1
warmup_steps = 200  # gpt-3 paper: 715
max_steps = 19073
eval_interval = 250
validation_steps = 20
hellaswag_eval = True
checkpoint_interval = 5000
# -----------------------------------------------------------------------------


# simple launch:
# python train.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train.py


# need generate hf token using `huggingface-cli login` before running train.py
if use_huggingface:
    from huggingface_hub import HfApi
    from hf_data import download, upload

    hf_api = HfApi()
    futures = []
    token = Path.home() / '.cache/huggingface/token'
    assert token.exists(), "Run `huggingface-cli login` to generate token first"


# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # autodetect device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.mps.is_available():
        device = 'mps'
    print(f'using device: {device}')

device_type = 'cuda' if device.startswith('cuda') else 'cpu'

if device_type != 'cuda':
    total_batch_size = B * T  # DEBUG: print each step

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
elif torch.mps.is_available():
    torch.mps.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

assert total_batch_size % (B * T * ddp_world_size) == 0, 'make sure total_batch_size is divisible by B * T * ddp_world_size'
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f'total desired batch size: {total_batch_size}')
    print(f'=> calculated gradient accumulation steps: {grad_accum_steps}')

# download dataset from huggingface
if use_huggingface:
    if master_process:
        download(dataset_repo=dataset_repo, dataset_dir=dataset_dir)
    if ddp:
        dist.barrier()  # ensure all processes wait for completion of the download

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train', verbose=master_process)
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val', verbose=master_process)

torch.set_float32_matmul_precision('high')  # only on cuda: reduce internal precision of matmul

# Create a log directory for writing checkpoints and logging
log_dir = Path(log_dir)
log_dir.mkdir(exist_ok=True)
log_file = log_dir / 'log.txt'

# Start / Resume Training
if resume_training:
    # get latest checkpoint file
    checkpoint_files = [f for f in log_dir.iterdir() if f.startswith("model_") and f.endswith(".pt")]
    assert len(checkpoint_files) > 0, "no checkpoints found"
    checkpoint_files = sorted(checkpoint_files)
    last_checkpoint = checkpoint_files[-1]
    checkpoint_path = log_dir / last_checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # load model state
    model = GPT(checkpoint['config'])
    model.to(device)
    model.load_state_dict(checkpoint['model'])
    # load optimizer state
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type, verbose=master_process)
    optimizer.load_state_dict(checkpoint['optimizer'])
    # load step (which will also load learning rate)
    current_step = checkpoint['step'] + 1
    # load training data state
    train_loader.set(checkpoint['train_loader'])
    if master_process:
        print(f"resuming training from step {current_step} with a validation loss of {checkpoint['val_loss']:.4f}")
else:
    # create model
    model = GPT(GPTConfig(vocab_size=vocab_size))
    model.to(device)
    current_step = 0
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type, verbose=master_process)
    # clear the log file
    with open(log_file, "w") as f: # open for writing to clear the file
        pass

def unwrap_model(model):
    # Unwrap DDP
    if hasattr(model, 'module'):
        model = model.module
    # Unwrap torch.compile
    if hasattr(model, '_orig_mod'):
        model = model._orig_mod
    return model

if use_compile and device_type == 'cuda':
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# Use the unwrap_model function to get the raw model
raw_model = unwrap_model(model)

def get_lr(step):
    # 1) linear warmup for warmup_steps steps
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # 2) if step > max_steps, return min learning rate
    if step > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# training loop
for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # evaluate validation loss
    if step % eval_interval == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            for _ in range(validation_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                val_loss_accum += loss.detach()
            val_loss_accum /= validation_steps # diff from Andrej - less prone to floating point errors
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f'validation loss: {val_loss_accum.item():.4f}')
            with open(log_file, 'a') as f:
                f.write(f'{step} val {val_loss_accum.item():.4f}\n')
            if step > 0 and (step % checkpoint_interval == 0 or last_step):
                # write model checkpoints
                train_loader_checkpoint = {'current_shard': train_loader.current_shard,
                                           'current_position': train_loader.current_position}
                checkpoint_path = log_dir / f'model_{step:05d}.pt'
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item(),
                    'train_loader': train_loader_checkpoint,

                }
                torch.save(checkpoint, checkpoint_path)
                if use_huggingface:
                    futures.append(
                        upload(api=hf_api, file_path=checkpoint_path, model_repo=model_repo)
                    )

    # evaluate hellaswag
    if hellaswag_eval and (step % eval_interval == 0 or last_step):
        num_correct_norm = 0
        num_total = 0
        # Use unwrapped (uncompiled) model for evaluation
        model_for_eval = unwrap_model(model)
        model_for_eval.eval()
        for i, example in enumerate(iterate_examples('val')):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model_for_eval(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    # once in a while generate from the model (except step 0, which is noise)
    if (step > 0 and step % eval_interval == 0) or last_step:
        samples = generate(model=model, encoder=enc, prompt="Hello, I'm a language model,",
                           seed=(42 + ddp_rank), device=device)
        for i, sample in enumerate(samples):
            print(f"rank {ddp_rank} sample {i}: {sample}")

    # do one step of the optimization
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # added after video, this field is also used by the forward pass.
        # we want gradient sync/average between processes only on last accumulation step
        # for efficiency; official way is to use context manager: with model.no_sync():
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0  # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step: {step:5d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/s: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if use_huggingface and master_process:
    for future in futures:
        future.result()  # wait for upload to complete

if ddp:
    destroy_process_group()
