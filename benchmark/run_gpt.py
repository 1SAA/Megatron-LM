import argparse
import gc
from functools import partial
import os
import sys
import time

import numpy as np

from megatron.utils import average_losses_across_data_parallel_group
from megatron.model import BertModel, GPTModel
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron import initialize_megatron, get_args, print_rank_0
from megatron.training import train_step, setup_model_and_optimizer
import torch

from utils import get_gpt_config, compute_gpt_parameter_count


def get_tflops(model_numel, batch_size, seq_len, step_time):
    return model_numel * batch_size * seq_len * 8 / 1e12 / (step_time + 1e-12)


def model_size_formatter(numel: int) -> str:
    GB_SIZE = 10**9
    MB_SIZE = 10**6
    KB_SIZE = 10**3
    if numel >= GB_SIZE:
        return f'{numel / GB_SIZE:.1f}B'
    elif numel >= MB_SIZE:
        return f'{numel / MB_SIZE:.1f}M'
    elif numel >= KB_SIZE:
        return f'{numel / KB_SIZE:.1f}K'
    else:
        return str(numel)


def get_gpt_functions():
    args = get_args()
    micro_batch_size = args.micro_batch_size
    seq_len = args.encoder_seq_length

    def model_provider(pre_process=True, post_process=True):
        model = GPTModel(num_tokentypes=0,
                         parallel_output=True,
                         pre_process=pre_process,
                         post_process=post_process)
        return model

    def loss_func(loss_mask, output_tensor):
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

        # Reduce loss for logging.
        #averaged_loss = average_losses_across_data_parallel_group([loss])
        averaged_loss = [0]
        return loss, {'lm loss': averaged_loss[0]}

    tokens = torch.ones((micro_batch_size, seq_len)).cuda().long()
    labels = torch.ones((micro_batch_size, seq_len)).cuda().long()
    loss_mask = torch.ones((micro_batch_size, seq_len)).cuda().int()
    attention_mask = \
        torch.ones(micro_batch_size, 1, seq_len, seq_len).cuda().bool()
    position_ids = torch.ones((micro_batch_size, seq_len)).cuda().long()

    def forward_step(data_iterator, model):
        output_tensor = model(tokens,
                              position_ids,
                              attention_mask,
                              labels=labels)
        return output_tensor, partial(loss_func, loss_mask)

    return model_provider, loss_func, forward_step


def benchmark_gpt_one_case(model_name, batch_size):
    # Model configs
    model_config = get_gpt_config(model_name)
    seq_len = model_config.get('seq_len')
    hidden_size = model_config.get('hidden_dim')
    num_layers = model_config.get('num_layers')
    num_heads = model_config.get('num_attention_heads')
    vocab_size = model_config.get('vocab_size')

    num_gpus = int(os.environ['WORLD_SIZE'])
    micro_batch_size = batch_size * num_gpus

    # Parallel configs
    # Initialize megatron
    sys.argv += ["--micro-batch-size", str(micro_batch_size)]
    sys.argv += ["--tensor-model-parallel-size", str(num_gpus)]
    sys.argv += ["--pipeline-model-parallel-size", "1"]
    sys.argv += ["--global-batch-size", str(micro_batch_size)]
    sys.argv += ["--num-layers", str(num_layers)]
    sys.argv += ["--hidden-size", str(hidden_size)]
    sys.argv += ["--num-attention-heads", str(num_heads)]
    sys.argv += ["--seq-length", str(seq_len)]
    sys.argv += ["--max-position-embeddings", str(seq_len)]
    sys.argv += ["--optimizer", "adam"]
    sys.argv += ["--train-iters", "4"]
    sys.argv += ["--lr", "0.00015"]
    sys.argv += ["--DDP-impl", "local"]
    sys.argv += ["--fp16"]
    sys.argv += ["--loss-scale", "8"]
    sys.argv += ["--checkpoint-activations"]
    sys.argv += ["--use-flash-attn"]
    # sys.argv += ["--no-masked-softmax-fusion"]
    # sys.argv += ["--no-async-tensor-model-parallel-allreduce"]
    # sys.argv += ["--no-scatter-gather-tensors-in-pipeline"]
    initialize_megatron()
    args = get_args()
    args.padded_vocab_size = vocab_size
    rank = torch.distributed.get_rank()

    # Check initialization
    assert mpu.get_data_parallel_world_size() == 1
    assert mpu.get_tensor_model_parallel_world_size() == num_gpus
    assert mpu.get_pipeline_model_parallel_world_size() == 1

    # Build model
    model_provider, loss_func, forward_step = get_gpt_functions()
    model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider, ModelType.encoder_or_decoder)
    parameter_count = compute_gpt_parameter_count(num_layers, hidden_size, vocab_size)
    print(f"the model size is {model_size_formatter(parameter_count)}")
    get_tflops_func = partial(get_tflops, parameter_count, micro_batch_size // num_gpus, seq_len)
    
    def run_func():
        train_step(forward_step, None, model, optimizer, lr_scheduler)

    # Warmup and reset timers
    run_func()
    # timers = get_timers()
    # names = list(timers.timers.keys())
    # for name in names:
    #     timers(name).reset()
        
    # Run benchmark
    cost = []
    for i in range(3):
        start = time.time()
        run_func()
        torch.cuda.synchronize()
        span = time.time() - start
        cost.append(span)
    
    cost = np.median(np.array(cost))
    print_rank_0(f'Median TFLOPS is {get_tflops_func(cost):.3f}')


if __name__ == "__main__":
    batch_size = int(sys.argv[-1])
    del sys.argv[-1]
    
    model_name = sys.argv[-1]
    del sys.argv[-1]

    benchmark_gpt_one_case(model_name, batch_size)
