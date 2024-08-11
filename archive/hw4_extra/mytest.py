import sys
sys.path.append('./python')
sys.path.append('./apps')
import numpy as np
import pytest
import torch
import itertools
import mugrade
import os

import needle as ndl
import needle.nn as nn

from simple_ml import *
from models import LanguageModel

def test_attention_layer(batch_size, seq_len, input_dim, num_head, dim_head, causal, dropout, device):

    np.random.seed(19943)

    q = np.random.randn(
        batch_size, seq_len, input_dim
    ).astype(np.float32)
    k = np.random.randn(
        batch_size, seq_len, input_dim
    ).astype(np.float32)
    v = np.random.randn(
        batch_size, seq_len, input_dim
    ).astype(np.float32)

    layer = nn.AttentionLayer(
        input_dim, num_head, dim_head, 
        dropout=dropout, causal=causal, device=device)

    result = layer(
        ndl.Tensor(q, device=device),
        ndl.Tensor(k, device=device),
        ndl.Tensor(v, device=device),
    )

    result = result.numpy()
    
        
    current_input_id = "-".join([str(x) for x in (
        batch_size, seq_len, input_dim, num_head, dim_head, causal, dropout, device
    )])
    
    output_dir = "/root/workspace/LightTorch/archive/hw4_extra"
    # Save the result to a text file.
    reshaped_result = result.reshape(-1, result.shape[-1])
    result_path = os.path.join(output_dir, f"test_attention_layer_result-{current_input_id}.txt")
    np.savetxt(result_path, reshaped_result, fmt='%.6f')

    labels_path = (
        "./tests/hw4_extra/data/" + 
        "test_attention_layer-{}.npy"
        .format(current_input_id))

    with open(labels_path, 'rb') as f:
        label_result = np.load(f)
    
    reshaped_label_result = label_result.reshape(-1, label_result.shape[-1])
    label_result_path = os.path.join(output_dir, f"test_attention_layer_label_result-{current_input_id}.txt")
    np.savetxt(label_result_path, reshaped_label_result, fmt='%.6f')
    
    
test_attention_layer(4, 5, 27, 8, 32, True, 0.1, ndl.cpu())

    