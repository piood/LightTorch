from typing import List
from ltorch.autograd import Tensor
import ltorch.backend_ndarray.ndarray as ndarray
from ltorch import ops
import ltorch.init as init
import numpy as np
from .nn_sequence import Embedding
from .nn_basic import (
    Parameter, 
    Module, 
    ReLU,
    Dropout,
    LayerNorm1d,
    Linear,
    Sequential
)


import math


class MultiHeadAttention(Module):
    """
    The multi-head self attention module.
    """
    def __init__(
        self,
        *,
        dropout = 0.,
        causal = False,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        self.causal = causal
        self.dropout = Dropout(dropout)
        

    def create_causal_mask(self, i, j, device):
        """
        return a triangular causal mask.
        """
        mask = -np.finfo(np.float32).max * np.triu(
            np.ones((1, 1, i, j), dtype=np.float32), j - i + 1)

        return ndarray.array(
            mask, device=device)

    def matmul(self, a, b_transpose):
        """
        batched matrix multiplication;
        """
        '''
        a_shape = (*a.shape[:-1], 1, *a.shape[-1:])
        a = a.reshape(a_shape)

        b_transpose_shape = (*b_transpose.shape[:-2], 1, *b_transpose.shape[-2:])
        b_transpose = b_transpose.reshape(b_transpose_shape)

        broadcast_shape = list(a_shape)
        broadcast_shape[-2] = b_transpose_shape[-2]
        a = a.broadcast_to(broadcast_shape)

        broadcast_shape = list(b_transpose_shape)
        broadcast_shape[-3] = a_shape[-3]
        b_transpose = b_transpose.broadcast_to(broadcast_shape)

        print("a shape: ", a.shape)
        print("b_transpose shape: ", b_transpose.shape)
        return (a * b_transpose).sum(len(a.shape) - 1)
        '''
        a_shape = (*a.shape, 1)
        b_shape = (*b_transpose.shape[:-2], 1, b_transpose.shape[-2], b_transpose.shape[-1])
        a_reshaped = a.reshape(a_shape)
        b_reshaped = b_transpose.reshape(b_shape)
        broadcast_shape = list(a_shape)
        broadcast_shape[-1] = b_shape[-1]
        a_reshaped = a_reshaped.broadcast_to(broadcast_shape)
        b_reshaped = b_reshaped.broadcast_to(broadcast_shape)
        out = (a_reshaped * b_reshaped).sum(len(broadcast_shape) - 2)
        out_shape = list(a.shape)
        out_shape[-1] = b_transpose.shape[-1]
        return out.reshape(tuple(out_shape))

    def softmax(self, logit):
        """
        The softmax function; 
        """
        max_val = Tensor(
            logit.realize_cached_data().max(axis=3),
            device=logit.device,
            dtype=logit.dtype,
            requires_grad=False
        )

        max_val = max_val.reshape((*logit.shape[:-1], 1))
        max_val = max_val.broadcast_to(logit.shape)

        probs = ops.exp(logit - max_val)

        denom = probs.sum(axes=3)
        denom = denom.reshape((*logit.shape[:-1], 1))
        denom = denom.broadcast_to(logit.shape)

        return probs / denom

    def forward(
        self,
        q, k, v,
    ):
        """
        The forward function of the MultiHeadAttention activation function.
        Input: three states q, k, v, with shape (batch_size, num_head, seq_len, dim_head)
        Output: the activation output `result` and attention softmax probability `probs` (with dropout applied)
        """
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, keys_values_len, k_dim = k.shape
        _, _, _, v_dim = v.shape

        assert q_dim == k_dim == v_dim

        '''
        ### BEGIN YOUR SOLUTION
        # 计算注意力得分
        scores = self.matmul(k, q) / math.sqrt(k_dim)
        
        # 处理因果掩码（如果有）
        if self.causal:
            mask = self.create_causal_mask(queries_len, keys_values_len, device=self.device)
            scores = scores + Tensor(mask.broadcast_to(scores.shape), device=self.device, dtype=self.dtype, requires_grad=False)
        
        # 计算softmax概率
        probs = self.softmax(scores)
        
        # 应用dropout
        probs = self.dropout(probs)
        
        # 计算最终输出
        #result = self.matmul(probs, v)
        v_transpose = ops.transpose(v, axes=(2, 3))
        result = self.matmul(probs, v_transpose)
        
        ### END YOUR SOLUTION

        return result, probs
        '''
        probs = self.matmul(q, k.transpose()) / np.sqrt(q_dim)
        if self.causal:
            mask = self.create_causal_mask(queries_len, keys_values_len, device=self.device)
            # print("MASK", mask.shape, "PROBS", probs.shape)
            probs = probs + Tensor(mask.broadcast_to(probs.shape), device=self.device, dtype=self.dtype, requires_grad=False)
        probs = self.softmax(probs)
        probs = self.dropout(probs)
        result = self.matmul(probs, v)
        
        return result, probs


class AttentionLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        *,
        k_features: int = None,
        v_features: int = None,
        out_features: int = None,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        if k_features is None:
            k_features = q_features
        if v_features is None:
            v_features = q_features
        if out_features is None:
            out_features = q_features

        self.q_features = q_features
        self.k_features = k_features
        self.v_features = v_features
        self.out_features = out_features

        self.num_head = num_head
        self.dim_head = dim_head

        self.prenorm_q = LayerNorm1d(
            q_features, device=device, dtype=dtype)
        self.prenorm_k = LayerNorm1d(
            k_features, device=device, dtype=dtype)
        self.prenorm_v = LayerNorm1d(
            v_features, device=device, dtype=dtype)

        inner_dim = num_head * dim_head
        
        self.q_projection = Linear(
            q_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.k_projection = Linear(
            k_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.v_projection = Linear(
            v_features, inner_dim, bias=False,
            device=device, dtype=dtype)

        self.attn = MultiHeadAttention(
            dropout=dropout, causal=causal,
            device=device, dtype=dtype)

        self.out_projection = Linear(
            inner_dim, out_features, bias=False,
            device=device, dtype=dtype)

    def forward(
        self,
        q, k=None, v=None,
    ):
        """
        The forward function of the self-attention layer.
        Input: `q` with shape (batch_size, q_len, q_dim)
               `k` (if not None) with shape (batch_size, kv_len, k_dim)
               `v` (if not None) with shape (batch_size, kv_len, v_dim)
        Output: the output `result` with shape (batch_size, kv_len, out_features)
        """

        if k is None:
            k = q
        if v is None:
            v = q

        batch_size, queries_len, q_dim = q.shape
        _, keys_values_len, k_dim = k.shape
        _, _, v_dim = v.shape

        result = None

        ### BEGIN YOUR SOLUTION
        
        # 对输入进行层标准化
        q = self.prenorm_q(q)
        k = self.prenorm_k(k)
        v = self.prenorm_v(v)
        
        # 线性投影
        q = self.q_projection(q)
        k = self.k_projection(k)
        v = self.v_projection(v)

        shape_q = (batch_size, queries_len, self.num_head, self.dim_head)
        q = q.reshape(shape_q)
        q = ops.transpose(q, axes=(1, 2))
        shape_k = (batch_size, keys_values_len, self.num_head, self.dim_head)
        k = k.reshape(shape_k)
        k = ops.transpose(k, axes=(1, 2))
        v = v.reshape(shape_k)
        v = ops.transpose(v, axes=(1, 2))

        x, att_mat = self.attn(q, k, v)
        x = ops.transpose(x, axes=(1, 2))
        x = x.reshape((batch_size, queries_len, self.num_head * self.dim_head)) 
        x = self.out_projection(x)
        result = x
        
        ### END YOUR SOLUTION

        return result


class TransformerLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        hidden_size: int,
        *,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        ### BEGIN YOUR SOLUTION
        self.attention = AttentionLayer(
            q_features, num_head, dim_head,
            dropout=dropout, causal=causal,
            device=device, dtype=dtype
        )
        
        self.norm1 = LayerNorm1d(q_features, device=device, dtype=dtype)
        self.norm2 = LayerNorm1d(q_features, device=device, dtype=dtype)
        
        self.linear1 = Linear(q_features, hidden_size, device=device, dtype=dtype)
        self.linear2 = Linear(hidden_size, q_features, device=device, dtype=dtype)
        
        self.relu = ReLU()
        
        self.dropout = Dropout(dropout)
            
        ### END YOUR SOLUTION

    def forward(
        self,
        x
    ):
        """
        The forward function of a Transformer Layer.
        Input: the hidden states from previous layers `x` with shape (batch_size, seq_len, x_dim)
        Ouput: the hidden states after the Transformer Layer `x` with shape (batch_size, seq_len, x_dim)
        """

        batch_size, seq_len, x_dim = x.shape

        ### BEGIN YOUR SOLUTION
        residual = x
        
        #x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + residual
        
        residual = x
        x = self.norm2(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = x + residual
        ### END YOUR SOLUTION

        return x


class Transformer(Module):

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        num_layers: int, 
        *,
        num_head: int = 8,
        dim_head: int = 32,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
        batch_first = False,
        sequence_len = 2048
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype
        self.batch_first = batch_first

        ### BEGIN YOUR SOLUTION
        # 定义 Transformer layers
        self.layers = Sequential(
            *[TransformerLayer(embedding_size, num_head, dim_head, hidden_size,
                               dropout=dropout, causal=causal,
                               device=device, dtype=dtype) for _ in range(num_layers)]
        )

        # 定义位置编码
        #self.pos_embedding = Embedding(sequence_len, embedding_size, device=device, dtype=dtype)
        
        ### END YOUR SOLUTION
        
    def positional_encoding(seq_len, d_model):
        # 初始化位置编码矩阵
        pos_encoding = np.zeros((seq_len, d_model))
        
        # 生成位置序列 (0 到 seq_len-1)
        positions = np.arange(seq_len)[:, np.newaxis]
        
        # 生成维度索引序列 (0 到 d_model-1)
        dims = np.arange(d_model)[np.newaxis, :]
        
        # 计算频率指数
        angle_rates = 1 / np.power(10000, (2 * (dims // 2)) / np.float32(d_model))
        
        # 生成位置编码
        pos_encoding[:, 0::2] = np.sin(positions * angle_rates[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(positions * angle_rates[:, 1::2])
        
        # 添加batch维度
        pos_encoding = pos_encoding[np.newaxis, ...]
        
        return pos_encoding

    def forward(
        self,
        x, h=None
    ):

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        ### BEGIN YOUR SOLUTION
        # 获取输入序列的长度
        batch_size, seq_len, d_model = x.shape

        '''
        # 创建位置序列
        positions = np.arange(seq_len).reshape(1, seq_len)
        positions = Tensor(positions, device=self.device, dtype=self.dtype, requires_grad=False)
        positions = positions.broadcast_to((batch_size, seq_len))
        positions = ops.transpose(positions, axes=(1, 0))

        # 获取位置编码
        pos_emb = self.pos_embedding(positions)
        pos_emb = ops.transpose(pos_emb, axes=(1, 0))
        '''
        pos_emb = Transformer.positional_encoding(seq_len, d_model)
        
        pos_emb = Tensor(pos_emb, device=self.device, dtype=self.dtype, requires_grad=False)
        
        pos_emb = ops.broadcast_to(pos_emb, (batch_size, seq_len, d_model))
        # 将位置编码添加到输入嵌入中
        x = x + pos_emb

        # 通过 Transformer layers
        x = self.layers(x)
        ### END YOUR SOLUTION

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        return x, init.zeros_like(x)