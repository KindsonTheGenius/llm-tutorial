# Concize self-attention class
import torch.nn as nn
import torch

class SelfAttention_v1(nn.Module):
    # Initialize the trainable weight matrices
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vector = attention_weights @ values
        return context_vector


# Imporoved Self-Attention Class
class SelfAttention_v2(nn.Module):

    # The weights are initialized using a different weights
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_keys = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_values= nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_keys(x)
        queries = self.W_query(x)
        values = self.W_values(x)

        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vector = attention_weights @ values
        return context_vector


# Self Attention with masked causal attention and dropout
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_lenght, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_lenght, context_lenght), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attention_scores = queries @ keys.transpose(1,2)

        attention_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context_vector = attention_weights @ values
        return context_vector
    
    # Updates on the CausalAttention class
    # 1. a dropout layer was added
    # 2. the call to register_buffer() was added. Buffers are automatically moved to the appropriate device along with our model
    # 3. we transpose dimention 1 and 2, keeping the batch dimension at the first position 0
    # 4. notice the operation with a trailing underscore (_). This means that the operation is performed in place, saving memory



    #  Multihead attention 
    class MultiHeadAttentionWrapper(nn.Module):
        def __init__(self, d_in, d_out, context_lenght, num_heads, dropout, qkv_bias=False):
            nn.ModuleList(
                [CausalAttention(d_in, d_out,context_lenght, dropout, qkv_bias) for _ in range(num_heads)]
            )

        def forward(self, x):
            return torch.cat([head(x) for head in self.heads], dim=-1)