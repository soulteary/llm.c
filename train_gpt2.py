"""

GPT-2 训练和推理的代码示例。
将模型权重保存到文件中，作为初始化从 C 中读取。

参考：

1) OpenAI 官方发布的 GPT-2 TensorFlow 实现：
https://github.com/openai/gpt-2/blob/master/src/model.py
2) Huggingface/transformers PyTorch 实现：
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import os
import math
import struct
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# OpenAI 使用的 GELU 激活函数版本
# GELU (Gaussian Error Linear Units)是一种激活函数，结合了 ReLU 和 Dropout 的优点，能够缓解梯度消失问题并提高模型的泛化能力。
class NewGELU(nn.Module):
    # 前向传播时，对输入进行GELU变换并返回结果。
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


# 实现了 GPT-2 中的因果自注意力机制
# 因果自注意力机制中为防止信息泄露，只允许每个位置的 token 与其之前的 token 进行交互。
class CausalSelfAttention(nn.Module):
    # 初始化了注意力机制所需的线性变换和参数。
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # 定义了一个线性变换层 self.c_attn，用于计算注意力机制中的 Query、Key 和 Value 向量。
        # config.n_embd 表示输入的嵌入维度大小，即每个 token 的嵌入向量的维度，3 * config.n_embd 表示输出的维度大小，是输入维度的 3 倍。
        # 这个线性变换层的作用是将输入的嵌入向量映射到 Query、Key 和 Value 向量。
        # 
        # 由于是多头注意力机制，所以实际上是将输入映射到了所有注意力头的 Query、Key 和 Value 向量，并将它们在维度上拼接在一起。
        # 假设有 n_head 个注意力头，每个头的维度为 head_dim，则 config.n_embd = n_head * head_dim，经过这个线性变换后，输出的维度为 3 * config.n_embd。
        # 可以理解为 [n_head * head_dim_query, n_head * head_dim_key, n_head * head_dim_value] 的拼接。
        # 在后续的代码中，会将这个输出再分割成 Query、Key 和 Value 向量，并分别计算注意力权重和值。
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # 输出投影
        # 定义了一个线性变换层 self.c_proj，用于将注意力计算后的输出进行投影。
        # 输入和输出的维度都为 config.n_embd，表示嵌入向量的维度。
        # 这个投影层的作用是将注意力计算后的结果转换为与输入相同维度的向量，以便与输入进行残差连接。
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # 正则化
        # 存储了配置中的注意力头数 config.n_head 和嵌入维度 config.n_embd。
        # 这些参数在计算注意力时会用到，例如将输入拆分为多个注意力头，以及确保嵌入维度与注意力计算的维度匹配。
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        
        # 定义了一个名为 bias 的缓冲区，用于存储注意力掩码。
        # 使用 torch.tril 函数创建了一个下三角矩阵，大小为 (config.block_size, config.block_size)，表示序列长度与序列长度之间的掩码。
        # 下三角矩阵中的元素为1，表示允许注意力在当前位置及其之前的位置之间流动。
        # 使用 view 函数将下三角矩阵重塑为 (1, 1, config.block_size, config.block_size) 的形状，以便在注意力计算时广播到批次和头的维度。
        # 这个掩码的作用是确保在因果自注意力中，每个位置只能与其之前的位置进行交互，防止信息泄露。
        # 作者吐槽：虽然命名为 bias，但实际上它更像是一个掩码，用于掩盖注意力权重矩阵中的特定位置。这里沿用了 OpenAI 和 Hugging Face 的命名习惯。
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    # 方法对输入进行自注意力计算（因果自注意力机制的前向传播函数），并返回结果。
    # - 将输入通过线性变换得到 Query、Key、Value。
    # - 计算注意力权重矩阵，并使用 bias 掩码进行因果掩码，防止信息泄露。
    # - 对注意力权重进行 softmax 归一化。
    # - 将注意力权重与 Value 相乘，得到注意力输出。
    # - 对注意力输出进行线性变换，得到最终的输出。
    def forward(self, x):
        # 获取输入张量 x 的维度信息：批次大小 B、序列长度 T 和嵌入维度 C（即 config.n_embd）。
        B, T, C = x.size()
        # 将输入张量 x 通过线性变换层 self.c_attn 计算 Query、Key 和 Value 向量。
        # 输出张量 qkv 的维度为 (B, T, 3 * C)，其中包含了所有注意力头的 Query、Key 和 Value 向量的拼接。
        qkv = self.c_attn(x)
        # 将 qkv 张量沿着第二个维度（即嵌入维度）拆分为 Query、Key 和 Value 向量。
        # 拆分后的 q、k、v 张量的维度均为 (B, T, C)。
        q, k, v = qkv.split(self.n_embd, dim=2)
        # 将 Query、Key 和 Value 向量重塑为多头形式，并调整维度顺序。
        # 使用 view 函数将张量重塑为 (B, T, self.n_head, C // self.n_head) 的形状，其中 self.n_head 表示注意力头的数量，C // self.n_head 表示每个头的维度大小。
        # 使用 transpose 函数将第二个维度（序列长度）和第三个维度（注意力头）进行交换，得到 (B, self.n_head, T, C // self.n_head) 的形状。
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # 计算注意力权重矩阵 att。
        # 将 Query 矩阵 q 与 Key 矩阵 k 的转置进行矩阵乘法，得到注意力得分矩阵。
        # 将注意力得分矩阵除以 Key 向量维度的平方根，作为缩放因子，以稳定梯度。
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # 将注意力权重矩阵 att 中被掩码的位置填充为负无穷大。
        # 使用 self.bias 张量作为掩码，将其中为0的位置对应的注意力权重设置为负无穷大，以防止在因果自注意力中出现信息泄露。
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # 对注意力权重矩阵 att 应用 softmax 函数，将其归一化为概率分布。
        # 在最后一个维度上应用 softmax，即在序列长度维度上进行归一化。
        att = F.softmax(att, dim=-1)
        # 将注意力权重矩阵 att 与 Value 矩阵 v 进行矩阵乘法，得到注意力输出 y。
        # 注意力权重矩阵的维度为 (B, nh, T, T)，Value 矩阵的维度为 (B, nh, T, hs)，乘法结果的维度为 (B, nh, T, hs)。
        y = att @ v
        # 将注意力输出 y 的维度调整回原始顺序。
        # 使用 transpose 函数将第二个维度（注意力头）和第三个维度（序列长度）进行交换，得到 (B, T, nh, hs) 的形状。
        # 使用 contiguous 函数确保张量在内存中是连续的。
        # 使用 view 函数将张量重塑为 (B, T, C) 的形状，将所有注意力头的输出并排放置。
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # 将注意力输出 y 通过线性变换层 self.c_proj 进行输出投影。
        # 输出投影的目的是将注意力输出的维度调整为与输入张量 x 相同的维度 (B, T, C)。
        y = self.c_proj(y)
        # 返回因果自注意力的输出张量 y。
        return y

# 实现了 GPT-2 中的多层感知机(MLP)部分
# 定义了一个多层感知机模块，包含两个线性变换层和一个激活函数。
# 它的作用是对输入向量进行非线性变换，将其映射到更高维度的中间表示，然后再映射回原始维度。
# 这种变换可以增加模型的表达能力和非线性特征提取能力。
# 在 Transformer 模型中，MLP 模块通常用于对注意力机制的输出进行进一步的非线性变换，以提高模型的性能。
# MLP 继承自 nn.Module，表示这是一个 PyTorch 的神经网络模块。
class MLP(nn.Module):
    # 定义了 MLP 类的构造函数 __init__，接受一个 config 参数，用于配置 MLP 的超参数。
    def __init__(self, config):
        # 调用 super().__init__() 来初始化父类 nn.Module 的构造函数。
        super().__init__()
        # 定义了一个线性变换层 self.c_fc，将输入的嵌入向量 config.n_embd 维度映射到 4 * config.n_embd 维度。
        # 这个线性变换层的作用是将输入向量转换为更高维度的中间表示。
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        # 定义了一个激活函数 self.gelu，使用 NewGELU 类的实例。
        # NewGELU 是一个自定义的激活函数，类似于 GELU（高斯误差线性单元）。
        self.gelu    = NewGELU()
        # 定义了一个线性变换层 self.c_proj，将中间表示的 4 * config.n_embd 维度映射回原始的嵌入向量维度 config.n_embd。
        # 这个线性变换层的作用是将中间表示转换为与输入向量相同维度的输出向量。
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)

    # 定义了 MLP 类的前向传播函数 forward，接受一个输入张量 x。
    def forward(self, x):
        # 将输入张量 x 通过线性变换层 self.c_fc 进行转换，得到中间表示。
        x = self.c_fc(x)
        # 对中间表示应用激活函数 self.gelu，引入非线性变换。
        x = self.gelu(x)
        # 将激活后的中间表示通过线性变换层 self.c_proj 进行转换，得到与输入向量相同维度的输出张量。
        x = self.c_proj(x)
        # 返回转换后的输出张量 x。
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
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
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

# a few utilities for saving params/grads/activations to files for loading in C
def write_fp32(tensor, file):
    file.write(tensor.detach().numpy().astype("float32").tobytes())

def write_tensors(model_tensors, L, file):
    write_fp32(model_tensors["transformer.wte.weight"], file) # (V, C)
    write_fp32(model_tensors["transformer.wpe.weight"], file) # (T, C)
    for i in range(L): # (L, C)
        write_fp32(model_tensors[f"transformer.h.{i}.ln_1.weight"], file)
    for i in range(L): # (L, C)
        write_fp32(model_tensors[f"transformer.h.{i}.ln_1.bias"], file)
    for i in range(L): # (L, 3C, C)
        write_fp32(model_tensors[f"transformer.h.{i}.attn.c_attn.weight"], file)
    for i in range(L): # (L, 3C)
        write_fp32(model_tensors[f"transformer.h.{i}.attn.c_attn.bias"], file)
    for i in range(L): # (L, C, C)
        write_fp32(model_tensors[f"transformer.h.{i}.attn.c_proj.weight"], file)
    for i in range(L): # (L, C)
        write_fp32(model_tensors[f"transformer.h.{i}.attn.c_proj.bias"], file)
    for i in range(L): # (L, C)
        write_fp32(model_tensors[f"transformer.h.{i}.ln_2.weight"], file)
    for i in range(L): # (L, C)
        write_fp32(model_tensors[f"transformer.h.{i}.ln_2.bias"], file)
    for i in range(L): # (L, 4C, C)
        write_fp32(model_tensors[f"transformer.h.{i}.mlp.c_fc.weight"], file)
    for i in range(L): # (L, 4C)
        write_fp32(model_tensors[f"transformer.h.{i}.mlp.c_fc.bias"], file)
    for i in range(L): # (L, C, 4C)
        write_fp32(model_tensors[f"transformer.h.{i}.mlp.c_proj.weight"], file)
    for i in range(L): # (L, C)
        write_fp32(model_tensors[f"transformer.h.{i}.mlp.c_proj.bias"], file)
    write_fp32(model_tensors["transformer.ln_f.weight"], file) # (C, )
    write_fp32(model_tensors["transformer.ln_f.bias"], file) # (C, )

def write_model(model, filename):
    # everything we need to instantiate the model
    # 1) header is: version int, GPTConfig ints, padding to 1024 bytes
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240326 # magic
    header[1] = 1 # checkpoint version = 1
    header[2] = model.config.block_size
    header[3] = model.config.vocab_size
    header[4] = model.config.n_layer
    header[5] = model.config.n_head
    header[6] = model.config.n_embd
    # 2) the parameters on CPU are next
    params = {name: param.cpu() for name, param in model.named_parameters()}
    # now write
    with open(filename, "wb") as file:
        # header
        file.write(header.numpy().tobytes())
        # model parameters
        write_tensors(params, model.config.n_layer, file)
    print(f"wrote {filename}")

def write_state(model, x, y, logits, loss, filename):
    # the state is used for debugging.
    # it contains information about the input, logits, loss, and the parameter gradients
    # this can be used for checking the computation correctness in C
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240327 # magic
    header[1] = 1 # run state version = 1
    header[2] = x.size(0) # batch size of the batch, B
    header[3] = x.size(1) # temporal extent of the batch, T
    grads = {name: param.grad.cpu() for name, param in model.named_parameters()}
    with open(filename, "wb") as file:
        # header
        file.write(header.numpy().tobytes())
        # input x
        file.write(x.cpu().numpy().astype("int32").tobytes()) # (B, T)
        # targets y
        file.write(y.cpu().numpy().astype("int32").tobytes()) # (B, T)
        # logits (result of the model forward pass)
        write_fp32(logits.cpu(), file)
        # loss (single float, result of the cross entropy loss)
        write_fp32(loss.cpu(), file)
        # gradients
        write_tensors(grads, model.config.n_layer, file)
    print(f"wrote {filename}")

def write_tokenizer(enc, filename):
    n = enc.max_token_value + 1
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240328 # magic
    header[1] = 1 # tokenizer version = 1
    header[2] = n # number of tokens
    with open(filename, "wb") as file:
        file.write(header.numpy().tobytes())
        for i in range(n):
            b = enc.decode_bytes([i])
            length = len(b)
            assert length < 256, f"Token length exceeds 255: {length}"
            file.write(struct.pack("<B", length))  # Write the length as a 1-byte unsigned integer
            file.write(b)  # Write the actual bytes
    print(f"wrote {filename}")

if __name__ == "__main__":
    import time
    import argparse
    import tiktoken

    # default settings will overfit a tiny batch of data
    # and save model weights and debug state to disk on the first iteration
    # if you'd like to e.g. time the forward pass only, call this script as:
    # python train_gpt2.py --inference_only 1 --write_tensors 0 --sequence_length 1024
    parser = argparse.ArgumentParser()
    parser.add_argument("--write_tensors", type=int, default=1, help="write tensors to disk")
    parser.add_argument("--inference_only", type=int, default=0, help="only run inference")
    parser.add_argument("--compile", type=int, default=0, help="torch.compile the model")
    parser.add_argument("--tensorcores", type=int, default=0, help="use tensorcores")
    parser.add_argument("--num_iterations", type=int, default=10, help="number of iterations to run")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--sequence_length", type=int, default=64, help="sequence length")
    args = parser.parse_args()
    B, T = args.batch_size, args.sequence_length
    assert 1 <= T <= 1024

    # select a reasonable device to run on
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

    # seed the random number generators
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # init the tokenizer
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

    write_tokenizer(enc, "gpt2_tokenizer.bin")

    if args.tensorcores:
        torch.set_float32_matmul_precision('high')

    # load the GPT-2 model weights
    model = GPT.from_pretrained("gpt2")
    model.train()
    model.to(device)
    if args.compile:
        print("compiling the model...")
        model = torch.compile(model)

    # load the tokens
    # prefer to use tiny_shakespeare if it's available, otherwise use tiny_stories
    # we're using val instead of train split just because it is smaller/faster
    shake_tokens_bin = "data/tiny_shakespeare_val.bin"
    story_tokens_bin = "data/TinyStories_val.bin"
    assert os.path.isfile(shake_tokens_bin) or os.path.isfile(story_tokens_bin), "you must run prepro on some dataset"
    tokens_bin = shake_tokens_bin if os.path.isfile(shake_tokens_bin) else story_tokens_bin
    assert os.path.isfile(tokens_bin)
    print(f"loading cached tokens in {tokens_bin}")
    with open(tokens_bin, "rb") as f:
        tokens = np.frombuffer(f.read(), dtype=np.int32)

    # np -> tensor, long, on device
    tokens = torch.tensor(tokens)
    tokens = tokens.to(torch.long)
    tokens = tokens.to(device)

    # lightweight dataloader
    def get_batch():
        assert B*T+1 <= len(tokens), "not enough tokens"
        # for 338,025 tokens. E.g. with B=8 T=1024, this will yield 41 batches before looping
        i = 0
        while True:
            x = tokens[i:i+B*T].view(B, T)
            y = tokens[i+1:i+B*T+1].view(B, T)
            yield x, y
            i += B*T
            if i + B*T + 1 >= len(tokens):
                i = 0 # in prod we'd want to randomize the start point a bit

    # forward backward for a few iterations
    data_iter = iter(get_batch())
    x, y = next(data_iter) # we'll overfit this batch below
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    timings = []
    for i in range(args.num_iterations):
        t0 = time.time()
        logits, loss = model(x, y)
        if not args.inference_only:
            optimizer.zero_grad()
            loss.backward()
            # on the first iteration only, save the state dict to file for later reference
            if i == 0 and args.write_tensors:
                write_model(model, "gpt2_124M.bin")
                write_state(model, x, y, logits, loss, "gpt2_124M_debug_state.bin")
            optimizer.step()
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        if i > args.num_iterations - 20:
            timings.append(t1-t0)
        print(f"iteration {i}, loss: {loss.item()}, time: {(t1-t0)*1000:.3f}ms")
    if len(timings) > 0:
        print(f"final 20 iters avg: {np.mean(timings)*1000:.3f}ms")

    # before we end, let's also do one round of inference
    # we'll kick off the generation with "<|endoftext|>", which designates the start of a new sequence
    start = "<|endoftext|>"
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    # run generation for 16 time steps (tokens)
    max_new_tokens = 16
    temperature = 1.0
    top_k = 40
    model.eval()
    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    print(decode(y[0].tolist()))
    print('---------------')
