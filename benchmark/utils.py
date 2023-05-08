def gpt2_4b():
    return dict(
        seq_len=1024,
        hidden_dim=3072,
        num_layers=32, 
        num_attention_heads=24,
        vocab_size=50260
    )


def gpt2_10b():
    return dict(
        seq_len=1024,
        hidden_dim=4096,
        num_layers=48, 
        num_attention_heads=32,
        vocab_size=50260
    )


def gpt2_15b():
    return dict(
        seq_len=1024,
        hidden_dim=8192,
        num_layers=18, 
        num_attention_heads=64,
        vocab_size=50260
    )


def gpt2_20b():
    return dict(
        seq_len=1024,
        hidden_dim=8192,
        num_layers=24, 
        num_attention_heads=64,
        vocab_size=50260
    )


def get_gpt_config(name: str):
    if name == 'gpt2-4b':
        return gpt2_4b()
    elif name == 'gpt2-10b':
        return gpt2_10b()
    elif name == 'gpt2-15b':
        return gpt2_15b()
    elif name == 'gpt2-20b':
        return gpt2_20b()
    else:
        raise NotImplementedError


def compute_gpt_parameter_count(num_layers, hidden_size, vocab_size):
    return num_layers * (
            # self-attention
            hidden_size * (3 * hidden_size + 1) +
            hidden_size * (hidden_size + 1) +
            # mlp
            hidden_size * (4 * hidden_size + 1) +
            hidden_size * 4 * (hidden_size + 1) +
            # layer norm
            hidden_size * 4
           ) + vocab_size * (hidden_size + 1)
