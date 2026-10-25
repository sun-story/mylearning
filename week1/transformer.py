
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048, dropout=0.1):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(
            position * div_term[:pe[:, 1::2].shape[1]]
        )

        self.register_buffer("pe", pe.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.reshape(
            batch_size, seq_len, self.num_heads, self.d_k
        )
        return x.transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        q = self.split_heads(self.w_q(query))
        k = self.split_heads(self.w_k(key))
        v = self.split_heads(self.w_v(value))

        x, attn = self.attention(q, k, v, mask)

        x = x.transpose(1, 2).contiguous()
        x = x.reshape(x.size(0), x.size(1), self.d_model)
        x = self.w_o(x)

        return self.dropout(x), attn


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model, num_heads, dropout
        )
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        attn_out, attn = self.self_attn(
            x, x, x, src_mask
        )
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x, attn


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model, num_heads, dropout
        )
        self.cross_attn = MultiHeadAttention(
            d_model, num_heads, dropout
        )
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, src_mask=None):
        attn_out, self_attn = self.self_attn(
            x, x, x, tgt_mask
        )
        x = self.norm1(x + self.dropout1(attn_out))

        attn_out, cross_attn = self.cross_attn(
            x, memory, memory, src_mask
        )
        x = self.norm2(x + self.dropout2(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))

        return x, self_attn, cross_attn


class Encoder(nn.Module):
    def __init__(
        self, num_layers, d_model, num_heads,
        d_ff, dropout=0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, src_mask=None):
        attentions = []

        for layer in self.layers:
            x, attn = layer(x, src_mask)
            attentions.append(attn)

        return self.norm(x), attentions


class Decoder(nn.Module):
    def __init__(
        self, num_layers, d_model, num_heads,
        d_ff, dropout=0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, memory, tgt_mask=None, src_mask=None):
        self_attentions = []
        cross_attentions = []

        for layer in self.layers:
            x, self_attn, cross_attn = layer(
                x, memory, tgt_mask, src_mask
            )
            self_attentions.append(self_attn)
            cross_attentions.append(cross_attn)

        return self.norm(x), self_attentions, cross_attentions


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=6,
        d_ff=1024,
        max_len=2048,
        dropout=0.1,
        pad_idx=0,
    ):
        super().__init__()

        self.pad_idx = pad_idx

        self.src_embedding = nn.Embedding(
            src_vocab_size, d_model, padding_idx=pad_idx
        )
        self.tgt_embedding = nn.Embedding(
            tgt_vocab_size, d_model, padding_idx=pad_idx
        )

        self.positional_encoding = PositionalEncoding(
            d_model, max_len, dropout
        )

        self.encoder = Encoder(
            num_layers, d_model, num_heads, d_ff, dropout
        )
        self.decoder = Decoder(
            num_layers, d_model, num_heads, d_ff, dropout
        )

        self.output_projection = nn.Linear(
            d_model, tgt_vocab_size
        )

        self.d_model = d_model

    def make_src_mask(self, src):
        return (src == self.pad_idx)[:, None, None, :]

    def make_tgt_mask(self, tgt):
        batch_size, seq_len = tgt.shape
        pad_mask = (tgt == self.pad_idx)[:, None, None, :]
        causal_mask = torch.triu(
            torch.ones(
                seq_len, seq_len,
                device=tgt.device, dtype=torch.bool
            ),
            diagonal=1,
        )
        return pad_mask | causal_mask[None, None, :, :]

    def encode(self, src):
        src_mask = self.make_src_mask(src)
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        memory, attentions = self.encoder(x, src_mask)
        return memory, src_mask, attentions

    def decode(self, tgt, memory, src_mask):
        tgt_mask = self.make_tgt_mask(tgt)
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x, self_attns, cross_attns = self.decoder(
            x, memory, tgt_mask, src_mask
        )
        logits = self.output_projection(x)
        return logits, self_attns, cross_attns

    def forward(self, src, tgt):
        memory, src_mask, enc_attns = self.encode(src)
        logits, dec_attns, cross_attns = self.decode(
            tgt, memory, src_mask
        )
        return logits, {
            "encoder": enc_attns,
            "decoder": dec_attns,
            "cross": cross_attns,
        }

    @torch.no_grad()
    def generate(self, src, bos_idx, eos_idx, max_len=100):
        was_training = self.training
        self.eval()

        memory, src_mask, _ = self.encode(src)
        ys = torch.full(
            (src.size(0), 1),
            bos_idx,
            dtype=torch.long,
            device=src.device,
        )
        finished = torch.zeros(
            src.size(0), dtype=torch.bool, device=src.device
        )

        for _ in range(max_len):
            logits, _, _ = self.decode(ys, memory, src_mask)
            next_token = logits[:, -1, :].argmax(dim=-1)
            next_token = torch.where(
                finished,
                torch.full_like(next_token, eos_idx),
                next_token,
            )
            ys = torch.cat([ys, next_token[:, None]], dim=1)
            finished |= next_token.eq(eos_idx)

            if finished.all():
                break

        if was_training:
            self.train()

        return ys


def make_batch(batch_size, vocab_size, max_len, device):
    lengths = torch.randint(
        3, max_len + 1, (batch_size,)
    )

    src = torch.zeros(
        batch_size, max_len + 1, dtype=torch.long
    )
    tgt = torch.zeros(
        batch_size, max_len + 2, dtype=torch.long
    )

    for i, length in enumerate(lengths.tolist()):
        tokens = torch.randint(3, vocab_size, (length,))
        src[i, :length] = tokens
        src[i, length] = 2

        tgt[i, 0] = 1
        tgt[i, 1:length + 1] = tokens
        tgt[i, length + 1] = 2

    return src.to(device), tgt.to(device)


def train_demo():
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    vocab_size = 100
    pad_idx, bos_idx, eos_idx = 0, 1, 2

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=512,
        max_len=128,
        dropout=0.1,
        pad_idx=pad_idx,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=3e-4
    )
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    model.train()

    for step in range(1, 1001):
        src, tgt = make_batch(
            batch_size=32,
            vocab_size=vocab_size,
            max_len=16,
            device=device,
        )

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        logits, _ = model(src, tgt_input)

        loss = criterion(
            logits.reshape(-1, vocab_size),
            tgt_output.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )
        optimizer.step()

        if step % 100 == 0:
            print(f"step={step}, loss={loss.item():.4f}")

    model.eval()

    src, _ = make_batch(
        batch_size=2,
        vocab_size=vocab_size,
        max_len=8,
        device=device,
    )

    prediction = model.generate(
        src, bos_idx, eos_idx, max_len=20
    )

    print("source:", src.cpu())
    print("prediction:", prediction.cpu())


if __name__ == "__main__":
    train_demo()
