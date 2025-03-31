import random
import re
import json
from random import choice, sample
import torch
import torch.nn.functional as f
from torch import nn
from tokenizers import BertWordPieceTokenizer

def read_hinglish_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            processed_content = re.sub("[.,!?\\-]", '', content.lower())
            return processed_content.split('\n')
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return []

FILE_PATH = 'Hinglish_data3.txt'
SENTENCES = read_hinglish_file(FILE_PATH)

tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=True,
)

valid_sentences = [s for s in SENTENCES if len(s.strip()) > 0]
tokenizer.train_from_iterator(
    valid_sentences,
    vocab_size=1000,
    min_frequency=1,
    special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"],
    wordpieces_prefix="##",
)

word_index = tokenizer.get_vocab()
index_word = {v: k for k, v in word_index.items()}
print("Vocabulary size:", len(word_index))

def max_len(sents):
    max_length = 0
    for sent in sents:
        encoded = tokenizer.encode(sent)
        max_length = max(max_length, len(encoded.tokens))
    return max_length + 1

def pad_sentence(tokens, length):
    if len(tokens) < length:
        tokens += ["[PAD]"] * (length - len(tokens))
    return tokens

def mask_sentence(size, mask_indices):
    m = [False] * size
    for v in mask_indices:
        if v < size:
            m[v] = True
    return m

def preprocess_sentences(sentences):
    sent = []
    mask = []
    max_l = max_len(sentences)

    for sentence in sentences:
        encoded = tokenizer.encode(sentence)
        tokens = encoded.tokens
        token_ids = encoded.ids
        
        if len(token_ids) == 0:
            continue

        p = min(len(token_ids), max(1, round(len(token_ids) * 0.15)))
        mask_indices = sample(range(len(token_ids)), p)
        
        masked_tokens = []
        for idx in range(len(tokens)):
            if idx in mask_indices:
                if random.random() < 0.15:
                    masked_tokens.append(choice(list(word_index.keys())))
                else:
                    masked_tokens.append("[MASK]")
            else:
                masked_tokens.append(tokens[idx])
        
        masked_tokens = ["[CLS]"] + masked_tokens
        masked_tokens = pad_sentence(masked_tokens, max_l)
        sent.append(masked_tokens)
        
        mask.append(mask_sentence(max_l, [i+1 for i in mask_indices]))
    
    data = {"sentences": sent, "mask": mask}
    with open("preprocessed_sentences.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return sent, mask

def form_ds(sentences):
    x, y = [], []
    for i in range(len(sentences) - 1):
        x.append(sentences[i] + sentences[i + 1])
        y.append(1)

    for i in range(len(sentences) - 1):
        new_sentences = [v for j, v in enumerate(sentences) if j != i + 1]
        s = choice(new_sentences)
        x.append(sentences[i] + s)
        y.append(0)
    
    dataset = [{"input": x[i], "label": y[i]} for i in range(len(x))]
    
    with open("dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    
    print("Dataset saved as dataset.json")
    return x, y

def tokenize(sentences):
    res = []
    for s in sentences:
        ids = [word_index.get(token, word_index["[UNK]"]) for token in s]
        res.append(ids)
    return res

def get_attn_pad_mask(seq_q):
    return seq_q.data.eq(word_index["[PAD]"])

class JointEmbedding(nn.Module):
    SEGMENTS = 2

    def __init__(self, vocab_size, size):
        super(JointEmbedding, self).__init__()
        self.token_emb = nn.Embedding(vocab_size, size)
        self.segment_emb = nn.Embedding(vocab_size, size)
        self.position_emb = nn.Embedding(vocab_size, size)
        self.norm = nn.LayerNorm(size)

    def forward(self, input_tensor, segment_tensor):
        pos_tensor = torch.arange(input_tensor.size(1), dtype=torch.long)
        pos_tensor = pos_tensor.expand_as(input_tensor)
        output = self.token_emb(input_tensor) + self.segment_emb(segment_tensor) + self.position_emb(pos_tensor)
        return self.norm(output)

class EncoderLayer(nn.Module):
    def __init__(self, dim, dim_ff, dropout=0.1, num_heads=4):
        super(EncoderLayer, self).__init__()
        dim_q = dim_k = max(dim // num_heads, 1)
        self.multi_head = MultiHeadAttention(num_heads, dim, dim_q, dim_k)
        self.position_feed_forward = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, dim),
            nn.Dropout(dropout)
        )

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor):
        output = self.multi_head(input_tensor, attention_mask)
        return self.position_feed_forward(output)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dim_inp, dim_q, dim_k):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([
            AttentionHead(dim_inp, dim_q, dim_k) for _ in range(num_heads)
        ])
        self.linear = nn.Linear(dim_q * num_heads, dim_inp)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor):
        scores = torch.cat([h(input_tensor, attention_mask) for h in self.heads], dim=-1)
        return self.linear(scores)

class AttentionHead(nn.Module):
    def __init__(self, dim_inp, dim_q, dim_k):
        super(AttentionHead, self).__init__()
        self.q = nn.Linear(dim_inp, dim_q)
        self.k = nn.Linear(dim_inp, dim_k)
        self.v = nn.Linear(dim_inp, dim_k)

    def forward(self, input_tensor: torch.Tensor, attention_mask):
        query, key, value = self.q(input_tensor), self.k(input_tensor), self.v(input_tensor)
        scale = query.size(1) ** 0.5
        scores = torch.bmm(query, key.transpose(1, 2)) / scale
        scores = scores.masked_fill_(attention_mask, -1e9)
        attn = f.softmax(scores, dim=-1)
        context = torch.bmm(attn, value)
        return context

class BERT(nn.Module):
    def __init__(self, vocab_size, size, encoder_size, num_heads=4):
        super(BERT, self).__init__()
        self.embedding = JointEmbedding(vocab_size, size)
        self.encoder = EncoderLayer(size, encoder_size, num_heads=num_heads)
        self.token_prediction_layer = nn.Linear(size, len(word_index))
        self.classification_layer = nn.Linear(size, 2)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor):
        embedded = self.embedding(input_tensor, input_tensor)
        encoded_sources = self.encoder(embedded, attention_mask)
        classification_embedding = encoded_sources[:, 0, :]
        classification_output = self.classification_layer(classification_embedding)
        token_output = self.token_prediction_layer(encoded_sources)
        return classification_output, token_output

if __name__ == '__main__':
    if not valid_sentences:
        print("No valid sentences found in file. Exiting.")
        exit()
        
    vocab_size = len(word_index)
    ds, mask = preprocess_sentences(valid_sentences)
    x_s, y = form_ds(ds)
    x = tokenize(x_s)
    inp = torch.tensor(x, dtype=torch.long)
    inp_mask = get_attn_pad_mask(inp)
    emb_size = 64
    encoder_size = 12
    bert = BERT(vocab_size, emb_size, encoder_size, num_heads=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(bert.parameters(), lr=0.0001)
    optimizer.zero_grad()

    ds_size = inp.size(0)
    for i in range(20000):
        j = random.randint(0, ds_size - 1)
        inp_x, inp_y = inp[j, :].unsqueeze(0), torch.Tensor([y[j]]).long()
        mask = get_attn_pad_mask(inp_x)
        class_out, token_out = bert(inp_x, mask)
        loss = criterion(class_out, inp_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 100 == 0:
            print(f"Epoch {i}. Loss {loss.item()}")

    for j in range(len(x)):
        inp_x, inp_y = inp[j, :].unsqueeze(0), torch.Tensor([y[j]]).long()
        class_out, _ = bert(inp_x, get_attn_pad_mask(inp_x))
        print(f"True: {inp_y.item()}, Pred: {class_out.argmax().item()}")