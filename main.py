import random
import re

from random import choice, sample

import pandas as pd
import torch
import torch.nn.functional as f

from torch import nn
from torch.utils.data import Dataset

text = (
    'Hey, kya haal hai? Aaj ka din kaisa raha?\n'
    'Sab badhiya hai, thoda busy tha kaam mein. Tum batao?\n'
    'Main bhi thoda busy tha, lekin shaam ko doston ke saath ghoomne gaya.\n'
    'Wah, maze kar rahe ho! Kahan gaye the?\n'
    'Cafe gaye the, phir thodi der park mein time spend kiya. Tum kya kar rahe the?\n'
    'Maine ghar par thoda kaam kiya, phir ek movie dekhi.\n'
    'Kaunsi movie dekhi? Kaisi lagi?\n'
    'Inception dekhi, bohot mast thi. Thoda dimaag lagana padta hai is movie mein.\n'
    'Haan yaar, wo movie kaafi mind-blowing hai. Mujhe bhi bohot pasand aayi thi.\n'
    'Haan sach mein! Vaise weekend ke plans kya hain?\n'
    'Socha hai ki family ke saath thoda time spend karunga. Tumhare kya plans hain?\n'
    'Main bhi shayad friends ke saath outing karu, ya phir ghar par relax karu.\n'
    'Sounds good! Chalo phir, baad mein baat karte hain. Take care!\n'
    'Haan, milte hain jaldi. Bye!\n'
    'Arre suno, ek baat puchni thi.\n'
    'Haan bolo, kya hua?\n'
    'Kya tumhe kal ke match ka score pata hai? Maine dekha nahi.\n'
    'Haan, India jeet gaya! Kaafi close match tha.\n'
    'Wah, badhiya! Kaun sa player accha khela?\n'
    'Virat Kohli ne ek zabardast inning kheli, aur Bumrah ne last overs mein achha bowling kiya.\n'
    'Mazaa aa gaya sunke! Agla match kab hai?\n'
    'Agla match Sunday ko hai. Milkar dekhen kya?\n'
    'Haan, bilkul! Ek sath match dekhna aur bhi mazedar hota hai.\n'
    'Sahi kaha! Chalo phir, weekend fix ho gaya. Kal milte hain.\n'
    'Theek hai, milte hain kal. Good night!\n'
    'Good night, bye!'
)



SENTENCES = re.sub("[.,!?\\-]", '', text.lower()).split('\n')  
word_list = list(set(" ".join(SENTENCES).split()))

word_index = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3} 
index_word = {v: i for i, v in word_index.items()}

print(word_list)
print(word_index)
print(index_word)
print(SENTENCES)

print('-------------------------------------------------')

def max_len(sents):
    m = 0
    for v in sents:
        v = v.split()
        m = len(v) if len(v) > m else m
    return m


def pad_sentence(sentence, length):
    l = len(sentence)
    if l < length:
        sentence += ["[PAD]"] * (length - l)
    return sentence


def mask_sentence(size, mask_indices):
    m = [False for _ in range(size)]
    for v in mask_indices:
        m[v] = True
    return m


import json

def preprocess_sentences(sentences):
    sent = []
    mask = []
    max_l = max_len(sentences) + 1

    for sentence in sentences:
        sentence = sentence.split()
        p = round(len(sentence) * 0.15)
        mask_indices = sample(range(len(sentence)), p)

        p = round(len(mask_indices) * 0.15)
        mask_wrong_indices = sample(mask_indices, p)

        for v in mask_indices:
            if v in mask_wrong_indices:
                symb = choice(word_list)
            else:
                symb = "[MASK]"
            sentence[v] = symb

        mask.append(mask_sentence(max_l, mask_indices))

        s = ["[CLS]"] + sentence
        s = pad_sentence(s, max_l)
        sent.append(s)
    data = {"sentences": sent, "mask": mask}
    json_output = json.dumps(data, indent=4, ensure_ascii=False)
    with open("preprocessed_sentences.json", "w", encoding="utf-8") as f:
        f.write(json_output)

    return sent, mask

import json
from random import choice

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
    return x,y

def tokenize(sentences):
    res = []
    for s in sentences:
        tokens = [word_index[v] for v in s]
        res.append(tokens)
    return res


def get_attn_pad_mask(seq_q):
    return seq_q.data.eq(0)


class JointEmbedding(nn.Module):
    SEGMENTS = 2  # 0 - first sentence, 1 - second sentence. 2 is amount of segments

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

        self.dim_inp = dim_inp

        self.q = nn.Linear(dim_inp, dim_q)
        self.k = nn.Linear(dim_inp, dim_k)
        self.v = nn.Linear(dim_inp, dim_k)

    def forward(self, input_tensor: torch.Tensor, attention_mask):
        # input_tensor = input_tensor.squeeze(0)
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

        self.token_prediction_layer = nn.Linear(size, 1)
        self.classification_layer = nn.Linear(size, 2)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor):
        embedded = self.embedding(input_tensor, input_tensor)
        encoded_sources = self.encoder(embedded, attention_mask)

        classification_embedding = encoded_sources[:, 0, :]
        classification_output = self.classification_layer(classification_embedding)

        token_output = self.token_prediction_layer(encoded_sources)

        return classification_output, token_output


if __name__ == '__main__':
    for i, w in enumerate(word_list):
        word_index[w] = i + 4
        index_word[word_index[w]] = w
        vocab_size = len(word_index)

    ds, mask = preprocess_sentences(SENTENCES)
    x_s, y = form_ds(ds)
    x = tokenize(x_s)

    inp = torch.tensor(x, dtype=torch.long)

    inp_mask = get_attn_pad_mask(inp)

    emb_size = 64
    encoder_size = 12

    bert = BERT(len(word_index), emb_size, encoder_size, num_heads=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(bert.parameters(), lr=0.0001)
    optimizer.zero_grad()

    ds_size = inp.size(0)

    for i in range(20000):
        j = random.randint(0, ds_size - 1)
        inp_x, inp_y = inp[j, :].unsqueeze(0), torch.Tensor([y[j]]).long()
        inp_mask = get_attn_pad_mask(inp_x)
        class_out, token_out = bert(inp_x, inp_mask)

        loss = criterion(class_out, inp_y)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Epoch {i}. Loss {loss}")

    for j in range(len(x)):
        inp_x, inp_y = inp[j, :].unsqueeze(0), torch.Tensor([y[j]]).long()
        class_out, token_out = bert(inp_x, inp_mask)
        print(inp_y, class_out.argmax())