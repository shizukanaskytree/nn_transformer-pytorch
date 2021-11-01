"""
Language Translation with nn.Transformer and torchtext
======================================================

This tutorial shows, how to train a translation model from scratch using
Transformer. We will be using `Multi30k <http://www.statmt.org/wmt16/multimodal-task.html#task1>`__
dataset to train a German to English translation model.
"""


######################################################################
# Data Sourcing and Processing
# ----------------------------
#
# `torchtext library <https://pytorch.org/text/stable/>`__ has utilities for creating datasets that can be easily
# iterated through for the purposes of creating a language translation
# model. In this example, we show how to use torchtext's inbuilt datasets,
# tokenize a raw text sentence, build vocabulary, and numericalize tokens into tensor. We will use
# `Multi30k dataset from torchtext library <https://pytorch.org/text/stable/datasets.html#multi30k>`__
# that yields a pair of source-target raw sentences.
#
#
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k
from typing import Iterable, List
import numpy as np
import sys

# matplot
import matplotlib.pyplot as plt

# logging
import log
logger = log.get_logger(__name__)
torch.set_printoptions(linewidth=1000)

sys.stdout.flush()

# import debugpy
# debugpy.listen(5678)
# debugpy.wait_for_client()
# debugpy.breakpoint()


from torch.utils.tensorboard import SummaryWriter
# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('./runs/transformer_seq2seq_tfboard')


SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

# Place-holders
token_transform = {}
vocab_transform = {}


# Create source and target language tokenizer. Make sure to install the dependencies.
# pip install -U spacy
# python -m spacy download en_core_web_sm
# python -m spacy download de_core_news_sm
token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')

# print('token_transform:')
# print(token_transform)
# {'de': functools.partial(<function _spacy_tokenize at 0x7f38523201f0>, spacy=<spacy.lang.de.German object at 0x7f3991f15700>), 'en': functools.partial(<function _spacy_tokenize at 0x7f38523201f0>, spacy=<spacy.lang.en.English object at 0x7f3851a7be20>)}

# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        tmp3 = language_index[language]
        logger.debug(f"tmp3: {tmp3}")

        tmp2 = data_sample[tmp3]
        logger.debug(f"tmp2: {tmp2}")

        tmp = token_transform[language](tmp2)
        logger.debug(f"tmp: {tmp}")

        yield tmp

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # Training data Iterator
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    # Create torchtext's Vocab object
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

# Set UNK_IDX as the default index. This index is returned when the token is not found.
# If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
  vocab_transform[ln].set_default_index(UNK_IDX)

######################################################################
# Seq2Seq Network using Transformer
# ---------------------------------
#
# Transformer is a Seq2Seq model introduced in `“Attention is all you
# need” <https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf>`__
# paper for solving machine translation tasks.
# Below, we will create a Seq2Seq network that uses Transformer. The network
# consists of three parts. First part is the embedding layer. This layer converts tensor of input indices
# into corresponding tensor of input embeddings. These embedding are further augmented with positional
# encodings to provide position information of input tokens to the model. The second part is the
# actual `Transformer <https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html>`__ model.
# Finally, the output of Transformer model is passed through linear layer
# that give un-normalized probabilities for each token in the target language.
#


from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()

        # logger.debug(f"emb_size\n {emb_size}")

        tmp1 = torch.arange(0, emb_size, 2)
        # logger.debug(f'tmp1\n {tmp1}')

        tmp2 = math.log(10000)
        # logger.debug(f'tmp2\n {tmp2}')

        tmp = - tmp1 * tmp2 / emb_size
        # logger.debug(f'tmp\n {tmp}')

        den = torch.exp(tmp)
        # logger.info(f"den\n{den.shape}\n{den}")

        # den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        # logger.info(f"pos\n{pos.shape}\n{pos}")

        pos_embedding = torch.zeros((maxlen, emb_size))
        # logger.debug(f"pos_embedding\n {pos_embedding}")
        # logger.debug(f"maxlen\n {maxlen}")

        pos_embedding[:, 0::2] = torch.sin(pos * den)
        # logger.info(f'torch.sin(pos * den)\n{torch.sin(pos * den).shape}\n{torch.sin(pos * den)}')
        # logger.info(f'pos*den\n{(pos*den).shape}\n{pos*den}')

        # tensor_to_save = pos * den
        # np.savetxt('tensor_values.txt', tensor_to_save.numpy())

        # logger.debug(f'pos_embedding[:, 0::2]\n {pos_embedding}')

        pos_embedding[:, 1::2] = torch.cos(pos * den)
        # logger.debug(f'torch.cos(pos * den)\n{torch.cos(pos * den)}')
        # logger.debug(f'pos_embedding[:, 1::2]\n {pos_embedding}')

        pos_embedding = pos_embedding.unsqueeze(-2)
        # logger.debug(f'pos_embedding\n {pos_embedding}')

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        # logger.info(f"toke_embedding\t{token_embedding.shape}")

        tmp = self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
        return tmp

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        # logger.info('print tokens...')
        # print(tokens.shape)
        # print(tokens)

        tmp2 = tokens.long()
        tmp3 = math.sqrt(self.emb_size)
        tmp = self.embedding(tmp2) * tmp3
        # logger.info('print tmp...')
        # print(tmp.shape)
        # print(tmp)

        return tmp
        # return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        # print('generator, state_dict\n', self.generator.state_dict())

        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        # logger.info(f"{src_vocab_size}, {emb_size}")
        # print('src_tok_emb, state_dict\n', self.src_tok_emb.state_dict())
        # for key_src in self.src_tok_emb.state_dict().keys():
        #     tensor_src_key = self.src_tok_emb.state_dict()[key_src].shape
        #     print(tensor_src_key)


        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        # logger.info(f"{tgt_vocab_size}, {emb_size}")
        # print('tgt_tok_emb, state_dict\n', self.tgt_tok_emb.state_dict())
        # for key_tgt in self.tgt_tok_emb.state_dict().keys():
        #     tensor_tgt_key = self.tgt_tok_emb.state_dict()[key_tgt].shape
        #     print(tensor_tgt_key)

        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

        # print('+'*70)
        # print('positional_encoding, state_dict\n', self.positional_encoding.state_dict())
        # print('positional_encoding, parameters\n', self.positional_encoding.parameters())
        # print(list(self.positional_encoding.parameters()))
        # print('-'*70)


    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        # logger.info(f"src\n{src.shape}\n{src}")
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        t3 = self.src_tok_emb(src)
        t2 = self.positional_encoding(t3)
        t1 = self.transformer.encoder(t2, src_mask)
        return t1
        # return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        t3 = self.tgt_tok_emb(tgt)
        t2 = self.positional_encoding(t3)
        t1 = self.transformer.decoder(t2, memory, tgt_mask)
        return t1
        # return self.transformer.decoder(self.positional_encoding(
        #                   self.tgt_tok_emb(tgt)), memory,
        #                   tgt_mask)


######################################################################
# During training, we need a subsequent word mask that will prevent model to look into
# the future words when making predictions. We will also need masks to hide
# source and target padding tokens. Below, let's define a function that will take care of both.
#


def generate_square_subsequent_mask(sz):
    logger.info('sz') # sz is size
    print(sz)

    t2 = torch.ones((sz, sz), device=DEVICE)
    logger.info('t2')
    print(t2)

    t1 = torch.triu(t2) == 1
    logger.info('t1')
    print(t1)

    mask = t1.transpose(0, 1)
    logger.info('mask')
    print(mask)
    ### original code:
    # mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)

    t3 = mask
    t4 = mask == 0
    logger.info('t4')
    print(t4)

    t5 = t3.float().masked_fill(t4, float('-inf'))
    logger.info('t5')
    print(t5)

    t7 = mask == 1
    logger.info('t7')
    print(t7)

    t6 = t5.masked_fill(t7, float(0.0))
    logger.info('t6')
    print(t6)

    print('='*70)

    return t6

    ### original code:
    # mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)) # to cmt
    # return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


######################################################################
# Let's now define the parameters of our model and instantiate the same. Below, we also
# define our loss function which is the cross-entropy loss and the optmizer used for training.
#
torch.manual_seed(0)

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

# logger.info('transformer, state_dict\n', transformer.state_dict())


for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

######################################################################
# Collation
# ---------
#
# As seen in the ``Data Sourcing and Processing`` section, our data iterator yields a pair of raw strings.
# We need to convert these string pairs into the batched tensors that can be processed by our ``Seq2Seq`` network
# defined previously. Below we define our collate function that convert batch of raw strings into batch tensors that
# can be fed directly into our model.
#


from torch.nn.utils.rnn import pad_sequence

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

# src and tgt language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor


# function to collate data samples into batch tesors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

######################################################################
# Let's define training and evaluation loop that will be called for each
# epoch.
#

from torch.utils.data import DataLoader

def train_epoch(model, optimizer):
    model.train()
    losses = 0
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    # logger.info(f'print train_iter...')
    # print(train_iter)

    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for i, (src, tgt) in enumerate(train_dataloader):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        # tfboard
        if i == 0:
            # print('src:', src)
            # print('tgt:', tgt)

            writer.add_histogram('src', src)
            writer.add_histogram('tgt', tgt)
            writer.add_graph(model, (src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask))
            writer.close()


        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(train_dataloader)


def evaluate(model):
    model.eval()
    losses = 0

    val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_dataloader)

######################################################################
# Now we have all the ingredients to train our model. Let's do it!
#

from timeit import default_timer as timer
NUM_EPOCHS = 18

for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer)
    end_time = timer()
    val_loss = evaluate(transformer)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    logger.info('src')
    print(src)

    logger.info('src_mask')
    print(src_mask)

    logger.info('max_len')
    print(max_len)

    logger.info('start_symbol')
    print(start_symbol)

    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    logger.info("memory")
    print(memory)

    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    logger.info("ys")
    print(ys)

    for i in range(max_len-1):
        memory = memory.to(DEVICE)

        print(i)
        print(generate_square_subsequent_mask(ys.size(0)))

        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(DEVICE)

        out = model.decode(ys, memory, tgt_mask)
        logger.info('out')
        print(out)

        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()

    t2 = list(tgt_tokens.cpu().numpy())
    t3 = vocab_transform[TGT_LANGUAGE].lookup_tokens(t2)
    t4 = " ".join(t3)
    t5 = t4.replace("<bos>", "")
    t1 = t5.replace("<eos>", "")
    return t1
    # return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")


######################################################################
#

print(translate(transformer, "Eine Gruppe von Menschen steht vor einem Iglu ."))


######################################################################
# References
# ----------
#
# 1. Attention is all you need paper.
#    https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
# 2. The annotated transformer. https://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding