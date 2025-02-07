# import argparse
# import logging
# import os
# from typing import Any, Dict, Tuple
# import torch
# from transformers import DistilBertTokenizer, DistilBertModel

# import numpy as np

# logger = logging.getLogger(__name__)


# def data2feats(args: argparse.Namespace, sent: str, label: str, word_to_idx: Dict,
#                label_to_idx: Dict) -> Tuple[Any, Any]:
#     sent = sent.split()
#     if args.task == "trec":
#         feat = np.zeros([int(os.getenv("MAX_LEN_TREC"))], dtype=np.int64)
#     else:
#         feat = np.zeros([int(os.getenv("MAX_LEN_AGNEWS"))], dtype=np.int64)

#     for word_idx, word in enumerate(sent):
#         feat[word_idx] = word_to_idx[word]

#     label = np.array(label_to_idx[label], dtype=np.int64)

#     return feat, label

# def data2feats2(args: argparse.Namespace, sent: str,  word_to_idx: Dict,
#                label_to_idx: Dict) -> Tuple[Any, Any]:
#     sent = sent.split()
#     if args.task == "trec":
#         feat = np.zeros([int(os.getenv("MAX_LEN_TREC"))], dtype=np.int64)
#     else:
#         feat = np.zeros([int(os.getenv("MAX_LEN_AGNEWS"))], dtype=np.int64)

#     for word_idx, word in enumerate(sent):
#         feat[word_idx] = word_to_idx[word]


#     return feat




import argparse
import logging
import os
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def data2feats(args: argparse.Namespace, sent: str, label: str, word_to_idx: Dict,
               label_to_idx: Dict) -> Tuple[Any, Any]:
    sent = sent.split()
    if args.task == "trec":
        feat = np.zeros([int(os.getenv("MAX_LEN_TREC"))], dtype=np.int64)
    else:
        feat = np.zeros([int(os.getenv("MAX_LEN_AGNEWS"))], dtype=np.int64)

    for word_idx, word in enumerate(sent):
        feat[word_idx] = word_to_idx[word]

    label = np.array(label_to_idx[label], dtype=np.int64)

    return feat, label

def data2feats2(args: argparse.Namespace, sent: str, word_to_idx: Dict,
               ) -> Tuple[Any, Any]:
    sent = sent.split()
    if args.task == "trec":
        feat = np.zeros([int(os.getenv("MAX_LEN_TREC"))], dtype=np.int64)
    else:
        feat = np.zeros([int(os.getenv("MAX_LEN_AGNEWS"))], dtype=np.int64)

    for word_idx, word in enumerate(sent):
        feat[word_idx] = word_to_idx[word]


    return feat



def initialize_train_pool_test(args: argparse.Namespace, train: np.ndarray, pool: np.ndarray, test: np.ndarray, tokenizer) -> Tuple:
    logger.info("Generating training, pool, and test instances...")

    X_train, y_train = [], []
    X_pool = []
    X_test, y_test = [], []

    for sent, cls in train:
        feat = tokenizer.encode_plus(
                    sent,                      # Sentence to encode.
                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    max_length = 512,           # Pad & truncate all sentences.
                    truncation = True, 
                    padding = "max_length",
                    return_attention_mask = True,   # Construct attn. masks.
                    return_tensors = 'pt',     # Return pytorch tensors.
        )
        X_train.append(feat)
        y_train.append(cls)

    if len(pool) != 0:
        for sent in pool:
            feat = tokenizer.encode_plus(
            sent,                      # Sentence to encode.
            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
            max_length = 512,           # Pad & truncate all sentences.
            truncation = True, 
            padding = "max_length",
            return_attention_mask = True,   # Construct attn. masks.
            return_tensors = 'pt',     # Return pytorch tensors.
        )
            X_pool.append(feat)

    else:
        X_pool = []
        y_pool = []
    for sent, cls in test:
        feat = tokenizer.encode_plus(
                    sent,                      # Sentence to encode.
                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    max_length = 512,           # Pad & truncate all sentences.
                    truncation = True, 
                    padding = "max_length",
                    return_attention_mask = True,   # Construct attn. masks.
                    return_tensors = 'pt',     # Return pytorch tensors.
        )
        X_test.append(feat)
        y_test.append(cls)

    logger.info(f"Statistics: {len(X_train)} train, {len(X_pool)} pool, {len(X_test)} test.")

    X_train, y_train = X_train, y_train
    X_pool= X_pool
    X_test, y_test = X_test, y_test

    return X_train, y_train, X_pool, X_test, y_test


def initialize_train_pool_test2(args: argparse.Namespace, train: np.ndarray, pool: np.ndarray, test: np.ndarray,
                               word_to_idx: Dict,
                               label_to_idx: Dict) -> Tuple:
    logger.info("Generating training, pool, and test instances...")

    X_train, y_train = [], []
    X_pool = []
    X_test, y_test = [], []

    for sent, cls in train:
#         feat = tokenizer.encode_plus(
#                     sent,                      # Sentence to encode.
#                     add_special_tokens = True, # Add '[CLS]' and '[SEP]'
#                     max_length = 512,           # Pad & truncate all sentences.
#                     truncation = True, 
#                     padding = "max_length",
#                     return_attention_mask = True,   # Construct attn. masks.
#                     return_tensors = 'pt',     # Return pytorch tensors.
#         )
        feat, cls = data2feats(args, sent, cls, word_to_idx, label_to_idx)
        X_train.append(feat)
        y_train.append(cls)
    for sent, cls in test:
        feat, cls = data2feats(args, sent, cls, word_to_idx, label_to_idx)
        X_test.append(feat)
        y_test.append(cls)

    if len(pool) != 0:
        for sent in pool:
#             feat = tokenizer.encode_plus(
#             sent,                      # Sentence to encode.
#             add_special_tokens = True, # Add '[CLS]' and '[SEP]'
#             max_length = 512,           # Pad & truncate all sentences.
#             truncation = True, 
#             padding = "max_length",
#             return_attention_mask = True,   # Construct attn. masks.
#             return_tensors = 'pt',     # Return pytorch tensors.
#         )
            feat =  data2feats2(args, sent,word_to_idx)
            X_pool.append(feat)

    else:
        X_pool = []
        y_pool = []
#         feat = tokenizer.encode_plus(
#                     sent,                      # Sentence to encode.
#                     add_special_tokens = True, # Add '[CLS]' and '[SEP]'
#                     max_length = 512,           # Pad & truncate all sentences.
#                     truncation = True, 
#                     padding = "max_length",
#                     return_attention_mask = True,   # Construct attn. masks.
#                     return_tensors = 'pt',     # Return pytorch tensors.
#         )

    logger.info(f"Statistics: {len(X_train)} train, {len(X_pool)} pool, {len(X_test)} test.")

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_pool= np.array(X_pool)
    X_test, y_test = np.array(X_test), np.array(y_test)

    return X_train, y_train, X_pool, X_test, y_test