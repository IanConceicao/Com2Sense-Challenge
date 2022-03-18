from ast import arg
import csv
from distutils import dist
import glob
import json
import random
from pathlib import Path
import os
from enum import Enum
from typing import List, Optional, Union

from sklearn.metrics import accuracy_score

import tqdm
import numpy as np

import torch
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
)


def mask_tokens(inputs, tokenizer, args, special_tokens_mask=None):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK,
    10% random, 10% original.
    inputs should be tokenized token ids with size: (batch size X input length).
    """

    # The eventual labels will have the same size of the inputs,
    # with the masked parts the same as the input ids but the rest as
    # args.mlm_ignore_index, so that the cross entropy loss will ignore it.
    labels = inputs.clone()

    # Constructs the special token masks.
    if special_tokens_mask is None:
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool, device=args.device)
    else:
        special_tokens_mask = special_tokens_mask.bool()

    #Make matrix, fill with odds of being masked
    distribution = torch.full(labels.shape, args.mlm_probability, device=args.device)
    
    #Don't mask special tokens, assign probability 0
    distribution.masked_fill_(special_tokens_mask, 0)

    #Convert distribution to bernoulli, says which indices to mask
    bern = torch.bernoulli(distribution).type(torch.BoolTensor).to(args.device)

    #Fill in all NON-Masked spots
    labels.masked_fill_(~bern, args.mlm_ignore_index)

    # For 80% of the time, we will replace masked input tokens with  the
    # tokenizer.mask_token (e.g. for BERT it is [MASK] for for RoBERTa it is
    # <mask>, check tokenizer documentation for more details)
    
    to_replace_with = None
    r = torch.rand(1)[0]
    if r <= 0.8:
        to_replace_with = tokenizer.mask_token_id
    elif r <= 0.9:
        to_replace_with = torch.randint(28996,(1,))[0] #TODO replace 50,000 with actual vocab size
    
    if to_replace_with != None:
        for i in range(inputs.shape[0]):
            for j in range(inputs.shape[1]):
                if bern[i,j]: 
                    inputs[i,j] = to_replace_with

    # For 10% of the time, we replace masked input tokens with random word.
    # Hint: you may find function `torch.randint` handy.
    # Hint: make sure that the random word replaced positions are not overlapping
    # with those of the masked positions, i.e. "~indices_replaced".

    # For the rest of the time (10% of the time) we will keep the masked input
    # tokens unchanged

    return inputs, labels

#They are all numpy arrays
#Given a specific domain, evaluate the accuracy of tasks with that domain
def accuracy_of_domain(preds, labels, domains, domain_to_keep):

    if preds.shape != labels.shape or preds.shape != domains.shape:
        raise Error("Must all have the same type!")

    indices_to_keep = np.where(domains == domain_to_keep)[0]
    preds = preds.take(indices_to_keep)
    labels = labels.take(indices_to_keep)

    print("Looking at domain", domain_to_keep)
    print("indices to keep shape", indices_to_keep.shape)
    print("preds shape", preds.shape)
    print("labels shape",  labels.shape)
    print()

    return accuracy_score(labels, preds)


def pairwise_accuracy(guids, preds, labels):

    acc = 0.0  # The accuracy to return.
    
    # predictions and labels w.r.t the `guid`. 
    correct = 0
    total = len(preds) / 2
    for i in range(0, len(preds), 2):
        if preds[i] == labels[i] and preds[i+1] == labels[i+1]:
            correct += 1
    
    acc = correct / total
    return acc
     


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    pass


if __name__ == "__main__":

    class mlm_args(object):
        def __init__(self):
            self.mlm_probability = 0.4
            self.mlm_ignore_index = -100
            self.device = "cpu"
            self.seed = 42
            self.n_gpu = 0

    args = mlm_args()
    set_seed(args)

    # Unit-testing the MLM function.
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    input_sentence = "I am a good student and I love NLP."
    input_ids = tokenizer.encode(input_sentence)
    input_ids = torch.Tensor(input_ids).long().unsqueeze(0)
    
    inputs, labels = mask_tokens(input_ids, tokenizer, args,
                                 special_tokens_mask=None)
    inputs, labels = list(inputs.numpy()[0]), list(labels.numpy()[0])

    ans_inputs = [101,  146,  103,   170, 103,  2377, 103,  146,  1567, 103,   2101,  119, 102]

    ans_labels = [-100, -100, 1821, -100, 1363, -100, 1105, -100, -100, 21239, -100, -100, -100]

    if inputs == ans_inputs and labels == ans_labels:
        print("Your `mask_tokens` function is correct!")
    else:
        raise NotImplementedError("Your `mask_tokens` function is INCORRECT!")


    # Unit-testing the pairwise accuracy function.
    guids = [0, 0, 1, 1, 2, 2, 3, 3]
    preds = np.asarray([0, 0, 1, 0, 0, 1, 1, 1])
    labels = np.asarray([1, 0,1, 0, 0, 1, 1, 1])
    acc = pairwise_accuracy(guids, preds, labels)
    
    if acc == 0.75:
        print("Your `pairwise_accuracy` function is correct!")
    else:
        raise NotImplementedError("Your `pairwise_accuracy` function is INCORRECT!")

    ####
