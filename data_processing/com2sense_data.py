import os
import sys
import json
import csv
import glob
import pprint
import numpy as np
import random
import argparse
import os
from tqdm import tqdm
from .utils import DataProcessor
from .utils import Coms2SenseSingleSentenceExample
from transformers import (
    AutoTokenizer,
)


class Com2SenseDataProcessor(DataProcessor):
    """Processor for Com2Sense Dataset.
    Args:
        data_dir: string. Root directory for the dataset.
        args: argparse class, may be optional.
    """

    def __init__(self, data_dir=None, args=None, **kwargs):
        """Initialization."""
        self.args = args
        self.data_dir = data_dir

        # TODO: Label to Int mapping, dict type.
        self.label2int = {"True": 1, "False": 0}

    def get_labels(self):
        """See base class."""
        return 2  # Binary.

    def _read_data(self, data_dir=None, split="train"):
        """Reads in data files to create the dataset."""
        if data_dir is None:
            data_dir = self.data_dir
        
        my_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        data_dir = my_dir + '/' + data_dir
        json_path = os.path.join(data_dir, split+".json")
        data = json.load(open(json_path, "r"))
        examples = []

        for i in range(len(data)):
            datum = data[i]
            guid = i
            text = None
            label_1 = None
            label_2 = None
            domain = None
            scenario = None
            numeracy = None
            sent_1 = None
            sent_2 = None
            if "sent_1" in datum: 
                sent_1 = datum["sent_1"]
            if "sent_2" in datum:
                sent_2 = datum["sent_2"]
            if "label_1" in datum:
                label_1 = self.label2int[datum["label_1"]]
            if "label_2" in datum:    
                label_2 = self.label2int[datum["label_2"]]
            if "domain" in datum:
                domain = datum["domain"]
            if "scenario" in datum:
                scenario = datum["scenario"]
            if "numeracy" in datum:
                numeracy = datum["numeracy"]

            example_1 = Coms2SenseSingleSentenceExample(
                guid=guid,
                text=sent_1,
                label=label_1,
                domain = domain,
                scenario = scenario,
                numeracy = numeracy
            )

            example_2 = Coms2SenseSingleSentenceExample(
                guid=guid,
                text=sent_2,
                label=label_2,
                domain = domain,
                scenario = scenario,
                numeracy = numeracy
            )

            examples.append(example_1)
            examples.append(example_2)

        return examples

    def get_train_examples(self, data_dir=None):
        """See base class."""
        return self._read_data(data_dir=data_dir, split="train")

    def get_dev_examples(self, data_dir=None):
        """See base class."""
        return self._read_data(data_dir=data_dir, split="dev")

    def get_test_examples(self, data_dir=None):
        """See base class."""
        return self._read_data(data_dir=data_dir, split="test")


if __name__ == "__main__":

    # Test loading data.
    proc = Com2SenseDataProcessor(data_dir="datasets/com2sense")
    train_examples = proc.get_train_examples()
    val_examples = proc.get_dev_examples()
    test_examples = proc.get_test_examples()
    print()
    for i in range(3):
        print(test_examples[i])
    print()
