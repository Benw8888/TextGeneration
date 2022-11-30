import torch
import logging
import csv
import os
import gzip
from torch.utils.data import Dataset
from typing import List
from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast
from pkg_resources import resource_filename
import itertools
from tqdm.auto import tqdm
import numpy as np
import pickle as pickle
import pandas as pd
import math

csv.field_size_limit(2 ** 31 - 1)


class WrapperDataset(Dataset):
    """
    Dataset that is a wrapper of a TokenDatasetList. It takes the (data, work_id) output of TokenDatasetList inner_dataset
    and modifies it (ex: pulling the popularity from the work_id)
    """
    def __init__(
        self,
        inner_dataset,
        metadata_file_path="fanfics_metadata.csv",
        return_data=True,
        return_label=False,
        return_attention_mask=True,
        device = torch.device("cuda"),
        line_by_line = False,
        paragraph_split = False,  # If we're doing autoencoding, we split our dataset into paragraphs. This changes a lot
    ):
        self.dataset = inner_dataset # a TokenDatasetList
        self.block_size = inner_dataset.block_size
        self.metadata_file_path = metadata_file_path
        self.metadata = self.extract_popularities() #     access a fic using wrap_dataset.metadata.loc[[94746]]

        self.return_data = return_data
        self.return_label = return_label
        self.return_attention_mask = return_attention_mask

        self.device = device
        self.paragraph_split = paragraph_split

        if self.paragraph_split:
            self.extract_paragraphs()

    def __len__(self) -> int:
        return len(self.dataset)

    def get_score(self, fiction_id):
        #print("fic id: ",fiction_id)
        if isinstance(fiction_id,str):
            fiction_id = int(fiction_id) # .to(self.device)
        #print("row: ",self.metadata.loc[[fiction_id]])
        #for col in self.metadata.loc[[fiction_id]].keys():
        #    print("col ",col,self.metadata.loc[[fiction_id]][col])
        kudos = self.metadata.loc[[fiction_id]]["kudos"]
        #print("kudos: ",kudos)
        #print(type(kudos))
        return math.log(kudos+1)


    def __getitem__(self, item: int):
        super_get = self.dataset[item]
        output = dict()
        output['input_ids'] = super_get['input_ids']
        id = super_get['story_id']
        output['attention_mask'] = super_get['attention_mask']

        #score = self.get_score(id)
        #score = float(self.get_score(id))

        #output['labels'] = score

        return output["input_ids"]


    def extract_popularities(self):
        dataframe = pd.read_csv(self.metadata_file_path)
        dataframeT = dataframe.T
        dataframeT.columns = dataframeT.iloc[0]
        dataframe = dataframeT.T
        return dataframe

    def extract_paragraphs(self):
        # for each story in self.dataset, split into paragraphs

        def split_tokens(tokens):
            """
            Given a list of tokens, return the first paragraph.
            Max size 1024
            """
            min_length = 29
            end = min(1023,len(tokens)-1)
            # try to find newline split: 198
            while end >= min_length:
                if tokens[end] != 198:
                    end -= 1
                else:
                    return tokens[:end], end
            # newline failed, now we just do full length:
            end = min(1024, len(tokens))
            return tokens[:end], end

        for sub_dataset in self.dataset.datasets:
            copy_tokens = sub_dataset.tokens
            paragraph_list = []
            i=0
            while len(copy_tokens)>0:
                i+= 1
                if i%1000==0:
                    print(i)
                new_paragraph, end_index = split_tokens(copy_tokens)
                paragraph_list.append(new_paragraph)
                copy_tokens = copy_tokens[end_index:]
            sub_dataset.tokens = paragraph_list
        # modify self.dataset tokens

        self.dataset.paragraph_split = True
        self.dataset.calculate_cumulative_lengths()  # modify len, modify getitem, modify tokens

        pass
