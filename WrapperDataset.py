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
        only_data=False,
        only_label=False,
        device = torch.device("cuda")
    ):
        self.dataset = inner_dataset
        self.block_size = inner_dataset.block_size
        self.metadata_file_path = metadata_file_path
        self.metadata = self.extract_popularities() #     access a fic using wrap_dataset.metadata.loc[[94746]]

        self.only_data = only_data
        self.only_label = only_label

        self.device = device

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
        data = super_get[0]
        id = super_get[1]
        score = self.get_score(id)
        score = float(self.get_score(id))
        if self.only_label:
            return score
        if self.only_data:
            return data
        return data, score

    def extract_popularities(self):
        dataframe = pd.read_csv(self.metadata_file_path)
        dataframeT = dataframe.T
        dataframeT.columns = dataframeT.iloc[0]
        dataframe = dataframeT.T
        return dataframe
