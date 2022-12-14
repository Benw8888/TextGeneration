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

csv.field_size_limit(2 ** 31 - 1)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

STATIC_PATH = resource_filename(__name__, "static")

class TokenDatasetList(Dataset):
    """
    Class designed to be a list of TokenDataset's, to solve issues with overlapping tokens between stories.
    """
    def __init__(
        self,
        token_datasets,
        block_size=1024,
        line_by_line: bool = False,
        load=False, # load from saved tokens
        ignore_label = False,
    ):
        self.block_size = block_size
        self.line_by_line = line_by_line
        self.datasets = token_datasets

        self.paragraph_split = False
        self.lengths = None
        self.cumulative_lengths = None
        self.full_length = None
        self.calculate_cumulative_lengths() # defines self.lengths, self.cumulative_lengths, self.full_length


        self.story_ids = []
        for i in range(len(token_datasets)):
            print("token file path: ",token_datasets[i].file_path)
            self.story_ids.append(token_datasets[i].file_path[23:-4])

        self.ignore_label = ignore_label

    def calculate_cumulative_lengths(self):
        if self.paragraph_split:
            self.paragraph_lengths = [[len(paragraph) for paragraph in dataset.tokens] for dataset in self.datasets]
            self.fiction_lengths = [sum(fiction) for fiction in self.paragraph_lengths]

            # regime of sending one paragraph at a time:
            self.lengths = [len(paragraph_list) for paragraph_list in self.datasets]  # number paragraphs per fiction
            self.cumulative_lengths = [0] * len(self.lengths)  # cumulative number of paragraphs before i, exclusive
            for i in range(len(self.cumulative_lengths) - 1):
                self.cumulative_lengths[i + 1] = self.cumulative_lengths[i] + self.lengths[i]
            self.full_length = self.cumulative_lengths[-1] + self.lengths[-1]

            pass
        else:
            self.lengths = [len(self.datasets[i].tokens) for i in range(len(self.datasets))]
            self.cumulative_lengths = [0] * len(self.lengths)  # lengths
            for i in range(len(self.cumulative_lengths) - 1):
                self.cumulative_lengths[i + 1] = self.cumulative_lengths[i] + self.lengths[i]
            self.full_length = self.cumulative_lengths[-1] + self.lengths[-1]

    def save(self, cache_destination: str = "dataset_list_cache.p", compress: bool = True):
        if compress:
            open_func = gzip.open
        else:
            open_func = open
        with open_func(cache_destination, 'wb') as outp:
            pickle.dump(self, outp, -1)

    def save1(
        self, cache_destination: str = "dataset_list_cache.tar.gz", compress: bool = True
    ) -> None:
        assert self.full_length > 0, "No data loaded to save."

        if compress:
            open_func = gzip.open
            compress_str = "and compressing "
        else:
            open_func = open
            cache_destination = (
                "dataset_list_cache.npy"
                if cache_destination == "dataset_list_cache.tar.gz"
                else cache_destination
            )
            compress_str = ""

        logger.info(f"Caching {compress_str}dataset to {cache_destination}")

        with open_func(cache_destination, "wb") as f:
            np.save(f, self.datasets)

    def __len__(self) -> int:
        return self.full_length

    def __getitem__(self, item: int):
        if item >= self.full_length:
            print("TRIED TOO BIG ITEM")
            return None

        i=0  # find the fiction to pull from
        while i<len(self.datasets)-1 and item>= self.cumulative_lengths[i+1]:
            i+= 1

        #first_token = max(0,item-self.block_size-self.cumulative_lengths[i]+1)
        #last_token = item - self.cumulative_lengths[i]+1

        if self.paragraph_split:
            paragraph_id = item - self.cumulative_lengths[i]
            extracted_tokens = self.datasets[i].tokens[paragraph_id]
            pad_needed = self.block_size - len(extracted_tokens)
            if pad_needed > 0:
                data_segment = torch.as_tensor(
                    np.concatenate((extracted_tokens, np.zeros(pad_needed))).astype(np.int64, copy=False),
                    dtype=torch.long,
                )
            else:
                data_segment = torch.as_tensor(
                    extracted_tokens.astype(np.int64, copy=False),
                    dtype=torch.long,
                )
            attention_mask = [1]*len(extracted_tokens) + [0]*pad_needed
            return {'input_ids': data_segment, 'attention_mask': attention_mask, 'story_id': self.story_ids[i]}


        first_token = item-self.block_size-self.cumulative_lengths[i]
        extracted_tokens = self.datasets[i].tokens[max(first_token,0) : first_token+self.block_size]
        pad_needed = self.block_size-len(extracted_tokens)
        attention_mask = [1] * len(extracted_tokens) + [0] * pad_needed
        if pad_needed>0:
            data_segment = torch.as_tensor(
                np.concatenate((extracted_tokens,np.zeros(pad_needed))).astype(np.int64, copy=False),
                dtype=torch.long,
            )
        else:
            data_segment = torch.as_tensor(
                extracted_tokens.astype(np.int64, copy=False),
                dtype=torch.long,
            )
        # if self.ignore_label:
        #     return data_segment
        # else:
        #     return data_segment, self.story_ids[i]
        return {'input_ids': data_segment, 'attention_mask': attention_mask, 'story_id': self.story_ids[i]}

    def update_story_ids(self):
        self.story_ids = []
        for i in range(len(self.datasets)):
            #print("token file path: ", self.datasets[i].file_path[20:-3])
            self.story_ids.append(self.datasets[i].file_path[23:-4])

class TokenDataset(Dataset):
    """
    Class that merges TextDataset and LineByLineTextDataset from
    run_language_modeling.py in transformers, plus
    adds more ways to ingest text such as with CSVs.

    :param file_path: A string indicating the relative file path of the text
    to be tokenized, or the cached dataset.
    :param vocab_file: Path to a vocab file (generated by train_tokenizer())
    :param merges_file: Path to a merges file (generated by train_tokenizer())
    :param texts: A list of input texts (if providing texts manually)
    :param line_by_line: A boolean to indicate if the input file should be read
    line by line (True) or as a full text (False).
    :param from_cache: A string indicating if loading from a pregenerated MsgPack
    dump.
    :param header: A boolean indicating if loading from a CSV, if it has a header.
    :param save_cache: A boolean indicating whether to save the tokenized
    dataset as a MsgPack dump to load later.
    :param cache_destination: A string indicating where to save the cache.
    :param block_size: An integer indicating maximum length of the text document
    (usually set by the model architecture)
    :param tokenized_texts: Texts that are already tokenized; only should
    be used by merge_datasets().
    :param text_delim: delimiter to use to split bulk texts (default paragraph breaks)
    :param bos_token: String to override the beginning-of-string token
    :param eos_token: String to override the end-of-string token
    :param unk_token: String to override the unknown token
    :param pad_token: String to override the padding token
    :param progress_bar_refresh_rate: How often to update progress bar when loading
    """

    def __init__(
        self,
        file_path: str = None,
        vocab_file: str = os.path.join(STATIC_PATH, "gpt2_vocab.json"),
        merges_file: str = os.path.join(STATIC_PATH, "gpt2_merges.txt"),
        tokenizer: GPT2TokenizerFast = None,
        tokenizer_file: str = None,
        texts: List[str] = None,
        line_by_line: bool = False,
        from_cache: bool = False,
        header: bool = True,
        save_cache: bool = False,
        cache_destination: str = "dataset_cache.tar.gz",
        compress: bool = True,
        block_size: int = 1024,
        tokenized_texts=None,
        text_delim: str = "\n",
        bos_token: str = "<|endoftext|>",
        eos_token: str = "<|endoftext|>",
        unk_token: str = "<|endoftext|>",
        pad_token: str = "<|endoftext|>",
        progress_bar_refresh_rate: int = 20,
        **kwargs,
    ) -> None:

        self.line_by_line = False

        # Special case; load tokenized texts immediately
        if tokenized_texts is not None:
            self.tokens = tokenized_texts
            self.num_subsets = self.tokens.shape[0] - block_size
            self.block_size = block_size
            self.file_path = "merged TokenDataset"
            self.str_suffix = "by merging TokenDatasets."
            return

        assert any([texts, file_path]), "texts or file_path must be specified."

        if not tokenizer:
            if tokenizer_file:
                # load the custom tokenizer from a serialized tokenizer
                tokenizer = PreTrainedTokenizerFast(
                    tokenizer_file=tokenizer_file,
                    bos_token=bos_token,
                    eos_token=eos_token,
                    unk_token=unk_token,
                    pad_token=pad_token,
                    padding_side="left",
                )
            else:
                tokenizer = GPT2TokenizerFast(
                    vocab_file=vocab_file,
                    merges_file=merges_file,
                    bos_token=bos_token,
                    eos_token=eos_token,
                    unk_token=unk_token,
                    pad_token=pad_token,
                    verbose=False,
                    padding_side="left",
                )
                # https://github.com/huggingface/transformers/issues/10202
                tokenizer.add_special_tokens(
                    {"additional_special_tokens": ["<|endoftext|>"]}
                )

        # If a cache path is provided, load it.
        if from_cache:
            open_func = gzip.open if file_path.endswith(".gz") else open

            with open_func(file_path, "rb") as f:
                self.tokens = np.load(f)
            self.num_subsets = self.tokens.shape[0] - block_size
            self.block_size = block_size
            self.line_by_line = line_by_line
            self.str_suffix = "via cache."

            logger.info(
                f"TokenDataset containing {self.num_subsets:,} subsets loaded {self.str_suffix}"
            )
            return

        # if texts are present, just tokenize them.
        elif texts:
            self.str_suffix = "via application."

        # if a file is specified, and it's line-delimited,
        # the text must be processed line-by-line into a a single bulk file
        elif line_by_line:
            assert os.path.isfile(
                file_path
            ), f"{file_path} is not present in the current directory."

            text_delim = None
            self.line_by_line = True
            self.file_path = file_path
            self.str_suffix = f"from line-by-line file at {file_path}."

        # if a file is specified, and it's not line-delimited,
        # the texts must be parsed as a single bulk file.
        else:
            assert os.path.isfile(
                file_path
            ), f"{file_path} is not present in the current directory."
            if file_path.endswith(".csv"):
                logger.warning(
                    "You are tokenizing a CSV file, but you did not "
                    + "set line_by_line=True. Please change if unintended."
                )

            eos_token = ""
            header = False
            self.file_path = file_path
            self.str_suffix = f"from file at {file_path}."

        # Encode tokens in a batched manner to ensure constant memory usage
        if texts:
            self.tokens = encode_tokens_from_list(
                texts, eos_token, tokenizer, progress_bar_refresh_rate
            )
        else:
            self.tokens = encode_tokens_from_file(
                file_path,
                eos_token,
                tokenizer,
                text_delim,
                header,
                progress_bar_refresh_rate,
            )

        assert (
            self.tokens.shape[0] >= block_size
        ), f"There are fewer than {block_size} encoded tokens."
        self.num_subsets = self.tokens.shape[0] - block_size
        self.block_size = block_size

        if save_cache:
            self.save(cache_destination, compress=compress)

    def save(
        self, cache_destination: str = "dataset_cache.tar.gz", compress: bool = True
    ) -> None:
        assert self.tokens.shape[0] > 0, "No data loaded to save."

        if compress:
            open_func = gzip.open
            compress_str = "and compressing "
        else:
            open_func = open
            cache_destination = (
                "dataset_cache.npy"
                if cache_destination == "dataset_cache.tar.gz"
                else cache_destination
            )
            compress_str = ""

        logger.info(f"Caching {compress_str}dataset to {cache_destination}")

        with open_func(cache_destination, "wb") as f:
            np.save(f, self.tokens)

    def __len__(self) -> int:
        return self.num_subsets

    def __getitem__(self, item: int) -> torch.Tensor:
        return torch.as_tensor(
            self.tokens[item : (item + self.block_size)].astype(np.int64, copy=False),
            dtype=torch.long,
        )

    def __str__(self) -> str:
        return self.file_path if self.file_path is not None else "loaded dataset"

    def __repr__(self) -> str:
        return f"TokenDataset containing {self.num_subsets:,} subsets loaded {self.str_suffix}"


def get_lines_in_file(file_path: str, newline: str = None) -> int:
    """
    Returns the number of lines in a file to build progress bar.
    c.f. https://stackoverflow.com/a/16108605/9314418
    """

    with open(file_path, "r", encoding="utf-8", newline=newline) as f:
        return sum(1 for row in f)


def get_lines_in_file_csv(file_path: str, header: bool = True) -> int:
    """
    Returns the number of lines in a CSV to build progress bar.
    c.f. https://stackoverflow.com/a/16108605/9314418
    """

    with open(file_path, "r", encoding="utf-8") as f:
        if header:
            f.readline()
        reader = csv.reader(f)
        return sum(1 for row in reader)


def get_dtype(vocab_size: int):
    """
    Finds the appropriate numpy dtype depending on vocab size.

    The highest value for the dtype serves as a placeholder.
    """
    if vocab_size < 2 ** 8 - 1:
        return np.uint8
    elif vocab_size < 2 ** 16 - 1:
        return np.uint16
    elif vocab_size < 2 ** 32 - 1:
        return np.uint32

    return np.uint64


def encode_tokens_from_file(
    file_path: str,
    eos_token: str,
    tokenizer: GPT2TokenizerFast,
    newline: str,
    header: bool = True,
    progress_bar_refresh_rate: int = 20,
    batch_size: int = 1024,
) -> List[int]:
    """
    Retrieves texts from a newline-delimited file/CSV and returns texts.
    """

    is_csv = file_path.endswith(".csv")
    a_dtype = get_dtype(tokenizer.vocab_size)

    if is_csv:
        num_texts = get_lines_in_file_csv(file_path, header)
    else:
        num_texts = get_lines_in_file(file_path, newline)

    pbar = tqdm(
        total=num_texts,
        smoothing=0,
        leave=True,
        dynamic_ncols=True,
    )
    tokens = np.full((num_texts, 1), -1, dtype=a_dtype)
    num_batches = 0

    with open(file_path, "r", encoding="utf-8", newline=newline) as f_load:

        if header:
            f_load.readline()
        if is_csv:
            f_read = csv.reader(f_load)
            logger.info(f"Encoding {num_texts:,} rows from {file_path}.")
        else:
            f_read = f_load
            logger.info(f"Encoding {num_texts:,} sets of tokens from {file_path}.")

        # https://stackoverflow.com/a/6335876/9314418
        while True:
            if is_csv:
                batch = [
                    text[0] + eos_token
                    for text in list(itertools.islice(f_read, batch_size))
                ]
            else:
                batch = [
                    text + eos_token
                    for text in list(itertools.islice(f_read, batch_size))
                ]

            if not batch:
                break

            encoded_texts = tokenizer(
                batch,
                add_special_tokens=False,
                return_token_type_ids=False,
                return_attention_mask=False,
            )["input_ids"]

            for i, encoded_text in enumerate(encoded_texts):
                if len(encoded_text) > tokens.shape[1]:
                    cols_to_add = len(encoded_text) - tokens.shape[1]
                    tokens = np.concatenate(
                        (
                            tokens,
                            np.full(
                                (num_texts, cols_to_add),
                                -1,
                                dtype=a_dtype,
                            ),
                        ),
                        axis=1,
                    )
                tokens[
                    (num_batches * batch_size) + i, : len(encoded_text)
                ] = encoded_text

            num_batches += 1

            if num_batches % progress_bar_refresh_rate == 0:
                pbar.update(batch_size * progress_bar_refresh_rate)

    pbar.n = num_texts
    pbar.refresh()
    pbar.close()
    tokens = tokens.flatten()
    return tokens[tokens < np.array(-1, dtype=a_dtype)]


def encode_tokens_from_list(
    texts: List[str],
    eos_token: str,
    tokenizer: GPT2TokenizerFast,
    progress_bar_refresh_rate: int = 20,
    batch_size: int = 1024,
) -> List[int]:
    """
    Retrieves texts from a newline-delimited file/CSV and returns texts.
    """

    num_texts = len(texts)
    a_dtype = get_dtype(tokenizer.vocab_size)
    logger.info(f"Encoding {num_texts:,} texts.")

    pbar = tqdm(
        total=num_texts,
        smoothing=0,
        leave=True,
        dynamic_ncols=True,
    )
    tokens = np.full((len(texts), 1), -1, dtype=a_dtype)

    for i_start in range(num_texts // batch_size + 1):
        batch = [
            text + eos_token
            for text in texts[
                (i_start * batch_size) : ((i_start * batch_size) + batch_size)
            ]
        ]

        encoded_texts = tokenizer(
            batch,
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=False,
        )["input_ids"]

        for i, encoded_text in enumerate(encoded_texts):
            if len(encoded_text) > tokens.shape[1]:
                cols_to_add = len(encoded_text) - tokens.shape[1]
                tokens = np.concatenate(
                    (
                        tokens,
                        np.full(
                            (num_texts, cols_to_add),
                            -1,
                            dtype=a_dtype,
                        ),
                    ),
                    axis=1,
                )
            tokens[(i_start * batch_size) + i, : len(encoded_text)] = encoded_text

        if i_start % progress_bar_refresh_rate == 0:
            pbar.update(batch_size * progress_bar_refresh_rate)

    pbar.n = num_texts
    pbar.refresh()
    pbar.close()
    tokens = tokens.flatten()
    return tokens[tokens < np.array(-1, dtype=a_dtype)]


def merge_datasets(datasets: List[TokenDataset], equalize: bool = True) -> TokenDataset:
    """
    Merges multiple TokenDatasets into a single TokenDataset.
    This assumes that you are using the same tokenizer for all TokenDatasets.

    :param datasets: A list of TokenDatasets.
    :param equalize: Whether to take an equal amount of samples from all
    input datasets (by taking random samples from
    each dataset equal to the smallest dataset)
    in order to balance out the result dataset.
    """

    assert (
        isinstance(datasets, list) and len(datasets) > 1
    ), "datasets must be a list of multiple TokenDatasets."

    len_smallest = min([len(dataset) for dataset in datasets])
    block_size = datasets[0].block_size

    tokenized_texts = []

    for dataset in datasets:
        assert (
            dataset.block_size == block_size
        ), "The input datasets have different block sizes."
        if equalize:
            tokenized_texts.extend(dataset.tokens[0:len_smallest])
        else:
            tokenized_texts.extend(dataset.tokens)

    return TokenDataset(tokenized_texts=np.array(tokenized_texts), block_size=block_size)

