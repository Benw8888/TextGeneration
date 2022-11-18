from aitextgen.TokenDataset import TokenDataset
import numpy as np
import pandas as pd
from aitextgen.TokenDataset import TokenDataset, TokenDatasetList
from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast
import aitextgen
from aitextgen import aitextgen
import os
from pkg_resources import resource_filename
import pickle as pickle
import time
import torch
from torch import cuda
from torch.utils.data import random_split
import gzip
import WrapperDataset
import story_judger
from tqdm import tqdm

if __name__ == '__main__':
    # load cached dataset, created from create_datasets.py:

    STATIC_PATH = resource_filename(__name__, "aitextgen/static")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    #tokenizer = GPT2TokenizerFast(vocab_file=os.path.join(STATIC_PATH, "gpt2_vocab.json"),
    #                              merges_file=os.path.join(STATIC_PATH, "gpt2_merges.txt"), padding_side="left")
    # https://github.com/huggingface/transformers/issues/10202
    #tokenizer.add_special_tokens({"additional_special_tokens": ["<|endoftext|>"]})


    # CREATE DATASET

    with gzip.open('dataset_list_cache.p', 'rb') as inp:
        full_dataset = pickle.load(inp)

    wrap_dataset = WrapperDataset.WrapperDataset(full_dataset, device=device)

    wrap_dataset.dataset.update_story_ids()

    total_dataset_length = len(wrap_dataset)
    print("TOTAL DATASET LENGTH: ", total_dataset_length)
    train_dataset_length = 20000
    test_dataset_length = 2000
    remainder = total_dataset_length - train_dataset_length - test_dataset_length
    train_dataset, test_dataset, _ = random_split(wrap_dataset,[train_dataset_length,test_dataset_length, remainder])
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size = 64, num_workers=16,shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size = 1, num_workers=16,)

    # CREATE MODEL

    baseline_pretrained = True
    if baseline_pretrained:
        with gzip.open('baseline_judger.p', 'rb') as inp:
            baseline_judger = pickle.load(inp)
            baseline_judger.lstm.flatten_parameters()
    else:
        baseline_judger = story_judger.StoryJudger().to(device)

    loss_fn = torch.nn.functional.mse_loss

    # TRAIN MODEL
    if False:
        optimizer = torch.optim.Adam(baseline_judger.parameters(),lr=0.001,)

        num_epochs = 5
        for e in (range(num_epochs)):
            print("Starting Epoch ", e+1)
            total_loss = 0
            for step, (x,y) in enumerate(tqdm(train_dataloader, position=0, leave=True)):
                optimizer.zero_grad()
                output_pred = baseline_judger(x.to(device))
                loss = loss_fn(output_pred, y.to(device))
                loss.backward()
                optimizer.step()
                total_loss += torch.mean(loss)

                if (step+1)%50==0:
                    print("CURRENT LOSS: ", total_loss.item()/50)
                    total_loss = 0

        print("SAVING JUDGER MODEL: ")

        compress = True
        if compress:
            open_func = gzip.open
        else:
            open_func = open
        with open_func("baseline_judger.p", 'wb') as outp:
            pickle.dump(baseline_judger, outp, -1)

    print("CALCULATING TEST LOSS: ")

    with torch.no_grad():
        total_loss = 0
        cur_step = 0
        for step, (x, y) in enumerate(tqdm(test_dataloader, position=0, leave=True)):
            output_pred = baseline_judger(x.to(device))
            loss = loss_fn(output_pred, y.to(device))
            total_loss += torch.mean(loss)
            cur_step = step

        print("num steps: ", cur_step+1)
        print("TEST LOSS: ", total_loss.item() / (cur_step+1))


    exit()

    #file_name = "fanfics.csv_text_files/94746.txt"
    #new_story = TokenDataset(file_name, tokenizer=tokenizer, padding_side="left")

    # create default ai text gen model
    load_existing = False # turn to true once we already have a model stored on file
    if load_existing:
        ai = aitextgen(model_folder="trained_model2")
    else:
        ai = aitextgen()

    #full_dataset = TokenDataset(file_path="dataset_cache.tar.gz", from_cache=True, )
    #print(len(full_dataset.tokens)) #892140
    #print(len(tokenizer.decode(full_dataset.tokens)))

    # Train the model! It will save pytorch_model.bin periodically and after completion to the `trained_model` folder.
    ai.train(full_dataset, output_dir="trained_model3",batch_size=8, freeze_layers=4, num_steps=100, generate_every=10, save_every=1000, padding_side="left")

    # Generate text from it!
    ai.generate(2, prompt="Naruto formed his hands into a cross shape. \"Shadow Clone Jutsu!\" he yelled, and in a burst of smoke a hundred copies of him appeared.\n\n"
                           "Sasuke looked at him in disdain, grotesque hands growing out of his back from the effects of Orochimaru's curse mark. \"You can't stop me. I'll do anything I can to get the power to kill my brother Itachi. Even if that means turning to Orochimaru.\"\n\n"
                           "Naruto grit his teeth in determination. \"I'll stop you! I'll bring you back to Konoha for sure!\"" )

    #cuda.cudaDeviceReset()

    # With your trained model, you can reload the model at any time by
    # providing the folder containing the pytorch_model.bin model weights + the config, and providing the tokenizer.
    #ai2 = aitextgen(model_folder="trained_model")

    #ai2.generate(10, prompt="ROMEO:")






