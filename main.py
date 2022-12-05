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
import Encoder

if __name__ == '__main__':
    # load cached dataset, created from create_datasets.py:

    STATIC_PATH = resource_filename(__name__, "aitextgen/static")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tokenizer = GPT2TokenizerFast(vocab_file=os.path.join(STATIC_PATH, "gpt2_vocab.json"),
                                 merges_file=os.path.join(STATIC_PATH, "gpt2_merges.txt"), padding_side="right")
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|endoftext|>"]})


    # CREATE DATASET

    if True:
        with gzip.open('dataset_list_cache.p', 'rb') as inp:
            full_dataset = pickle.load(inp)

        print("Creating Wrapper Dataset")
        wrap_dataset = WrapperDataset.WrapperDataset(full_dataset, paragraph_split=True, return_label=True, device=device)

        wrap_dataset.dataset.update_story_ids()

        total_dataset_length = len(wrap_dataset)
        print("TOTAL DATASET LENGTH: ", total_dataset_length)

        #with gzip.open("wrap_dataset_cache.p", "wb") as f:
        #    np.save(f, wrap_dataset)


    #with gzip.open('wrap_dataset_cache.p', 'rb') as inp:
    #    wrap_dataset = pickle.load(inp)

    # TEST STORY JUDGER MODEL
    data_trial = wrap_dataset[100][None,:]
    print(data_trial.shape)
    print(data_trial)
    encoder = Encoder.Encoder()  # will be pretrained in the future
    storyjudger = story_judger.StoryJudger(encoder)
    outp = storyjudger(data_trial)

    print(outp)


    # TRAIN JUDGER MODEL:
    if False:
        train_dataset_length = 20000
        test_dataset_length = 2000
        remainder = total_dataset_length - train_dataset_length - test_dataset_length
        train_dataset, test_dataset, _ = random_split(wrap_dataset,
                                                      [train_dataset_length, test_dataset_length, remainder])
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, num_workers=16, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=16, )

        # CREATE MODEL

        baseline_pretrained = False
        if baseline_pretrained:
            with gzip.open('baseline_judger.p', 'rb') as inp:
                baseline_judger = pickle.load(inp)
                baseline_judger.lstm.flatten_parameters()
        else:
            baseline_judger = story_judger.StoryJudger().to(device)

        loss_fn = torch.nn.functional.mse_loss

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


        print("examining baseline judger: ")


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

    #TRAIN PARAGRAPH TO PARAGRAPH
    #RIGHT NOW OUTPUTS VEC TO VEC, MAY NEED TO TWEAK
    if False:
        dim_model = 8

        device = "cuda" if torch.cuda.is_available() else "cpu"

        #train_data is array of 9000/batch_size batches, each batch is array of shape batch_size, 2, num_paragraphs (no start and end), dim_model
        #dim_model is dimension of our paragraph embedding
        train_data = generatortrain.generate_unpadded_data(9000, dim_model)
        val_data = generatortrain.generate_unpadded_data(3000, dim_model)

        train_dataloader = generatortrain.batchify_data(train_data)
        val_dataloader = generatortrain.batchify_data(val_data)

        model = generator.Generator(dim_model=dim_model, num_heads=2, num_encoder_layers=3, num_decoder_layers=3, dropout_p=0.1).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss(reduction='none')

        train_loss_list, validation_loss_list = generatortrain.fit(model, opt, loss_fn, train_dataloader, val_dataloader, 2)

        model.eval()
        example = generatortrain.generate_unpadded_data(5, dim_model)[3]
        X, y = torch.Tensor(np.array([example[0]])).to(device), torch.Tensor(np.array([example[1]])).to(device)
        y_input = y[:, :-1]
        y_expected = y[:, 1:]

        tgt_mask = model.get_tgt_mask(y_input.size(1)).to(device)
        pad_mask = model.create_pad_mask(X[:, :, 0], -1)
        pred = model(X, y_input, tgt_mask, src_pad_mask=pad_mask, tgt_pad_mask=pad_mask[:, :-1])
        print(X, y_input)
        print(y_expected, pred)

    #file_name = "fanfics.csv_text_files/94746.txt"
    #new_story = TokenDataset(file_name, tokenizer=tokenizer, padding_side="left")

    # create default ai text gen model
    load_existing = False # turn to true once we already have a model stored on file
    if load_existing:
        ai = aitextgen(model_folder="trained_model4")
    else:
        ai = aitextgen()

    # Train the model! It will save pytorch_model.bin periodically and after completion to the `trained_model` folder.
    # alreay trained: 6560, 43440, 50000,
    print("starting training")
    ai.train(wrap_dataset, output_dir="trained_model_temp",batch_size=1, freeze_layers=True, num_layers_freeze=4, num_steps=1, generate_every=1000, save_every=500, padding_side="right")

    print("Done Training")
    # Generate text from it!
    #ai.generate(10)

    #cuda.cudaDeviceReset()

    #ai2.generate(10, prompt="ROMEO:")
    
    


