from aitextgen.TokenDataset import TokenDataset
import numpy as np
import pandas as pd
from aitextgen.TokenDataset import TokenDataset, TokenDatasetList
from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast, AutoTokenizer
import aitextgen
from aitextgen.train import ATGProgressBar, ATGTransformer
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
from datetime import datetime
import story_judger
import torch.optim as optim
from tqdm import tqdm
import Encoder
import EncoderDecoderTraining
import generator
import generatortrain
import pytorch_lightning as pl


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
    # print(tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map['bos_token']))
    # print(tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map['pad_token']))
    # print(tokenizer.decode([0,0,0,0]))
    # print(tokenizer.decode([50256, 50256, 50256, 50256]))
    # exit()


    # CREATE DATASET

    if True:
        with gzip.open('dataset_list_cache.p', 'rb') as inp:
            full_dataset = pickle.load(inp)

        print("Creating Wrapper Dataset")
        wrap_dataset = WrapperDataset.WrapperDataset(full_dataset, paragraph_split=True, return_label=False, device=device)
        #chunk_list_dataset = WrapperDataset.ChunkListDataset(full_dataset, num_chunks=30, return_label=True, device=device)

        wrap_dataset.dataset.update_story_ids()
        #chunk_list_dataset.dataset.update_story_ids() # probably not necessary as the line above does this

        total_dataset_length = len(wrap_dataset)
        print("TOTAL DATASET LENGTH: ",total_dataset_length)





    #with gzip.open('wrap_dataset_cache.p', 'rb') as inp:
    #    wrap_dataset = pickle.load(inp)

    # TRAIN BASELINE JUDGER MODEL:
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

    # Test Encoder DistilBert:
    if True:
        bert_encoder = Encoder.EncoderTransformer()
        data = wrap_dataset[10]
        print(data)
        data=data['input_ids']
        print(data)
        data = data[None,:,]
        print(data)
        bert_encoder(data)


    # Train AutoEncoder:
    if True:
        print("Start training autoencoder")
        initial_decoder_file_path = "trained_model4"
        decoder_file_path = None #"auto_decoder.p"
        encoder_file_path = None #"auto_encoder.p"
        output_dir = "auto2_"
        is_gpu_used = torch.cuda.is_available()
        num_workers = os.cpu_count()
        hparams = dict(
            weight_decay=0.05,
            learning_rate=1e-3,
            adam_epsilon=1e-8,
            warmup_steps=0,
            batch_size=4,
            num_steps= 15000, #12500,
            pin_memory=is_gpu_used,
            num_workers=num_workers,
            save_every= 1000,
            generate_every=0,
            use_tpu=False,
        )
        n_generate = 1
        avg_loss_smoothing = 0.01
        run_id = f"ATG_{datetime.utcnow():%Y%m%d_%H%M%S}"
        progress_bar_refresh_rate= 20
        freeze_layers = False
        num_layers_freeze = 0
        save_gdrive = False
        train_params = dict(
            accumulate_grad_batches=1,
            gpus=-1,
            max_steps=hparams["num_steps"],
            gradient_clip_val=0.5,
            enable_checkpointing=False,  # checkpoint_callback deprecated in pytorch_lighning v1.7
            logger=False,
            enable_model_summary=None,
            # weights_summary and progress_bar_refresh_rate are removed in pytorch_lighning v1.7
            callbacks=[
                ATGProgressBar(
                    hparams["save_every"],
                    hparams["generate_every"],
                    output_dir,
                    n_generate,
                    is_gpu_used,
                    avg_loss_smoothing,
                    run_id,
                    save_gdrive,
                    progress_bar_refresh_rate,
                    freeze_layers,
                    num_layers_freeze,
                )
            ],
            plugins=None,
        )
        print("create trainer")
        trainer = pl.Trainer(**train_params)

        # Wrap the model in a pytorch-lightning module
        print("create model")
        train_model = EncoderDecoderTraining.WrappedAutoEncoder(wrap_dataset, hparams,
                                                                tokenizer, encoder_file_path=encoder_file_path,
                                                                decoder_file_path= decoder_file_path,
                                                                decoder_initial_path= initial_decoder_file_path,
                                                                device=device)

        print("train model")
        trainer.fit(train_model)
        print("TRAINED")

        #print(f"Saving trained model pytorch_model.bin to /{output_dir}")

        #train_model.save_pretrained(output_dir)
        # Implement saving model


    # Train Chunk Story Judger:
    if False:
        print("loading encoder")
        # with gzip.open('auto2_encoder.p', 'rb') as inp:
        #     encoder = pickle.load(inp)
        path = 'auto2_encoder.p'
        encoder = Encoder.Encoder()
        encoder.load_state_dict(torch.load(path))
        #encoder = Encoder.Encoder()
        print("creating storyjudger model")
        save_to_dir = "storyjudger_temp.p"
        storyjudger = story_judger.StoryJudger(encoder).to(device)

        # create datasets, return label bc story judger needs score
        print("creating chunk dataset")
        chunk_list_dataset = WrapperDataset.ChunkListDataset(full_dataset, num_chunks=30, return_label=True, device=device)

        train_dataset_length = 200
        test_dataset_length = 50
        remainder = total_dataset_length - train_dataset_length - test_dataset_length
        print("splitting dataset")
        train_dataset, test_dataset, _ = random_split(chunk_list_dataset,
                                                      [train_dataset_length, test_dataset_length, remainder])
        print("creating dataloader")
        num_workers = os.cpu_count()
        # batch size needs to be 1!
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=num_workers, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=num_workers, )

        # CREATE MODEL

        # baseline_pretrained = False
        # if baseline_pretrained:
        #     with gzip.open('baseline_judger.p', 'rb') as inp:
        #         baseline_judger = pickle.load(inp)
        #         baseline_judger.lstm.flatten_parameters()
        # else:
        #     baseline_judger = story_judger.StoryJudger().to(device)

        loss_fn = torch.nn.functional.mse_loss
        optimizer = optim.Adam(storyjudger.parameters(), lr=5e-3) # 1e-3

        #storyjudger.load_state_dict(torch.load("storyjudger.p"))

        num_epochs = 1
        if True: # TRAINING
            for e in (range(num_epochs)):
                print("Starting Epoch ", e + 1)
                total_loss = 0
                for step, batch in enumerate(tqdm(train_dataloader, position=0, leave=True)):
                    x = batch["input_ids"]
                    y = batch["labels"]
                    #seq_lens = batch["length"]  # for masking and padding
                    optimizer.zero_grad()
                    output_pred = storyjudger(x.to(device))
                    loss = loss_fn(output_pred, y.to(device))
                    loss.backward()
                    optimizer.step()
                    total_loss += torch.mean(loss)

                    if (step + 1) % 50 == 0:
                        try:
                            print("CURRENT LOSS: ", total_loss.item() / 50)
                        except:
                            pass
                        total_loss = 0
                    if (step+1) % 200 ==0 :
                        storyjudger.save_pretrained(save_to_dir)

                print("SAVING JUDGER MODEL: ")

                storyjudger.save_pretrained(save_to_dir)

        print("examining story judger: ")

        print("CALCULATING TEST LOSS: ")

        #storyjudger.load_state_dict(torch.load("storyjudger.p"))


        with torch.no_grad():
            total_loss = 0
            cur_step = 0
            for step, batch in enumerate(tqdm(test_dataloader, position=0, leave=True)):
                x = batch["input_ids"]
                y = batch["labels"]
                #seq_lens = batch["length"]  # for masking and padding
                output_pred = storyjudger(x.to(device))
                loss = loss_fn(output_pred, y.to(device))
                print(loss, y)
                total_loss += torch.mean(loss)
                cur_step = step

            print("num steps: ", cur_step + 1)
            print("TEST LOSS: ", total_loss.item() / (cur_step + 1))


    if False: # Load judger
        storyjudger = story_judger.StoryJudger(None).to(device)
        storyjudger.load_state_dict(torch.load("storyjudger.p"))


    # CHUNK GENERATOR
    # RIGHT NOW OUTPUTS VEC TO VEC, MAY NEED TO TWEAK
    if False:

        # train_data is array of 9000/batch_size batches, each batch is array of shape batch_size, 2, num_paragraphs (no start and end), dim_model
        # dim_model is dimension of our paragraph embedding
        # train_data = generatortrain.generate_unpadded_data(9000, dim_model)
        # val_data = generatortrain.generate_unpadded_data(3000, dim_model)
        #
        # train_dataloader = generatortrain.batchify_data(train_data)
        # val_dataloader = generatortrain.batchify_data(val_data)




        print("Start training chunk generator")

        encoder_file_path = "auto_encoder.p"
        load_chunk_generator_path = "chunk_generator.p"
        output_dir = "chunk_generator.p"
        is_gpu_used = torch.cuda.is_available()
        num_workers = os.cpu_count()
        hparams = dict(
            weight_decay=0.05,
            learning_rate=1e-3,
            adam_epsilon=1e-8,
            warmup_steps=0,
            batch_size=1,
            num_steps=25000,  # 12500,
            pin_memory=is_gpu_used,
            num_workers=num_workers,
            save_every=500,
            generate_every=0,
            use_tpu=False,
        )
        n_generate = 1
        avg_loss_smoothing = 0.01
        run_id = f"ATG_{datetime.utcnow():%Y%m%d_%H%M%S}"
        progress_bar_refresh_rate = 20
        freeze_layers = False
        num_layers_freeze = 0
        save_gdrive = False
        train_params = dict(
            accumulate_grad_batches=1,
            gpus=-1,
            max_steps=hparams["num_steps"],
            gradient_clip_val=0.5,
            enable_checkpointing=False,  # checkpoint_callback deprecated in pytorch_lighning v1.7
            logger=False,
            enable_model_summary=None,
            # weights_summary and progress_bar_refresh_rate are removed in pytorch_lighning v1.7
            callbacks=[
                ATGProgressBar(
                    hparams["save_every"],
                    hparams["generate_every"],
                    output_dir,
                    n_generate,
                    is_gpu_used,
                    avg_loss_smoothing,
                    run_id,
                    save_gdrive,
                    progress_bar_refresh_rate,
                    freeze_layers,
                    num_layers_freeze,
                )
            ],
            plugins=None,
        )
        print("create trainer")
        trainer = pl.Trainer(**train_params)

        # Wrap the model in a pytorch-lightning module
        print("create model")

        # get chunk dataset:
        num_chunks = 30

        train_model = generator.GeneratorTrainer(chunk_list_dataset, hparams, encoder_file_path,
                                                 load_chunk_generator_path, num_chunks=num_chunks, device=device)
        train_model.model.train()

        print("train model")
        trainer.fit(train_model)
        print("TRAINED")

    # GENERATE TEXT
    print("wrap 0: ",chunk_list_dataset[0])
    print("wrap 1: ",chunk_list_dataset[1])

    encoder = Encoder.Encoder()
    encoder.load_state_dict(torch.load("auto_encoder.p"))

    with torch.no_grad():
        for i in range(3):
            x = chunk_list_dataset[i]
            encoder_input = x['input_ids']
            paragraph_embeddings = encoder(torch.tensor(encoder_input, dtype=torch.int32))
            print(paragraph_embeddings)

    text_generator = generator.TextGenerator(generator_file_path="chunk_generator.p", decoder_file_path="auto_decoder.p")
    text_generator()

    exit()

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
    ai.train(wrap_dataset, output_dir="trained_model_temp",batch_size=1, freeze_layers=True, num_layers_freeze=4, num_steps=10, generate_every=1000, save_every=500, padding_side="right")

    print("Done Training")
    # Generate text from it!
    #ai.generate(10)

    #cuda.cudaDeviceReset()

    #ai2.generate(10, prompt="ROMEO:")






