import math
import numpy as np
import random

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import Encoder
import torch.utils.data


def train_loop(model, opt, loss_fn, dataloader, pad=True, pad_token=-1,
               device=None):  # loss_fn has reduction = 'None' iff pad=True
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """

    model.train()
    total_loss = 0

    if device == None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    for batch in dataloader:
        X, y = batch[:, 0], batch[:, 1]
        X, y = torch.tensor(X).to(device), torch.tensor(y).to(device)

        # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
        y_input = y[:, :-1]
        y_expected = y[:, 1:]

        # Get mask to mask out the next words
        sequence_length = y_input.size(1)
        tgt_mask = model.get_tgt_mask(sequence_length).to(device)

        pad_mask = None
        if pad:
            pad_mask = model.create_pad_mask(X[:, :, 0], pad_token)

        # Standard training except we pass in y_input and tgt_mask
        pred = model(X, y_input, tgt_mask, src_pad_mask=pad_mask, tgt_pad_mask=pad_mask[:, :-1])

        if pad:
            not_pad_mask = ~pad_mask[:, 1:, None]
            loss_matrix = loss_fn(pred, y_expected)
            loss_masked = loss_matrix * not_pad_mask
            loss = loss_masked.sum() / not_pad_mask.sum()
        else:
            loss = loss_fn(pred, y_expected)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.detach().item()

    return total_loss / len(dataloader)


def validation_loop(model, loss_fn, dataloader, pad=True, pad_token=-1, device=None):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """

    model.eval()
    total_loss = 0

    if device == None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.no_grad():
        for batch in dataloader:
            X, y = batch[:, 0], batch[:, 1]
            X, y = torch.tensor(X, dtype=torch.float, device=device), torch.tensor(y, dtype=torch.float, device=device)

            # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
            y_input = y[:, :-1]
            y_expected = y[:, 1:]

            # Get mask to mask out the next words
            sequence_length = y_input.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(device)
            pad_mask = None
            if pad:
                pad_mask = model.create_pad_mask(X[:, :, 0], pad_token)

            # Standard training except we pass in y_input and src_mask
            pred = model(X, y_input, tgt_mask, src_pad_mask=pad_mask, tgt_pad_mask=pad_mask[:, :-1])
            if pad:
                not_pad_mask = ~pad_mask[:, 1:, None]
                loss_matrix = loss_fn(pred, y_expected)
                loss_masked = loss_matrix * not_pad_mask
                loss = loss_masked.sum() / not_pad_mask.sum()
            else:
                loss = loss_fn(pred, y_expected)

            total_loss += loss.detach().item()

    return total_loss / len(dataloader)


def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """

    # Used for plotting later on
    train_loss_list, validation_loss_list = [], []

    print("Training and validating model")
    for epoch in range(epochs):
        print("-" * 25, f"Epoch {epoch + 1}", "-" * 25)

        train_loss = train_loop(model, opt, loss_fn, train_dataloader)
        train_loss_list += [train_loss]

        validation_loss = validation_loop(model, loss_fn, val_dataloader)
        validation_loss_list += [validation_loss]

        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print()

    return train_loss_list, validation_loss_list


def generate_random_data(n, dim_model):
    SOS_token = 2 * np.ones((1, dim_model))
    EOS_token = 3 * np.ones((1, dim_model))
    length = 5

    data = []

    for i in range(n // 2):
        X = np.concatenate((SOS_token, np.ones((length, dim_model)), EOS_token), axis=0)
        y = np.concatenate((SOS_token, np.ones((length, dim_model)), EOS_token), axis=0)
        data.append([X, y])

    for i in range(n // 2):
        X = np.concatenate((SOS_token, np.zeros((length, dim_model)), EOS_token), axis=0)
        y = np.concatenate((SOS_token, np.zeros((length, dim_model)), EOS_token), axis=0)
        data.append([X, y])

    np.random.shuffle(data)

    return data


def generate_unpadded_data(n, dim_model, length=11, padding_token=-1):  # no start or end
    data = []
    for i in range(n // 2):
        X = np.zeros((length, dim_model))
        y = np.zeros((length, dim_model))
        X[-1:, :] = padding_token
        y[-1:, :] = padding_token
        data.append([X, y])

    for i in range(n // 2):
        X = np.ones((length, dim_model))
        y = np.ones((length, dim_model))
        X[-3:, :] = padding_token
        y[-3:, :] = padding_token
        data.append([X, y])

    np.random.shuffle(data)

    return data


def batchify_data(data, batch_size=16, padding=False, padding_token=-1):  # ideally batch_size divides data length
    batches = []
    for idx in range(0, len(data), batch_size):
        # We make sure we dont get the last bit if its not batch_size size
        if idx + batch_size < len(data):
            # Here you would need to get the max length of the batch,
            # and normalize the length with the PAD token.
            if padding:
                max_batch_length = 0

                # Get longest sentence in batch
                for seq in data[idx: idx + batch_size]:
                    if len(seq) > max_batch_length:
                        max_batch_length = len(seq)
                        print(len(seq))

                # Append X padding tokens until it reaches the max length
                for seq_idx in range(batch_size):
                    remaining_length = max_batch_length - len(data[idx + seq_idx])
                    data[idx + seq_idx] += [padding_token] * remaining_length

            batches.append(np.array(data[idx: idx + batch_size]).astype(np.single))

    print(f"{len(batches)} batches of size {batch_size} with shape {batches[0].shape}")

    return batches


def encode_batch(book_list, encoder, max_par,
                 device=None):  # every batch is a list of books, every book is a list of dictionaries
    encoded_books = torch.zeros((len(book_list), max_par, encoder.output_dim))  # add encode.output_dim
    # we need to make encoded_books be the pad instead of zeros everywhere that is not
    for b, book in enumerate(book_list):
        for p, paragraph in enumerate(book):
            encoded_books[b, p, :] = encoder(paragraph)
    return encoded_books


def get_and_train_data(model, opt, loss_fn, dataloader, encoder, max_par, device=None):  # data is a list of book_lists
    model.train()

    if device == None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    for i, sample in enumerate(dataloader):
        batch = encode_batch(sample, encoder, max_par)

