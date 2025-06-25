#!/usr/bin/env python
# coding: utf-8

from BartDataset import BartDataset
from CustomBart_Atomic_Tokenizer import CustomBart_Atomic_Tokenizer
from CustomBart_FG_Tokenizer import CustomBart_FG_Tokenizer
from TqdmWrap import TqdmWrap
from DualInputDataset import DualInputDataset
from DualBartModel import DualBartModel, CustomBartModel
from utils import *

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import pandas as pd
import numpy as np
import random
import deepsmiles
from SmilesPE.tokenizer import *
from SmilesPE.pretokenizer import atomwise_tokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from rdkit import Chem
import codecs
from transformers import BartConfig
from tqdm.auto import tqdm
import itertools
import json
import os

from sklearn.metrics import mean_absolute_error


def training_model(combinations):
    for idx, (d_model1, encoder_ffn_dim1, num_attention_heads1, num_hidden_layers1, dropout1, lr1) in enumerate(combinations):

        (d_model2, encoder_ffn_dim2, num_attention_heads2, num_hidden_layers2, dropout2, lr2) = (d_model1, encoder_ffn_dim1, num_attention_heads1, num_hidden_layers1, dropout1, lr1)

        max_r2 = -100

        config1 = BartConfig(
            vocab_size=len(atomic_vocab),
            d_model=d_model1,
            encoder_ffn_dim=encoder_ffn_dim1,
            num_attention_heads=num_attention_heads1,
            num_hidden_layers=num_hidden_layers1,
            pad_token_id=tokenizer1.pad_token_id,
            max_position_embeddings=max_length,
            dropout=dropout1,
        )

        config2 = BartConfig(
            vocab_size=len(fg_vocab),
            d_model=d_model2,
            encoder_ffn_dim=encoder_ffn_dim2,
            num_attention_heads=num_attention_heads2,
            num_hidden_layers=num_hidden_layers2,
            pad_token_id=tokenizer2.pad_token_id,
            max_position_embeddings=max_length,
            dropout=dropout2,
        )

        model = DualBartModel(config1, config2, reg_mod)
        model.to(device)
        model.apply(weights_init)
        optimizer = AdamW(model.parameters(), lr=lr1, weight_decay=weight_decay)

        print(f"Model {idx+1} configurations: ")
        print(f"d_model1: {d_model1}, encoder_ffn_dim1: {encoder_ffn_dim1}, num_attention_heads1: {num_attention_heads1}, num_hidden_layers1: {num_hidden_layers1}")
        print(f"d_model2: {d_model2}, encoder_ffn_dim2: {encoder_ffn_dim2}, num_attention_heads2: {num_attention_heads2}, num_hidden_layers2: {num_hidden_layers2}")

        log_file_path = f"model/new_model2.log"
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        config_dict = {
            'd_model1': d_model1,
            'encoder_ffn_dim1': encoder_ffn_dim1,
            'num_attention_heads1': num_attention_heads1,
            'num_hidden_layers1': num_hidden_layers1,
            'd_model2': d_model2,
            'encoder_ffn_dim2': encoder_ffn_dim2,
            'num_attention_heads2': num_attention_heads2,
            'num_hidden_layers2': num_hidden_layers2,
            'dropout1': dropout1,
            'dropout2': dropout2,
            'lr': lr1,
            'regression_dim': reg_mod,
            'weight_decay': weight_decay,
        }

        with open(log_file_path, 'w') as outfile:
            outfile.write(json.dumps(config_dict) + '\n')

        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0
            all_preds_train = []
            all_labels_train = []

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", dynamic_ncols=True)):
                inputs1, inputs2 = batch['input_ids1'].to(device).long(), batch['input_ids2'].to(device).long()
                attention_mask1, attention_mask2 = batch['attention_mask1'].to(device).long(), batch['attention_mask2'].to(device).long()
                labels = batch['labels'].to(device).float()

                optimizer.zero_grad()

                outputs = model(input_ids1=inputs1, attention_mask1=attention_mask1,
                                input_ids2=inputs2, attention_mask2=attention_mask2)

                pred = outputs
                loss = nn.MSELoss()(pred, labels)
                total_train_loss += loss.item()

                all_preds_train.extend(pred.detach().cpu().numpy())
                all_labels_train.extend(labels.detach().cpu().numpy())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            avg_train_loss = total_train_loss / len(train_dataloader)
            all_preds_train = np.array(all_preds_train)
            all_labels_train = np.array(all_labels_train)
            avg_train_r2 = r2_score(all_labels_train, all_preds_train)
            avg_train_rmse = np.sqrt(mean_squared_error(all_labels_train, all_preds_train))
            avg_train_mae = np.mean(np.abs(all_labels_train - all_preds_train))

            # ------------------ 验证阶段 ------------------
            model.eval()
            total_eval_loss = 0
            all_preds = []
            all_labels = []

            for batch in val_dataloader:
                with torch.no_grad():
                    inputs1, inputs2 = batch['input_ids1'].to(device).long(), batch['input_ids2'].to(device).long()
                    attention_mask1, attention_mask2 = batch['attention_mask1'].to(device).long(), batch['attention_mask2'].to(device).long()
                    labels = batch['labels'].to(device).float()

                    outputs = model(input_ids1=inputs1, attention_mask1=attention_mask1,
                                    input_ids2=inputs2, attention_mask2=attention_mask2)

                    pred = outputs
                    loss = nn.MSELoss()(pred, labels)
                    total_eval_loss += loss.item()

                    all_preds.extend(pred.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)

            avg_val_loss = total_eval_loss / len(val_dataloader)
            avg_val_r2 = r2_score(all_labels, all_preds)
            avg_val_rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
            avg_val_rmae = np.mean(np.abs(all_labels - all_preds)) / (np.mean(np.abs(all_labels)) + 1e-8)

            log_dict = {
                'epoch': int(epoch + 1),
                'avg_train_loss': float(avg_train_loss),
                'avg_train_r2': float(avg_train_r2),
                'avg_train_rmse': float(avg_train_rmse),
                'avg_train_mae': float(avg_train_mae),
                'avg_val_loss': float(avg_val_loss),
                'avg_val_r2': float(avg_val_r2),
                'avg_val_rmse': float(avg_val_rmse),
                'avg_val_rmae': float(avg_val_rmae),
            }


            with open(log_file_path, 'a') as outfile:
                outfile.write(json.dumps(log_dict) + '\n')

            if avg_val_r2 > max_r2:
                torch.save(model.state_dict(), f"model/new_model2.pt")
                max_r2 = avg_val_r2


# ------------------ 启动入口 ------------------
if __name__ == "__main__":
    max_length = 250
    batch_size = 128
    data_path = "data/drug.csv"

    train_dataloader, val_dataloader = train_val_proc(data_path)

    d_models = [128]
    encoder_ffn_dims = [512]
    num_attention_heads = [4]
    num_hidden_layers = [4]
    dropouts = [0.3]
    learning_rates = [5e-5]
    reg_mod = [256, 128]

    weight_decay = 0.001
    num_epochs = 1000

    param_combinations = list(itertools.product(d_models, encoder_ffn_dims, num_attention_heads, num_hidden_layers, dropouts, learning_rates))
    combinations = param_combinations

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    training_model(combinations)
