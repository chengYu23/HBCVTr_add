#!/usr/bin/env python
# coding: utf-8


from BartDataset import BartDataset
from CustomBart_Atomic_Tokenizer import CustomBart_Atomic_Tokenizer
from CustomBart_FG_Tokenizer import CustomBart_FG_Tokenizer
from TqdmWrap import TqdmWrap
from DualInputDataset import DualInputDataset
from DualBartModel import DualBartModel, CustomBartModel
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, Dataset
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
from transformers import AdamW, BartTokenizer, BartForConditionalGeneration, BartConfig, get_linear_schedule_with_warmup, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, PreTrainedTokenizer
import re
from tqdm.auto import tqdm
from tqdm import tqdm
import itertools
import json
import os

def tokenize_with_progress(tokenizer, smiles_list, **kwargs):
    tokenized_smiles = []
    for smiles in tqdm(smiles_list, desc="Tokenizing"):
        pass
        try:
            tokenized_smiles.append(tokenizer(smiles, **kwargs))
        except Exception as e: 
            pass
    return tokenized_smiles

def batch_encode_with_progress(tokenizer, smiles_list, **kwargs):
    tokenized_smiles = []
    for smiles in tqdm(smiles_list, desc="Tokenizing"):
        pass

        try:
            tokenized_smiles.append(tokenizer(smiles, **kwargs))
        except Exception as e:
            pass
            raise e 
    return tokenizer.pad(tokenized_smiles, padding='max_length', max_length=max_length, return_tensors='pt')

def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
            
def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
            
            
def dataprep(text):
    # 筛选 pACT < 1 的行
    filtered_text = text[text['pACT'] < 1]

    # 提取 smiles 和 labels
    smiles = filtered_text['smiles'].tolist()
    labels = filtered_text['pACT'].tolist()
    labels_array = np.array(labels)

    return smiles, labels_array

def atomic_voc_load():
    file_path = 'data/atomic_vocab.txt'
    atomic_vocab = []

    with open(file_path, 'r') as file:
        for line in file:
            for item in line.split(','):
                atomic_vocab.append(item.strip().strip("'").strip('"'))
    return atomic_vocab

def fg_voc_load():
    file_path = 'data/fg_vocab.txt'
    fg_vocab = []

    with open(file_path, 'r') as file:
        for line in file:
            for item in line.split(','):
                fg_vocab.append(item.strip().strip("'").strip('"'))

    return fg_vocab

def norm_label(labels_array):
    min_val = np.min(labels_array)
    max_val = np.max(labels_array)
    normalized_labels = (labels_array - min_val) / (max_val - min_val)
    labels = normalized_labels.tolist()
    return labels

def data_preproc(smiles_data, labels):

    train_smiles, val_smiles, train_labels, val_labels = train_test_split(smiles_data, labels, test_size=0.2, random_state=47)
    input_encodings1_train = batch_encode_with_progress(tokenizer1, train_smiles, truncation=True, max_length=max_length, padding='max_length')
    input_encodings1_val = batch_encode_with_progress(tokenizer1, val_smiles, truncation=True, max_length=max_length, padding='max_length')

    input_encodings2_train = batch_encode_with_progress(tokenizer2, train_smiles, truncation=True, max_length=max_length, padding='max_length')
    input_encodings2_val = batch_encode_with_progress(tokenizer2, val_smiles, truncation=True, max_length=max_length, padding='max_length')

    train_dataset = DualInputDataset(input_encodings1_train['input_ids'], 
                                     input_encodings1_train['attention_mask'],
                                     input_encodings2_train['input_ids'], 
                                     input_encodings2_train['attention_mask'],
                                     train_labels)
    val_dataset = DualInputDataset(input_encodings1_val['input_ids'], 
                                   input_encodings1_val['attention_mask'],
                                   input_encodings2_val['input_ids'], 
                                   input_encodings2_val['attention_mask'],
                                   val_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader

def train_val_proc(data_path):
    df = pd.read_csv(data_path)
    smiles_data, labels = dataprep(df)

    labels = norm_label(labels)

    train_dataloader, val_dataloader = data_preproc(smiles_data, labels)
    
    return train_dataloader, val_dataloader

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

        log_file_path = f"model/hcv_model.log"

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

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", dynamic_ncols=True)):
                inputs1, inputs2 = batch['input_ids1'].to(device).long(), batch['input_ids2'].to(device).long()
                attention_mask1, attention_mask2 = batch['attention_mask1'].to(device).long(), batch['attention_mask2'].to(device).long()

                labels = batch['labels'].to(device).float()
                optimizer.zero_grad()

                outputs = model(input_ids1=inputs1, attention_mask1=attention_mask1,
                                input_ids2=inputs2, attention_mask2=attention_mask2)
                pred = outputs
                loss = torch.nn.MSELoss()(pred, labels)

                total_train_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            avg_train_loss = total_train_loss / len(train_dataloader)            

            model.eval()
            total_eval_loss = 0
            total_eval_r2 = 0

            # Evaluate data for one epoch
            for batch in val_dataloader:
                with torch.no_grad():
                    inputs1, inputs2 = batch['input_ids1'].to(device).long(), batch['input_ids2'].to(device).long()
                    attention_mask1, attention_mask2 = batch['attention_mask1'].to(device).long(), batch['attention_mask2'].to(device).long()

                    labels = batch['labels'].to(device).float()

                    outputs = model(input_ids1=inputs1, attention_mask1=attention_mask1,
                                    input_ids2=inputs2, attention_mask2=attention_mask2)
                    pred = outputs
                    loss = torch.nn.MSELoss()(pred, labels)
                    total_eval_loss += loss.item()
                    total_eval_r2 += r2_score(labels.cpu().numpy(), pred.cpu().detach().numpy())

            avg_val_loss = total_eval_loss / len(val_dataloader)
            avg_val_r2 = total_eval_r2 / len(val_dataloader)

            log_dict = {
                'epoch': epoch+1,
                'avg_train_loss': avg_train_loss,
                'avg_val_loss': avg_val_loss,
                'avg_val_r2': avg_val_r2,

            }

            with open(log_file_path, 'a') as outfile:
                outfile.write(json.dumps(log_dict) + '\n')

            if avg_val_r2 > max_r2:
                torch.save(model.state_dict(), f"model/hcv_model.pt")
                max_r2 = avg_val_r2
                print(max_r2)
                
atomic_vocab = ['8', '[H+]', '[N-]', '.', '[SbH2]', 'Cl', '[Te]', '%11', '[Se]', '[N@@]', '[c:9]', '[Dy+3]', '[2H]', '[S+]', '[O:3]', '[n:10]', '[K]', '[O:8]', '[Cr+2]', '[s:14]', '[Ni]', '[Pd+3]', '[S:2]', '[C@@H]', '[Zn+]', '[Au+3]', '[CH3:9]', '[CH2:15]', '[P@+]', '[Eu]', '[C@H]', '[Ir+4]', '[cH:13]', '[c-]', '4', '[c:7]', '[S]', '[Os]', '[Ru+3]', '[TeH2]', '[Co+4]', '[Sn+4]', '[S@]', '[15NH2]', '[Cu+]', '[cH:2]', '[Ga+3]', '[CH:2]', '[S:15]', '[nH+]', '[3H]', '[SeH]', '[CH2:13]', '[Zr+4]', '[13CH3]', '(', '[O+]', '[CH3:1]', '[o+]', 'F', '[NH-]', '[I:11]', '[c:5]', '[Ti+3]', '[Li]', '[Fe+2]', '[Cl:11]', '[N@+]', 'p', '[se]', '[n:7]', '1', '[P+]', '[Cu+2]', '[SeH2]', '[CH2:14]', '[Hf+3]', '*', '[CH2:10]', '[Ho+3]', 'N', '[C-]', '[PH]', '[Th+4]', '[NH+]', '%10', '[Fe+3]', '[I+]', '[Ag]', '[Mn+2]', '[Nd]', '3', 'B', '[F-]', '[c:2]', '[P@]', '[n:11]', '[SH2-2]', '[O]', '[n:8]', '[CH2:11]', '[H-]', '[Co+2]', '[c:6]', '[N+]', '[B@-]', '[Ir+3]', '/', '[Sn]', '[n:9]', '[OH:7]', '[c:10]', '[Si]', '[Sn+2]', '%14', '[cH:14]', '[P]', '[SiH]', '[c:3]', '[Th]', '[CH2:12]', '[TeH3]', '[Tm+3]', '%15', '[15n]', '[NH3+]', '[V+4]', '[C@]', '[cH:3]', '[Pd+2]', '[Br+2]', '[Pb+2]', '[SnH+3]', '[Ce+3]', '[cH:10]', '[Ni+6]', '[S-]', '[Cl+3]', 'I', '[c:8]', '[Mn]', '[Ru+6]', '#', '[La+3]', '[Ga]', '[13C]', '[NH2:17]', '[BH3-]', '[Rh+3]', '[C:10]', '[Co+3]', '[C]', '[SH-]', '[CH]', '[Hf+4]', '[Cl:18]', '[SH]', '[Br:16]', '[B-2]', 'O', '[Ag+3]', 'o', 's', '[Ru]', '[Ni+2]', '[W+5]', '[Y+3]', '[CH-]', 'S', '\\', '[Ag+]', 'C', '[13cH]', '[Mg+]', ')', '[Ba+2]', '[cH:7]', '[P@@]', '[SnH2]', '[Pd+]', '[n:4]', '9', '[o:9]', '[cH:16]', '[s+]', '[cH:11]', '[c:4]', '%13', '[Ge+4]', '6', '[CH2:8]', '[cH:8]', '[Pd]', '[Hf]', '[CH2-]', '[Br:11]', '[O:17]', 'Br', '[Pr+3]', '[C@@]', '[13C@@H]', '[N]', '[Pt]', '[13CH2]', '[Au+]', '[c:15]', '[SnH]', '[cH:9]', '[n+]', '[Pt+4]', '[N:10]', '[Ru+5]', '[13CH]', '[Cl:19]', '-', '[SiH2]', '[Zn]', '[Gd+3]', '[N:11]', '[Ti+]', '[CH2:16]', '[OH:9]', '[15nH]', '[n:6]', '[O:16]', '[Pt+2]', '[13C@H]', '[Ta+5]', '2', '[U+]', '[Yb+3]', '[S@@]', '[S-2]', '[cH:5]', '[Rh+2]', '[cH:4]', '[Zn+2]', '[Ru+2]', '[Fe]', '[NH2+]', '[cH:17]', '[Eu+3]', '%12', '[Rh+]', 'c', '[W+2]', '[O-]', '[B-]', '[NH2:18]', '[cH:1]', 'n', '%16', 'P', '[cH-]', '[nH]', '[Cl:10]', '[CH2:9]', '[Ir+]', '=', '5', '7', '[N+2]', '[13c]', '[Er]', '[n-]', '[c:12]', '[cH:6]', '[GeH]', '[S@+]', '[P-]', '[N@]', '_', '[UNK]', '[MASK]']
fg_vocab = ['c3)cc2', '[C@]1(O)', 'c1ccc(', 'c1cccc(Cl)c1', 'C(=O)NC(=O)N', 'c2ccnc(N', 'NCC', 'S(=O)(=O)N(', 'c3c(=O)', '=C1', 'CNC(=O)', 'Cl)c(Cl)c3)', 'c1cnccc1', '[C@]1(C)CC', 'C(C)(C)C)', 'c3=O)', 'c2cccc([N+](=O)[O-])', '(C', 'CCCC1)', '[O:17]', 'c2ccc(OC)', 'C=CC(', 'Cc2ccc(O)cc2)', '[C@H]1CC[C@H]2', 'c(O)c1', 'O[C@H]3', 'CCC(N)', '[NH3+]', 'C(=O)C', 'O=C(N', '-c2cn', '/C=C\\', 'C12', 'c1cccc', 'c1C)', '3)', 'F)cc2)c1', '[SbH2]', 'Cc1c2c(', 'C1N(', 'Cc1ccco1', '[nH]', 'CCc1cc(', 'C(=O)NC)', 'Cl)c1)', 'c2cc(N', '[K]', 'Cc1ccccc1)', 'C2)n1', 'C(N)=O)', 'CCNCC2)', 'C1=C(C)', 'S(=O)(=O)', '[C@]12', '=C(\\', '[15NH2]', 'c4cccnc4)', 'C4', '[C@H]3CC', 'CC[C@@H](', '[C@@H](CO', ')ccc1', 'c2ncccc2', 'C2=C(O)', 'C(F)(F)F)cc3)', 'C1CC(', 'c1cccc(-', 'CC#N)', 'C(=O)c1c(', 'C2CC', 'Cc1', 'nc(N2', 'c1c(O', 'CCN(CC', 'CCCO', 'C(=O)N1CCN(', 'Cl)cc2)', 'Cc1ccc(-n2', '[15nH]', 'CCCC1', 'Cc2ccccc2)CC1', 'C[C@H](NC(=O)', '[Br+2]', 'F)cc(F)', 'Nc1cc(', 'CCCN1', 'C1CCCCC1)', 'Cc1ccc(-', 'c(N4', 'c2)cc1OC', '[C@@](C)(', 'cs2)', 'c3c(c2)', 'CCc3ccccc3)', '/C=N\\', 'c2ccc(-', 'c2cc(Cl)cc(Cl)', '[Th]', 'C(=O)', 'S(=O)(N', 'ccc4', 'c3cc(OC)ccc3', 'OCC', '=O)', 'CC(NC(', 'FC(', 'CCCCCCC', 'NC(=O)C', 'c3cccc4cccc', '[n:10]', '[SH2-2]', 'c2ccc(NC(=O)', 'C)c(', 'C3=N', 'CCN4', 'CCS', '=C(N', ')cc2', 'c4cc5c(', 'F)c(Cl)', 'c1cc(NC(=O)', 'CCC[C@@H]2', '[I:11]', 'Cl)cc2', 'c2ccc(O)cc2)', '(C)C)cc1', 'CCCCC3', 'Br)cc2)', 'C1C(', 'C1CCCC', 'C1CCCCC1', 'C2)c1', 'CC(NC(=O)', '-c2cc(', 'O1)', 'cc2c(', 'O3)', 'C(C(F)(F)F)', 'CCC1', 'Cn1', 'CN(C(=O)', 'c3Cl)', 'c(-n2', 'c1c(NC(=O)', '3)cc2', 'C(C)(C)O)', 'c1c(-', 'c2cccn', 'c3c(cc(', 'S)=N', '[Co+3]', 'CC2=O', 'c6ccccc6', 'c3cc(-', '[c:5]', 'c1ccc(C(N', 'c2ccco', 'CN(CC)', 'c(=O)n2', 'OC(C)(C)C)', 'C(CO)', 'CCCCO', '[NH2:17]', '[N+](C)(C)C)', '[13CH3]', '4)c3)', 'CC1CC1)', 'OC[C@H]1', 'CCN1CCC(', 'C(=O)NO)', 'c4ncc', '[C@@H](C)C', 'ccccc4', 'C1(C)C', '(C)C)', '[3H]', 'c2ccc3[nH]', '[C@H]1C', 'O=C(O)', 'S1', '/C=C(\\', 'nc(-c3ccccc3)', '/C1', 'CN1C(=O)', 'C1(c2ccccc2)', '[Mg+]', 'c(C)', 'ncc2', '[C@H](O', 'cn2)CC1', 'N3CCOCC3)', 'c12', 'CCCC2', 'c1cc(=O)', 'c3c(n2)', 'cc4c(', 'C2C(', 'C2(C)C', 'NC(', '/C1=C/', 'ccc(C)', 'cc(N', 'CN2CC', 'nc(NC(=O)', 'c1ncc(', '3)cc(', 'c3n2)', '[C@H](CC(C)C)', 'C(=O)N1CC', '[N+](=O)[O-])ccc1', ')cc(OC)c1OC', 'C(CS', 'c4ccc(F)cc4', '[C@H]1[C@H](', 'COc1cccc(', '[C@H]2O', '[N+]1', 'Cl)', 'SC(', 'CCC1)', '/N=C2\\', 'CCC[C@H]2', 'nc(S', 'Nc1cc', 'CN(CC', 'c1cnc(N', '[Ni]', 'COc1ccc2c(c1)', 'CC4)cc3)', 'CCOC(', 'c1ccccc1)', 'C1=C(O)', 'c1cc(Br)ccc1', 'OCO5)', 'CCOCC1)', '-c1cc(', 'oc1', '[Fe+3]', '[nH]cc2', 'COc1cc2', 'Clc1ccc(', '[C@@]2', 'c2ccc(OCC', 'Nc1nc(', 'C(C)C)cc1', 'N1CCN(C', '/C(C)', 'C(/C=C/', 'CCCCCCN', '/C=C/C', 'c1ccc(C(F)(F)F)cc1', 'C1CCN(C', '(C)c1', 'nc(-c2ccccc2)', 'OCc3ccccc3)', 'Nc1nc(N', 's2)c1', 'OCC)', 'OC[C@H]1O[C@@H](', 'c1c(O)cc(', 'N(C(=O)N', 'CCCCC1', '[C]', 'cc21', 'COc1ccc2c(', 'nc1)', '[C@@H](OC(C)=O)', 'c1cc(C2', '=O)ccc(', 'COc1ccccc1N1CCN(', 'NC(=O)', 'CC(c2ccccc2)', 'COc1ccc(-', 'nc12', 'c1c(C(F)(F)F)', '[cH:14]', '[NH2:18]', '[S@@]', 'c1cc(-n2', '/C=N/NC(=O)', 'c3n(', 'n1c(N', '=[N+]=[N-]', 'c2ccccc2O)', 'nnc(', 'c1ccc(C(', 'S(=O)(=O)N3', 'CCCCC2', 'c1c(C)cc(', 'N=C(N)', 's2)cc1', 'c2n(C)', '4)cc3)', 'C1=N', 'COC', 'CC(=O)N3', 'CCO)', 'c(-', 'N1C', 'CCCCC)', 'c2cn(C)', 'C[C@H]4', 'c1cc(-c2n', '=[N+]', '[Ir+3]', 'c2cc3c(cc2', '[n:4]', 'o2)c1', 'c3ccc(OC)cc3', 'Nc1nc(-', '[C@@H](CO)', 'c4cccc(', 'CCN(CC)C(=O)', '[Y+3]', 'C4=O)', 'C[C@@H](NC(=O)', 'cccc2)cc1', 'c5ccccc5', 'C2=O', 'C(=O)N/N=C/', 'Cc2n', '[OH:7]', 'c3ccc(', 'COc1ccc(-n2', 'n3cc', '/C(C)=N/', 'c2cccc(F)c2)', '[C@@H](OC(=O)', 'c1ccc(Br)cc1)', 'cc3)cc', '[C@H](O)[C@@H]1O', 'Cn1cc', '[Ho+3]', '[OH:9]', 'CC1(', '[C@]5', 'Cc1cc(C)cc(', 'c2cccc(O', 'c1cnc(', '[O-])', '[C@H]1O', 'CCCN', 'c1cnn2', 'c2c(cc(', 'c4cc(Cl)cc', '[Eu]', 'CC2CCCCC2)', 'C(CNC(=O)', '[C@@]1(', '[O]', 'Cc1o', 'N)=O)', 'cccn', ')cc(O)', 'CCN(C(', 'c3c1', 'Cl)cc(', 'Nc1cccc(', 'cn', '[cH:13]', '[Cr+2]', 'C/C=C\\', '[C@@]5', '7', '[Nd]', 'cn3)', 'CCCO2)', 'c1ccc(S', 'CO2)', '[c-]', 'cn1', 'c1oc(', 'CC2)cc1)', 'c1ccc([N+](=O)[O-])cc1', 'c3n', 'c(=O)n1', 'CCOC1', '=O)cc(', 'c4cccc(Cl)', 'c3)', 'n2c(', '[Yb+3]', 'C(C)C)c1', 'c3cccc(Cl)', 'c(C(F)(F)F)', 'CC(=O)N(C)', 'c1cc2cc(', 'CC2)ccc1', '=O)n(', 'C5', 'c3cc(OC)c(OC)cc3', 'CC1)', 'c1cc(F)cc', 'c1c(NC(', '[C@]2', '[S:15]', 'c2c(', ')cc2)cc1', '-c2o', 'cc3c(', '[C@]4(C)', 'Cc1ccco1)', 'C(C)(C)[C@@H]5', '[n:11]', 'C(N1CCN(', 'c3c(Cl)', 'ccn1', 'CC(C(=O)N', 'Cc3ccc(', 'CC=C)', 'Cn1c2c(', 'cs3)', 'c1s', 'c1cccc(C)c1', '[C@H](CO)', 'c(C)cc1', 'Br)ccc1', 'c3ccc(Cl)c(Cl)c3)', '=O', 'C(C(=O)O)', 'CC(CO)', 'C(=', 'N1CCOCC1', 'c3cc(C(F)(F)F)cc', 'C)(', 'c3cn(C)', 'CC=', 'c1c(N', '[Ti+3]', 'c(F)cc1', 'C(F)(F)C(F)(F)', 'c3c(-', 'Br)c1', '=O)C', 'c3o', 'OC)c(OC)', 'CC(C)O', 'CCOC)', 'Cc1ccccc1', 'c2c1', 'c3cc(Cl)ccc3', 'c3cccc(C(F)(F)F)c3)', 'c(Br)', '[2H]', 'N(C(=O)', 'n3)cc2', 'CC2=O)', 'CC#', 'C(NC(=O)', 'ncc3', 'c1c(C(', 'nc3n2', 'c1ccc(C)', '[C@](C)(O)', '[S]', 'O[C@H](', 'C(=O)O)c1', '[C@@H](NC(', 'c1c2c(', '[C@H](C(C)C)', 'nc(NC(', 'c2)nc1', 'C2=O)c1', 'c2)cc(', 'C(=O)NC(=O)', '[C@H](C)CO)', 'CC[C@]4(C)', 'Cc1cn(', 'S)N', '/C=C2', 'NC1', 'c3cccn', 'nc3c2', 'oc2', '[N+](=O)[O-])', 'c3ccccc3C)', 'cc(-c3ccccc3)', '[nH]c2c1', 'OCC3', '[Hf]', '[c:8]', 'c', 'cc1-', 'C(F)(F)F', 'CC)cc1', '[N+]([O-])', 'CCC(C)(C)', 'c2cn', '[cH:6]', 'c(O', 'n2C', 'C3)cc2)', 'c4c3)', 'Cc1ccc(-c2n', 'n(C)c1', 'C2=O)', '[B-]', 'c2ccc(C(F)(F)F)cc2)', '[n:9]', 'c3C)', 'c2cc(F)', 'C1c2c(', '3)CC1', '2)ccc1', 'c1c(OC)', 'C3CCCC3)', 'CCn2', 'O[C@H](CO', 'O=C(/C=C/', 'C=C', 'c3c(OC)', 'C3CCCCC3', 'Br)', '[C@H]2', 'n2cnc3c(', '5', '[c:3]', 'c2ccccc2)ccc1', 'nc1', '[Si]', '/N=C(\\C)', '(c1ccccc1)', 'o1', 'Cc1n', 'o1)', 'c3ccccc3)cc', 'OC(C)(C)C', 'c23)cc1', 'COC(C)=O)', 'C#C', 'CCC2(', '[O:16]', 'c1c(Cl)c(', 'c2cc', 'S(=O)(=O)O)', 'CCN(C)C)', 'c2cc(Cl)cc', '[Pr+3]', 'cccc', 'c(C', '[N+](=O)[O-]', 'c2)', 'Cc1ccc(C(=O)N', '[nH]c3', 'C)ccc1', 'c(Cl)', 'COc2ccc(', 'C1CCC(', 'OCCO3)', 'c3-', '[CH2:13]', 'CC1', 'CCC(O)(', 'F)cc(', 'c1ncc', 'c3ccc(C)cc3', '[nH]c(=O)', '/C=C/C(', '=N', 'nc2ccccc12', 'cc(=O)', 'Cc1ccc(N', 'ccc1', 'CN3', 'N2', 'C=C3', '3)cc', 'nccc1', 'C(OCC', '[N+](=O)[O-])cc2', 'c3ccc([N+](=O)[O-]', 'c3ccncc3)', '[C@]', 'CC(C)C)', '[C@@]3(', 'c1cc2cccc', 'c(O)', 'NC(N)', '3)cc2)CC1', 'N[C@@H](', 'C[C@H](N', 'nc4)', '-c2[nH]', 'c2cc(Cl)c(', 'c2ccco2)', 'CCCN=C(N)N)', '[13C@@H]', 'nn(C', 'c2c(-', 'C1=C(C)N', 'c2ccc(Cl)cc2)cc1', 'nc(Cl)', 'c3nc(-', 'n2cc(', 'cn2)c1', 'cnc3', ')cc(OC)c1', 'c4ccccc4)CC3)', 'C[C@H](', 'OCO4', 'c(=O)[nH]c1=O', 'c3cccc(C)c3)', 'cc(O)', '[C@H]4', '[C@@]', '[SnH+3]', 'co', 'cc(NC(=O)', 'cc(OC)', 'c2cc(', 'C(F)(F)F)cc2', '2)c1', '[C@H](O)', '[C@H](', '(C)(', 'O=', 'o', 'c2=O)cc1', '%12', 'c1sc(', 'CNC', 'n2)C1', 'C(C', 'COCCO', 'CC[C@@H]2', 'c2cc(O)', 'CCC(=O)O)', '[C@@]2(C)', 'C(=O)N(CC)', '-c2ccc(-', '[cH:10]', 'c1ccncc1', '(=O)', 'c1c[nH]', 'B(O)', 'c4ccncc', 'NC(=O)[C@@H]1', '(O', 'OC(=O)', 'cn2', 'C(=O)N1CCC(', 'C(=O)OCC)', 'c2ccc(F)cc2)', 'Cn2c(=O)', 'C2CCN(C', '-2', 'c1c(F)', 'CC', 'c4ccc(C)cc4)', 'CC(C)(C)C', 'c4)cc', 'N#Cc1cccc(', '[C@@H](O)[C@H](O)', 'c12)', 'OC(C)=O)', 'Cc1cs', 'c3cccc(OC)c3)', 'C(F)(F)F)cc1', 'c(N(C)C)', '[O:3]', 'c1ccc2c(c1)OCO2)', 'P(=O)(O)O)', '[Pt+4]', 'N[C@H]1', 'OCC(', 'N4', 'c2c(=O)n1', 'Cl)cc2Cl)', 'CCC[C@H]1', 'N1)', '[C@@H]1C', '-c2n', 'c34)', 'c2ccs', 'c2ncnc3', 'Cc1c(C)', 'Oc1ccccc1', '[Hf+3]', 'C12CC3CC(CC(C3)', 'CC(=O)N1', 'nn(C)', 'O)', 'P(O)(O)', 'C(=O)C2', 'C)C1', 'c3cc(OC)c(OC)c(OC)c3)', 'c2ccc(OC)cc2', '[C@H](C)', '[C@@](O)(', 'c1c(F)cccc1', 'c3ccccc23)', 'CC12', '3c(', '[C@@H]1', '[Ru+5]', 'CSc2n', 'c[n+](', 'F)cc', 'C1O', 'CC(=O)N1CCN(', '.', 'c3ccccc3)CC2)', '[cH-]', 'C[S+]([O-])', 'c1ccc2c(c1)OCO2', 'c(F)cc2', '3)cc1)', 'c1ccc(-c2n', '[C@H](N)', 'CN(Cc1ccccc1)', 'Cc2cn', 'Cc2ccc(OC)cc2)', 'C1CC', 'OCO', '[C@@]3(C)CC', '[CH2:16]', '[C@@]2(', 'C21', 'C(=O)CS', 'Cl)cc3)', 'n(CC(=O)N', 'c1ccccn1', '%15', '/N=N/', 'ccc(OC)', 'cn3', 'nc3c(', 'CCCC)', 'Cl)cc3', 'CCC3', 'c3cccc(Cl)c3)', 'cc(C)c1', '[Ag+]', '[C@@]1(O)', 'C(C(C)C)', 'C(C)=O', 'c3s', 'CCNCC1', '/C=C2/', '/C=C/C=C/', 'c1c(O)ccc(', 'c1ccc(S(=O)(=O)N', '[C@@H]12', 'c2c(O)', 'n3', 'COc1ccc(Cl)cc1', 'C(NC(', 'n1c(', 'ncn3)', '[C@@]3', '=CC(=O)', 'c1n(', 'S(=O)(', 'C23CC4', '2)cc(', '[S-2]', 'CC=C(C)C)', 'Cc1ccc(', '[CH2:9]', '[N+](=O)[O-])cc2)', 'sc1', 'OC)cc2)', 'Cl)c(Cl)c1)', 'CC[C@]4(C)[C@H]3CC', '(-c3ccccc3)', 'F', 'c(=O)n(', 'Cc1c[nH]', 'CC(F)(F)', '[C@H](OC)', 'c3c(', '[N+2]', 'OC)cc(OC)', 'NS(=O)(=O)c1ccc(', 'C2CC(', 'C(=O)NC2', '[cH:16]', '2)cc1)', 'CNC(=O)N', '[Pd+]', 'c3cc(C)', 'C[C@]12', '[C@@H]2C1', 'S2(=O)=O', 'C(F)(F)F)cc2)', '[La+3]', '[se]', '=C2', '[SH-]', 's1)', 'c(C(N)=O)', 'C=C1', 'COC(', 'c1nnc(', 'c3cccc(', 'C2)cc1', 'c4[nH]', '[P@]', 'O=S1(=O)', 'Cc1ccc(-c2ccccc2', 'CC(CO', 'c4cn', 'c4ccc(OC)', '[NH+]', 'c1cccc(Cl)c1)', 'c3ccccn3)', '9', '[P@+]', 'C1=O)', 'c4ccc(N', 'c13)', 'N1CCC(', 'c2cc3cccc', 'C2)C3)', 'c1cc(Cl)cc(', 'N1CCOCC1)', '/C=C(/', '/C(=C\\', 'C(=O)CC', 'COc1ccccc1', 'n3C)', 'c4)CC3)', 'S(=O)(=O)c1ccc(', 'CCOCC', 'c1)N', 'c3cc4c(cc3', '[I+]', '[C@@H](O)[C@@H](O)', '=O)cc1', 'c2n', 'CC(N)=O)', '[N+](C)(C)', '/N=C1\\', 'Br)c1)', 'CCCN(C', 'c3cc(', 'c5cc(', 'c(=O)n3', '[N+](=O)[O-])cc1', 'C(C)', 'CCCN(C)C)', 'Cl)c(', 'cnc1', 'c1cc(Br)', '(Cl)', 'cccc3)', 'c1O)', '[CH]', 'nc1C', ')cc2)', 'C1N', '5)cc', 'c3ccccc3)CC2)cc1', 'Cc1cc(', 'CCSC)', '[Br:16]', 'CC2)C1', '=O)[nH]', 'c(N', '-c2ccccc2)', '[C@@]34', 'C[C@@H](', 'CN2CCN(', 'C3=', 'c3cc(Br)cc', 'CCCCCCCCCCCCCCCC', 'O4)', '[C@@H]1[C@@H](', 'ccn3)', 'OC)c(OC)cc1', 'CO', 'c3cccc(-', 'C#N)cc2)', 'CC[C@]3(C)', '[Cl:18]', 'c3c2', 'N2C(=O)', 'Cc2ccc(F)cc2)', 'O=C1N', 's3)', 'O[C@H](CO)[C@@H](O)[C@H](O)', 'n(C(C)C)', 'N1CC', '/C(=N/', 'N2CCCCC2)', 'C)cc1)', 'C(=O)N[C@@H]1', 'NC(=O)[C@H](CC(C)C)', 'O)c(O)', 'N(CC)CC)', 'C(=O)N2', 'CCN(S(=O)(=O)', 'OCc1ccccc1', 'c2c(Cl)cccc2Cl)', 'n(C)', 'S)', 'ccccc1', 'C#N)cc1)', 'C(=O)O)cc1)', '[SeH2]', 'N(C)C(=O)', '/', 'CC(C)C', 'c2ccc(-c3ccccc3)cc2)', 'cn4)',
            'n(CC', '[Se]', 'c(=O)c(', 'c(-c4ccccc4)', 'n[nH]1', 'c2cc3', 'c2nc(-', 'Cc1no', '[CH2-]', '[Ba+2]', 'C(c3ccccc3)', 'OC[C@H]2', '/C(=C/', 'c2cccc(C)c2)', 'C)CC1', 'CC(O)(', 'c1ccccc1)N', 'c1cccc(C(F)(F)F)c1', 'c1', 'c3ccc(C(F)(F)F)cc', 'c1ccc(Cl)', 'sc2c1', '[nH]c12', 'CCN1', 'N#', 'c3nc4ccccc4', 'cc4', 'c(NC(', 'C3=O)', '-c1c(', 'C1CC2', '[Ir+4]', 'CC(C)=', 'C(F)(F)F)c1', '[Ru+2]', 'C)O', 'c1)C', 'c3cc(O)', 'c3ccc(OC)cc3)', '(-c2ccccc2)', 'c2cc(C)cc(C)', 'c2cc(C)ccc2', 'CC3)cc', 'N3', 'C(=N)N)', 'c2nc3ccccc3', '4CCCC', 'CCOC(=O)', 'oc(', 'S(=O)(=O)N', 'N2CCN(C(=O)', 'OC[C@@H]2', 'C[C@H]3', 'CCCC(C)', '[PH]', 'C5)', 'CCc2c(', 'c2c(N)', '[Rh+3]', 'c2cc(Br)cc', 'c3cccs3)', 'CCCC(=O)', '[Cu+2]', 'CC(C)(C)N', 'CCNCC3)', 'c3ccc(O)cc3)', '/C=C1/', '[C@H](C)C', 'cc3', 'c3cs', 'c2ccc(N(C)C)', 'OC)c(', '[nH]c1', 'C)cc1', 'co1', '4CCCC4)', 'c3ccccc23)cc1', 'cc1', 'C1(C)', 'c2[nH]1', '(-', 'n2)', 'OCCCO', 'Oc1ccc(', 'C1)C2', '/C(=C(/', 'c2c1cccc2', 'OC4', 'ncnc2', 'SC', 'ncnc3', 'cccc4)', 'CCCC', 'C)', 'CCC(N3', 'c3nnn', 'c1cccnc1)', '[Te]', 'CCC2)', 'C=', 'C2(', 'Cc1cccc2', 'N)c1', 'C/C(=N\\', 'CO1', 'C3CCC3)', 'sc(', 'cccc(', 'S1(=O)', 'c1nc(N', 'C(N(', 'CC4)', '[C@H](Cc2ccccc2)', '/C=C/C(=O)N', 'c(S', 'c1c(OC)cc(', '=C', 'C(=N)N', 'c4c(', '[Pd]', '/C(', 'CC(C)(C)C)', 'c3cccc', '[Cl+3]', '5)cc4)', 'c5[nH]', '[P]', 'Cl)cc', 'c2)ccc1', 'CS(=O)(=O)N', 'c1ccc(F)cc1', 'C(C)(O)', '[C@H]1O)', '[Eu+3]', 'C(=O)NC(', 'CCN(C)CC2)', '5)', 'C)cc(', 'C(C(', 'C(C)=O)', '[c:12]', 'c4ccc(-', 'Cc1nc2c(', 'c3c2)', 'Cc2ccc(C)cc2)', 'N)cc1', 'c3cccc(N', 'c2cc1OC', '[C@@]2(O)', '-c2nc(-', 'c3cc(Cl)cc', 'nc2-', 'c1c(C)ccc(', 'c4ccco4)', 'c4', '=N\\', 'c2ncnc(N', 'ns', 'c1ccc(-c2ccccc2)cc1', 'Cn1cn', '[s:14]', 'c2-', 'CCN2', 'c2cc(-', '[C@@H]2CC', '#', 'n(-', 'CCC(C(=O)N', '[C@@](', 'C1CN(C(=O)', 'c1cc(OC)ccc1', 'C[C@@H]2', '[Ni+2]', 'C[C@@H]1', 'CCn1cc(', 'N=', 'cs2)cc1', 'c1ccc2ccccc2c1', 'c-2', '=C(/', 'C2CC2)', 'ncn2', '[N@]', 's2)', 'c1n', 'CNC(', '=O)CC2)', 'c3cc4c(', 'O=C(N/N=C/', 'c1ccc(Br)cc1', 'Nc1n', 'C(N1', 'c2c(=O)', 'CN(C)', 'c3ccc(Cl)cc3)', 'O=C1N(', 'C1CCN(', '=O)cc2', 'c2ccccc2)CC1', 'c2ccccc2)n1', 'c(OC)c3)', 'c2ccncc2', '[Sn+2]', 'O=c1', 'c4cc5', '/C', 'C(=O)N2CCN(', '[N+](C)', 'c1cc(C(=O)N', 'cc2C)', 'C2CCCC', 'CCOc1cc(', 'Cn2cc', 'OC)c1OC', '[SH]', '[N]', 'c(C)cc(', 'c2C1', 'C1CCN(C(=O)', 'C23', 'F)cc2)cc1', 'c3ccccc3', 'C3=C(', 'c(C)c1)', 'N(C)C)', 'c2cc(Cl)ccc2', 'C(c1cc(', 'c2nc1', 'COc1ccc(', 'C(CCC)', 'CCOCC2)c1', 'c2ccn', 'c2cc(C(=O)N', 'c3cc(C)ccc3', 'CCCO3)', 'CC(F)(F)F)', 'ccc21', 'c4c5c(', 'cccc5)', 'c1cn(C)', 'C/C=C/', 'c1cc(C)ccc1', 'Cc1cccc(', 'ncn1', 'N(CCCl)CC', 'C3)C2', 'c2ccc(Cl)cc2Cl)', 'CN(C)C', '[CH2:14]', 'C2CCCCC2)', 'OC)c(OC)c1', '[C@H]', 'n3c(', 'C1=S', '[C@]1', '[SnH2]', 'c1cc(Cl)cc', 'C/C(=C\\', '[W+2]', 'c4cc', 'C(N', '=O)cc2)', 'COc1ccc(-c2n', 'OCCO2)', 'sc1)', '[SeH]', 'CC(C)', 'Cc2ccccc2', 'CCCC(=O)N', 'CC(C)C[C@H](NC(=O)', 'c1ccc(C(=O)N', '[Ru+6]', 'C1C2(', 'CCCCC2)', 'NCCCC', 'c2ccccc2C1', '[C@H](C(=O)N', '[C@](C)(', '[nH]1)', '-c2nc(N', '[SiH]', 'c21)', 'P(O', 'nc2)cc1', 'C(=O)N[C@@H](', 'CCc1ccccc1', 'CCCCN(', '[N:11]', '(N)', 'c1c(F)c(', '[C@]12C', 'CCCN)', 'c3ccccc3-', '[c:9]', 'c3ccs', 'c1ccc(N', 'ncn', 'on2)', 'c([N+](=O)[O-])', '[C@@H]4', 'CC1CCN(', 'C(=O)NCCCC', 'n1cc', '=O)CC1', 'c1cccc(N', '[C@@H]1O', 'OC)c(OC)c1)', 'c2ccc3ccccc3', '=N2', 'n(C)c(=O)n(C)', 'P(', 'c(N)n1', 'c3ccc(F)cc3', 'c(N)', 'ccnc1', 'C3CC3)', 'ccc(C)c1', 'c2ccc(Br)cc2)', 'CC3)', 'c1ccc(O)cc1', 'c1sccc1', 'c4ccco', 'C1C2', 'C(F)(F)', 'C=C\\', 'Cn3', '/N=', 'NS(=O)(=O)', 'C(', 'nc3ccccc3', '[Cl:11]', 'C(C)C)', 'C(C)(C)C', 'NC(=O)c1ccccc1', '(C)(C)', 'nc(C)c1', 'n4)', 'N2CCCC', 'c(OC)c1)', 'CCOCC3)cc2)', 'nc(C(F)(F)F)', '%10', 'c1cc(C(', 'CCCCCCCCCC', 'cnc12', 'c(O)cc1', 'CCC4', 'c1ccc(C2', 'c3nc(N', '[S-]', 'c3ccc([N+](=O)[O-])cc3)', 'C(=O)O1', 'CCOCC1', 'F)cc1', '[B@-]', 'c2cc(Cl)', 'NC(=O)c1cc(', 'c2s', 'C#N', 'cc2c1', 'C=O)', 'nc(O)', 'ncc1)', 'c1cc(O)c(O)', '[nH]c2)', 'nc2c(', 'c3cc4', 'c1N', 'CCCNC(=O)', '-c2cc', '=C\\', 'c3nc(', 'NC(=O)c1ccc(', 'C3=O', 'cc3C)', 'c1cc(Cl)', '(=O)N', '/C2', '[C@](', 'c2cccc(C(F)(F)F)c2)', 'c2cc(=O)', 'Br)cc', 'c3cccc(O)', 'c2c[nH]c3ccccc23)', 'CC(O)', 'c2ccc(OC', '/N=C(/', 'c1nc(-c2ccccc2)', '[C@@H](O', 'O[C@@H]1', '[Er]', 'c(C)c(C)', 'O[C@H](C)', 'c3ccc(F)cc3)', 'c3ccc(N', 'C[C@H](NC(', 'CC[C@H](', 'c(OC)c1', 'C(C)(C)C)cc1', '=C(C)C)', 'c5cn', 'ccc(', 'C(=O)N1CCN(C(=O)', 'N3C(=O)', 'nn12', 'c3cccnc3)', 'O[C@H]2', 'N=C1', 'C(C(=O)', 'Cc1ccc(NC(=O)', 'Cn2', 'c4c(Cl)', 'c2ccccn2)', 'CCCNC(=N)N)', 'C(F)(F)F)c1)', '[C@H]2[C@@H](', 'p', 'n1cn', '(=O)o', 'sc3', 'CCC(=O)N', 'c([N+](=O)[O-])c1', 'COc1cc(OC)', 'c1ccccc1Cl', 'Cc1s', 'c1cncc(', 'c3ccccc3)', '[Co+2]', 'C[C@@H]3', 'c4n(', 'C1)', '/C=C1\\', 'nc2C)', 'c32)', 'OCC1', 'c2ccc(C)cc2', 'c2sc3c(', 'CN1CC', '/N=C/', 'OCCN(', 'c1c(OCC', 'CCCN(', 'CCNC(=O)', 'c1c(C2', 'c3ccco3)', 'c2ccc(N', '-c1n', 'CCC[C@@H](', 'c5ccccc5)', 'CCN(C)CC1)', 'nc2n1', 'CC[C@H](C)', 'Cc1ccc(O', 'N)ncn', 'c2c[nH]', 'c3ncccc3', 'c1ccccc1F', 'CC2)c1', 'cc2)cc1', 'COc1cccc2', '-c2c(', 'CC3(', 'C(C)=C(', 'nn1', 'N=C(', 'CC(=O)O)', 'c1ccc(Cl)c(', 'CC2CC2)', '[C@H](Cc1ccccc1)NC(=O)', '[C@@H](CC)', 'ccc12', 'N(c2ccccc2)', 'c2)cc1', 'OCCO4)', 'C1CC1)', 'c3ccccc3C2=O)', 'c1c(OC)ccc(', 'cccc4', 'c1cccc(O)c1', '/C=C/', '[C@@H]2O', 'P(=O)(O', '[Zn+]', 'C(=O)N1CCC[C@H]1', '[s+]', 'cn(', 'ncc4', 'c3ccccc3Cl)', '(=O)=O)cc1', 'c2ccc3ccccc3c2)', '[C@H]5', 'Br)cc2', 'CN1CCC(', 'COC(=O)', 'c2c1)', 'CCC=C(C)C)', '[13C@H]', 'N2CCC(', 'c6', 'c1ccc(Cl)cc1Cl', 'OCCO', 'Cc2ccc(Cl)cc2)', 'cnn2', 'Fc1ccc(', 'C(F)(F)F)c(', 'Nc1c(', '/C=N/', 'c1cccc([N+](=O)[O-])c1', 'nc2cc1', 'c1ccc(OC)', 'c2nc3c(', 'c1cn2', 'ccccc12', 'c2ccc(O)cc2', 'c3ncccn3)', 'c3ncn', 'nn1)', 'F)c(F)', 'C[C@H]2', 'c3cc(N', 'OC)c1', 'c2ccccc21', 'n(-c2ccccc2)', 'CC)', 'Cn1cc(', 'c3ccccc13)', 'CC2)n1', '[H-]', 'OCc1ccccc1)', 'c1co', 'oc(=O)', ')ccc1O', 'C(C)C)cc2)', 'cs2)c1', 'OCC2', 'cc(Cl)', 'c1cccc(C)', 'S2', '[N@@]', 'CCCC(=O)O)', 'c1ncccc1', 'CCO', 'C(CCCC', 'C(=O)N[C@H]1', 'c1c(Cl)', 'CC2)cc1', 'C1=C(', 'nc2N', 'c3cc(OC)c(OC)c(OC)', '[C@]2(C)', 'OC1=O', 'C(c2ccccc2)', 'C(F)(F)F)', 'C1CO', '[C@@H]3', 'CC[C@H]4', 'c3ccc4c(', 'C[C@@H](O)', '[C@@H]5', 'C(=C/', 'CCC2(CC1)', 'CCC(=O)', ')ccc3', 'COc1cc2c(', 'c2cnn3', 'c3ccc4c(c3)', 'C(N)', 'FC(F)(', '=C3', 'S', 'c2ccc(OC)c(OC)', 'c1c(-n2', 'c3c4c(', 'c2cc(OC)ccc2', 'c(C(=O)O)', 'nc(SC', 'c2nc(N', 'CCN(CC)', '[S@+]', '[cH:5]', 'cc1)', 'O=C(CS', 'CCN(', 'OCC(=O)', 'c2ccccc12', '[Zn]', 'P(=O)(', '[Sn+4]', 'c4s', 'N2CCN(', 'c1cccc(-c2n', 'C(=C)', 'C(C)=O)cc1', '[Rh+2]', 'NC1=O', 'CCCCC2)cc1', 'CC5)', 'c2cccc3ccccc23)', 'c2ccccc2c1)', 'n(', 'C(C#N)', '[C@@H](C', 'n2)cc(', 'c1cc(', 'c2ccc(-n3', '[C@@H](O)[C@H]1O', '[GeH]', 'c4o', 'n1C', '=C/', 'C(=O)OC(C)(C)C)', '[C@@H](C)CO)', 'ccccc12)', 'CC(=O)NC(', 'cc2)ccc1', '[C-]', 'CCCN(CCC)', 'CO2', 'COc1ccc(OC)c(', 'OC(C)C)', 'c2ccc(', 'C(N)=O', 'CCCN3', 'C(F)(F)F)cc(', 'Cc1ccc(Cl)cc1)', 'OCCOCC', 'c1ccc2[nH]', 'nn3', 'C1CC1', 'c3ccc4c(c3)OCO4)', '[cH:2]', '[C@H](CC)', 'c2cc(O', 'NC', '[Pb+2]', '[Br:11]', '[C@H]12', '3)cc2)c1', 'CCOCC3)', 'CCCN2C(=O)', 'c1cccc2c1', '[nH]n1', '[c:15]', '(=O)=O)c1', 'c2c(Cl)', 'CC[C@]4', 'SC1', 'nc(-', 'c1c(Cl)cccc1', 'c3ccc(Br', 'C(=O)N(C)C)', 'c3cc(F)cc', 'OCO3', 'o2)cc1', '[cH:17]', '([O-])', 'NC(=O)N', '[cH:4]', '[C@]3', '[C@@H]3CC', '[F-]', 'c3ccccc32)', 'c1nc2c(', 'C2(C)', 'c2cccc(OC)', 'Cc1c(Cl)', '[C@@]1(C)CC', 'nn2)', 'CCCCN)', 'c1c(=O)', 'CC[C@H](NC(=O)', ')ccc2', '=C/C', 'c2cccnc2)', 'c4nn', 'F)cc2)', 'c1-c1ccccc1', 'C(Cl)(Cl)', 'n2', '/C=C/C(=O)', 'CN2C(=O)', '[n:7]', '2)', 'c3ccccc3)cc2)', 'c1nc(-', 'c2cc(OC)c(OC)', 'c1nccc(', 'c2ccccc2C)', 'Oc3ccc(', 'NC(N', 'c(SC)', '(', 'nc(', 'c1ccc(O', '[N+](=O)[O-])cc(', 'COc1ccc(C(=O)', '2)C1', '[N+](=O)[O-])c1', 'CS', 'C(=O)Nc1ccccc1', 'c3ccc(O)', 'c1c(-c2n', '/N=C\\', 'c2ccccc2c1', 'c1cc(-c2ccc(', 'c1ccc(CN2', 'O[C@@H]3', 'c2)n1', '/C(=N\\', 'nc2ccccc21', 'O)cc1', 'CC1=', 'CN1CCN(', 'c2ccc(O', 'N(S(=O)(=O)', 'C(F)(', 'c1cn', 'S(N)', '(=O)=O)', 'C(=O)OC)', '[cH:9]', 'CC2)', 'c1c-2', 'C2=O)ccc1', 'nc3', 'nn2', 'n[nH]', 'c2n(', 'Cl)cc1)', 'c2nccc(', 'c2c(-c3ccccc3)', 'C1CCCC1)', '[cH:1]', '(O)', 'c5', 'COc1ccc2[nH]', 'CN4', '[nH]c2', 'c2ccc([N+](=O)[O-])cc2)', 'c4ccc(OC)cc4)', 'c3ccc(OC)', 'COc1ccc(C(=O)N', 'nc(C(=O)N', 'CC3', '[Ru]', 'ccc(-', 'N(C)', 'c1n[nH]', 'c2ccc(F)cc2)CC1', 'CC(=O)OC', '[CH2:15]', 'c2cc(Br)', 'CC4', 'C1CN(', 'N3CCN(', '[C@@]13', 'C=C)', 'Cn2cc(', '[N-]', 'c(CO', '[Pt]', 'N#Cc1ccc(', 'O=S(=O)(N', 'C(C)(C)', 'ccc2c1', 'NC(=O)C(', 'no', 'NC(=S)', 'c3ncc(', 'c4ccc(F)cc4)', 'N2CC', '[V+4]', '[C@H](C', '[C@@H](OC)', 'cc5', '[Th+4]', '/C(=C(\\', 'CN2CCC(', 'Cn2cn', 'C12CC3', 'C[C@H]1', '5CCOCC', 'ccc(N', 'CC1=N', 'nnn1', 'C4CCCCC4)', 'Cc1cc(O)', 'CCn1', 'C2)ccc1', 'c1cccc(F)c1', 'c2ccc(O)', '[Rh+]', 'c1o', 'c43)', '[NH-]', '-c2ccc(', '[C@@H](C)O)', 'C(F)(F)F)cc1)', 'c23)CC1', '[C@H]1', '[nH]c(', 'cc', 'COC(=O)N', '[C@H]3O)', 'C2', 'c(C(', 'c(NC(=O)N', 'CCN3CCOCC3)', 'COc1cc(N', 'c1ccc(OC)cc1)', 'cs1)', 'Cc2ccc(-', 'c4cc(', 'c2ncc', '=C(C)', 'c3ccc(O', 'nc3)', 'ccn1)', 'c2Cl)', 'c1ccccc1)c1ccccc1', '[N+](=O)[O-])cc1)', 'nnc2', 'CC(CC(', 'CCC)', 'n1)', 'c2)c1', 'c1ccc(O)c(O)', 'c(Cl)c2)', 'N)N', ')cc(C)', '[cH:3]', 'c2cccc(Cl)c2)', 'N)', 'cs', 'C(=O)O)cc2)', 'c(/C=C/', 'C1CCCN(', 'c(=O)[nH]c2=O)', '(CC', 'c4ccc(Cl)cc4', 'c(CN', 'c1ccco1', 'C1', 'OCC(=O)N', 'c(=O)o', 'c(N3', 'c2ccncc2)', 'c3ccc(Cl)', 'c3ccc(N4', 'O=c1cc(', '[S+]([O-])', '1C', 'c1ccc(NC(', 'sc(-', 'c(OC', 'cs1', 'c2ccc(Br)cc2', 'Nc1ccc(', 'c3ccc4ccccc4', 'cc3)cc2)', 'c2ccc(S(=O)(=O)N3', 'CC[C@H](O)', 'c1ccc(OCC', '[nH]c1)', 'C(=O)N1', 'NC(=O)[C@H](', 'c1nnc(-', '[S:2]', ')ccn1', 'c4cc(C)', 'CCN(C', 'c4ccc5c(', '4)CC3)', 'C(F)', 'c4n', 'OC1', '[Sn]', '[C@H]2CC[C@H](', 'c2ccc3c(', 'C1N(C(=O)', 'C)C', 'O=c1[nH]', '[C@@H](CC(C)C)', '[TeH2]', '[N+](=O)', 'c2O)', 'C(C)(C)C)cc2)', '[N+]', 'CC(C(=O)O)', 'CC(O)CO', '[C@H]1CO', 'C2CCN(', 'c3ccc(NC(=O)', 'c2cc(-c3ccccc3)', 'c3ccco', '[C@@H]', 'CCO1', 'c1n(C)', '/C=C3', 'C(=N)', 'c2C)cc1', 'ccc3', 'CO[C@@H]1', 'c2ccccc2)',
    'c45)', 'c2', 'C2CCCC2)', 'C1C', '[C@H]1CC', 'Br)cc1', 'Br)cc(', '[C@@H](NC(=O)', 'CC3CC(', 'C(\\', 'CCl)', 'CC(O', '(F)(F)F)', '[C@]2(O)', '[C@@]21C', '[Co+4]', 'c3cccc(F)', 'c1ncn2', 'Cc1cc(C)', 'c2sc(', 'cc2c(c1)', 'C3(', 'c(=O)n(-', '[C@]34', '-c2cs', 'N12', 'Oc1c(', 'CCC4)', 'c1-', 'ncc', 'c3ccc(S(=O)(=O)N', '[O:8]', '[C@@H](CC', 'c2co', 'CCN(C)CC', 'c3n[nH]', 'CC5', '[P-]', 'cc(C(=O)N', 'CC1CCCO', 'N(C(C)=O)', 'c(C)n1', 'Cc3ccccc3)', 'c2ccc(N3', 'cn1)', 'CCOCC4)', 'nc4', 'C4CC4)', 'CC(CC(C4)', 'c2nc(C)', 'N(C', 'c1ccccc1-', 'c1cs', '[nH]3)', '[C@@H]4CC', '=S)', 'c4ccc5c(c4)', '(C(=O)O)', 'C', 'c5c(', 'c2C)', 'C(C(=O)N', 'CCC12', '[C@]2(C)CC', 'c2nc(', 'N=C(N)N', 'ccc(O', 'c2ccc(F)cc2', 'C2C3', 'c(C(C)C)', 'c(C)cc(C)', 'c(F)c1', '[Mn]', 'COc1ccc(NC(=O)', 'no2)', 'CCCCCCCC', 'O=C(COC(=O)', 'oc(C)', 'OCc2ccccc2)', 'CC(C)(', 'CCOc1ccc(', 'CC(C)(O)', 'C(=O)c2ccccc2', '=O)o', 'c2cccc(Cl)', '=', 'c32)cc1', 'CN', 'c4c3', '3)cc1', 'c2ccc(Cl)cc2)CC1', 'CCN1C(=O)', 'CCCCC3)', '8', 'cc2)c1', 'O=S(', 'c2cccc(C)', 'P', 'C(C)(C)O', '[C@]3(C)', '[Fe+2]', 'c1cc(OCC', 'c1ccc(O)cc1)', 'cn(C)', '=O)cc1)', 'N=C', 'Cc1cccnc1)', '(F)', 'Cc2ccccc2)c1', 'c1nc2ccccc2', '[n+]([O-])', '/N=C(\\', 'C(=O)OCC(=O)', 'CCN(C)CC1', 'c2o', 'cn2)cc1', 'c3cc(C(=O)N', 'ccc(F)cc1', 'c2cccc3c2', 'c2cco', '[C@@H](N)', 'C)cc2)', 'CCCCN1', 'C(NC', 'Cc1cccs1)', 'c4cccn', 'NC(=O)/C=C/', 'c(OC)c2)', 'c1ccccc1F)', '[Zr+4]', '=C(', 'Nc1ccccc1', '4)cc', 'N1C(=O)', 'nc(SCC(=O)N', 'C(F)(F)F)cc', '[C@]1(C)', 'c1c(O)c(', 'c1ccc2c(', 'c2cc(OC)c(OC)c(OC)c2)', 'CCCCC1)', 'c(C)c(', 'Oc2c(', 'CCCOc1ccc(', '(N3', 'c1nc(C)', 'CN(C', '3)ccc2', 'sc2', '[N+](=O)[O-])cc', 'O', '[c:10]', 'c(Cl)cc1', 'ncnc32)', '=N/', '[BH3-]', 'c3ccccc3c(=O)', 'c1ccc(C)cc1)', 'c(-c2ccccc2)', '(c2ccccc2)', 'O=C', '-c2ccc(Cl)cc2)', 'c2cccc3', 'no1)', 'Cc1ccccc1)NC(=O)', '=N)N', 'N(CC(=O)N', 'Cc1ccc2c(c1)', 'n5)', 'Cc1nc(N', 'Cc2c(', 'c1cc(N', 'nn3)', 'c2ccc(S(=O)(=O)N', 'N1CCCCC1', 'C(C#N)=C(N)', 'Cc1cc', 'C(=O)C1', 'F)cc2', 'c2cc(F)c(', 'CCC1(', '(CO)', 'O=C1', 'C(O)C(O)', 'N[C@H](', 'OC)', 'n2C)', 'C(=O)N2CCC(', 'CCC3)', 'N(C)C', 'c1ccc(Cl)cc1', 'c3cc', 'CC(=O)N[C@@H](', 'CCOCC2)cc1', 'o2', 'CCCCCCCCCCCC', 'C(C)C', 'c1ncn', 'C=C(C)', 'O[C@@H]2', 'c(-c3ccccc3)', 'C(Cl)', 'cnc(N', '-c2ccccc2', 'C2=N', 'CCc2ccccc2)', 'Cc1cc2', '-c2nc(', 'c2cccc(C(F)(F)F)', 'C(=O)Nc2ccc(', 'c2ccccc21)', 'c4ccc(C(F)(F)F)cc', 'Br)cc1)', 'CC[C@@H](O)', 'c(Br)c1', 'CCCC4)', 'no1', 'c3ccc(C)', 'NC(=O)CO', '[C@@](C)(O)', 'CCC(', '(F)F)', '[Pd+3]', 'c1c(F)ccc(', 'c2cc(OC)c(OC)c(OC)', 'C=C(', 'CC)cc2)', 'C(=O)Nc1ccc(', 'CN(C(', 'O=S(=O)(', 'Cc1cn', 'c1cccc2', 'C2=', 'c(NC', 'N=C(N', 'CCN1C', '/C(=N/O)', 'nc1N', 'C2=C(', '=C(N)N)', '[Ga+3]', 'nn(', 'CCC3(CC2)', '6)', 'c2nnc(', 'P(=O)(O)', 'N)N)', '[CH2:8]', 'c1cccnc1', '-c1ccc(', '[C@@]1(C)', '1)', '[C@@]4(C)', '[c:7]', 'CCS(=O)(=O)', '[Au+]', 'CN(', 'ccc(NC(=O)', 'c3ccc(Cl)cc3', 'O2)cc1', '[S+]', ')cc1)', 'N1CCCC1', 'FC(F)(F)', 'c4ccccn4)', 'OCC(O)', 'C(=O)C(', '[C@H](N', 'c2ccc(Cl)c(Cl)c2)', 'Cc1ccc(F)cc1', 'n2)ccc1', 'ccc(F)c1', 'c2cn(', 'CN3CCN(', 'c3nc4c(', 'c2ccccc2)cc1', 'C=N', 'c(Cl)cc', 'n5', '[Cu+]', 'O)cc2)', 'C(=O)N[C@@H](Cc1ccccc1)', 'c1ccc(O)', '[Ta+5]', 'I)', 'Cc2cc(', 'n1c(C)', 'COc1cc(', '[C@@H]1CC', 'c2n(C', 'C(=S)N', 'c(C)c1', 'CCCCCC)', '#N)', 'O[C@H](CO)', 'CCCC2)', 'CCC[C@@H]1', 'nc(N', '[13CH]', 'S1(=O)=O', 'ncn2)', '[C@H](O)[C@@H](O)', 'c2c(C)cc(C)', 'nc1S', 'N1CCN(', 'C(=O)N[C@@H](C)', 'COc1cc', 'N(C(', '[cH:7]', 'C(=O)O)', 'OCC(N', '-n2', 'NC(C)=O)', 'cc(', 'c2ccccc2Cl)', 'c(C(=O)N', '[O-]', 'Cc1nc(-', 'c1=O)', 'c1cc(Cl)ccc1', 'Cc1cc(N', 'nccc2', 'c(C)c2', 'Cn2c(', 'COc1cc2c(cc1OC)', 'c-', '-c2', '[C@H](Cc1ccccc1)', '4)cc2)', 'C(=S)S', '[C@H](CC', 'C(F)(F)F)c3)', '[CH2:11]', 'CCCC2)cc1', 'O)c(', 'S(=O)(=O)N2', 'c2c(OC)', 'nc1-', 'O=C(', 'C2CCC2)', 'c1cccc(-n2', 'c2c(C)cccc2', 'c1ccc(CN', 'o3)', 'c2c(F)cc(', 'c2ccc(Cl)cc2)', 'cccc2)', 'CC[C@]43C)', 'c3cc(OC)c(OC)', 'OC)cc1', 'c3cccc(F)c3)', 'O)c1', 'CCN1CCN(', 'c(=O)cc(', '=O)C1', 'c2c(F)', 'c(OC)cc(', '-n1', 'ccc(OC)c1', 'c3ccc(F)cc', '[nH]2)c1', 'c1ccccc1', 'Cc1nc(', 'C(C)(', 'Cc1ccc(Cl)cc1', 'Cc1ccc2c(', '[o+]', 'c(O)c(', 'n2)c1', 'nc(C)', 'N', 'N2CCN(C', 'OCCO2', 'CC2)CC1', 'Br)c2)', 'c(CC)', ')', 'c2c3ccccc3', 'CC[C@@H]1', 'n1ccc(', 'c2cc(OCC', 'c2=O)c1', 'O=C(Nc1ccc(', 'OC(C)', 'c(C(N', 'n(-c2ccc(', 'c4cccs', '[Pd+2]', 'c3c(F)', '%11', 'CO)', 'c(=O)n(C)', '[C@H](OC(C)=O)', 'c(OCC', '[SnH]', 'n2)n1', 'c2oc(-', 's', '[C@@]4(C)CC', 'OO', 'c3nc(N)', 'cnc2c1ncn2', 's2', 'C1=C', 'C2)', 'c5cc', 'CCCC3', '[Os]', '[n:6]', '[C@@]1', 'n2c(=O)', 'CO[C@H]1', 'C[C@H]1CN(', 'cc(-', '[C@@H](C)', 'c1cc(F)c(', 'C#N)cc', 'c2ccc(Cl)c(Cl)', 'C#N)', '=C(O)', 'c23)c1', 'n2ccc(', 'c3c(Cl)cccc3', 'c4ccc(Cl)cc', '/C=C', 'c2c(F)cccc2', 'N2CCN(C(', 'c2c(Cl)cccc2', 'c1c(Cl)ccc(', 'C(O)', '[n+]3', 'c3cn', 'c3ccccc3)cc2', '[C@@]3(C)', '[c:4]', 'CC(', 'c1)', 'N(CCC)', 'OC)c(OC)c(OC)', '[nH]1', 'c3ccc(-', 'O=C(CSc1n', '[Ni+6]', 'c2ccccc2)CC1)', 'C(=O)N(C)', 'NC(=O)N1', 'CC(N', 'ncc(', 'c1c(C#N)', 'c3ccc(O)cc3', 'c2cccc(NC(=O)', '[P@@]', 'C(=O)N3', 'CCOc1ccccc1', 'c4n3)', 'C4CCCC', '[c:6]', 'OC)cc(', 'c3n2', 'c3', '3)cc21', 'c2cccc(O)', 'n4', '/N', 'c3)cc', 'cc2', 'c2ccccc2c1=O', '[CH2:12]', 'c3O)', '[C@H](C)N', 'CC(N)', 'S(C)', 'c3c(Cl)cc(', 'c3c(ccc(', 'Cc1ccc(OC)', 'c3c[nH]', 'CCN3', '[C@]1(', 'N(O)', 'c(-c2ccc(', 'Cc1ccc(C)', '[C@H](NC(=O)', 'no2)cc1', 'CCCCCC', 'F)cc3', 'P(O)', 'nc23)', '[O+]', 'F)ccc1', '(F)F', 'Cc2ccccc2)', 'N2CCCC2)', 'c2ccc(Cl)cc2)c1', '(=O)O', 'cc(C)', 'c2c(c1)', '(=O)=O', 'cc5)', 'n2cn', 'cnc2)', 'C1(', '[CH3:9]', 'c3c(O)', 'CC2', 'S(C)(=O)=O', 'cccc12', 'c[nH]1', 'c3cc4ccccc4', 'O1', '=C4', 'C)C(=O)', '[C@@]21', '-c2c3c(', '[13cH]', 'c3ccc(Cl)cc3Cl)', '-c2c[nH]', '[Ge+4]', 'CCNS(=O)(=O)', 'n(-c3ccccc3)', 'c1cc2c(', 'c2)cc', '1', 'C(=O)OC', 'ccn2)', '[S@]', 'Nc1ncc(', '=O)ccc1', 'c2ccccc2', 'c2ccc(C)cc2)', 'c1ccc(NC(=O)', 'C(=O)c1ccccc1', 'N(CC)CC', 'Cn1c(=O)', 'C=C2', '[C@@H]2O)', 'Cc1cccnc1', ')cc', 'n2c1', 'C(O)=C(', '[H+]', '[Gd+3]', '2', 'CC2)cc(', 'S(=O)(=O)N1', 'N2CCOCC2)', '4)cc3', 'c3ccccc3F)', '=C1/', 'n1c(N)', 'c1cc(-', 'CSc1n', 'c2c3', 'OC3', '[Li]', '[CH:2]', 'CC[C@H]1', '[13C]', 'C(N(C)C)', '[CH-]', '[c:2]', 'CCCN(C)', 'c4)', 'C(=O)N(C', '[C@@H](O)', 'C[C@H](N)', 'CC(=O)N2', 'C#N)cc1', '-c2ccc(F)cc2)', 'F)cc1)', 'c2ncn(', 'Cl)ccc1', 'CCC', 'P(=O)', 'oc(-', 'c4c(C)', 'n1c(=O)', '[C@]23', '[Ir+]', '[Cl:19]', 'c3cccc4', 'Cl)c1', 'CCCCNC(=O)', 'CCN2CCOCC2)', 'n3)', 'cc(C)cc1', '(CC)', 'C4)', 'OCO2', 'c1c(Br)', 'c5)', 'C(F)(F)F)ccc1', 'c(C)c2)', '[Ti+]', 'on1', '[cH:11]', '[Fe]', 'CC(=O)O', '[C@]2(', 'SCC(=O)N', 'C#', 'c(-c2n', 'c3cccc(C)', 'P(=O)(O)O', '[C@]3(', 'c2ccc(C#N)cc2)', 'CC2(', ')cc(-', 'C[C@@H]4', 'n3)cc', '6', '=N)', 'ncc(-', 'C3)cc1', 'CCN(C(=O)', 'C(=O)NCC', 'CC1=O', 'CCC(CC)', 'CC(=O)N(', '(C)(O)', 'c1cc(OC)c(', 'n(Cc3ccccc3)', 'nnc(-', 'C1(O)', '[W+5]', 'c2N', 'n3cn', 'Cc1ccc(F)cc1)', '=O)cc3)', '4CCOCC4)', 'c2n1', '[Dy+3]', '-', 'c(C(=O)', '[Mn+2]', '[C@H]2[C@H](', 'c(=O)[nH]', 'n1c(-', 'c2cc1', 'c2c(ccc(', 'CCC(N', 'c3ccccc32)cc1', 'c(=O)[nH]1', 'nc3C)', 'c2ccc(C(=O)N', 'CNS(=O)(=O)', 'C1=', 'nn2)c1', 'c2ccccc12)', 'c2cccc(', 'c(=O)c2c1', 'c2ccccc2OC)', '[N+]2', 'C4(', 'c2c1=O', 'C[C@@]12', 'C[C@@H](CO)', '[C@]4', '/C=C3\\', 'C/C=C\\C/C=C\\', 'C2)CC1', '[C@@]23', 'c2ccccc2)cc1)', 'c2ccccc2)c1', 'c2ccc3c(c2)', 'CC3)cc2', '[C@H](C(=O)O)', 'c2cc(N3', 'ccc5', 'C(=O)N(C)C', 'c1cccc(O', 'c[nH]', 'CN)', 'C(=O)N2C', '[nH+]', 'c1cccs1)', 'cc4)', 'c3ccc(C#N)cc3)', 'CC1(C)', 'O2', ')cc(', 'CCN(CC)CC)', 'c(OC)cc1', '%16', 'c4cccc5', 'c1-2', 'Cl)c(Cl)c1', 'c4c(F)', 'c4ccccc4)', 'C(CN', 'SC)', 'c1cc(OC)c(OC)c(OC)', 'n4cn', '[n+]2', 'CN(C)C(=O)', '/C=C(\\C)', 'CCNC(', 'I', 'N2C', 'c1c(C)', 'c(F)', 'n1c2c(', 'CCCCCCCC)', 'C1(N', 'N1', '[nH]2', 'C(N)=O)c(N', 'Oc1cc(', 'CC1CC1', 'OC(C)(C)', 'N(CC', 'c(C)c3)', 'C3)', '=C)', 'nc(SC)', '[13CH2]', 'c21', 'CCC3(', 'c1ccc2c(c1)', 'C#N)c1', 'c2cccc(F)', 'SS', 'c4ccncc4)', 'c3c(c1)', 'CCCC(N', 'C(=O)N[C@H](', 'c1cccc(', 'c3ccc(Br)cc3)', 'c2c(C#N)', '=C/C=C/', 'Cc1nn(C)', 'c1cc2c(cc1', '[P+]', 'cnc2', 'c2cccs2)', '[Cl:10]', 'CCCN1C(=O)', '/C(=N\\O)', '[C@]3(C)CC', 'cccc2', 'NC1=N', 'c3n(C)', 'CC(=O)N', ')cc4)', 'COC)', ')cc2c(', '[C@@]4', 'CCC(C)', 'C(=O)O)cc1', '[C@@H]2[C@@H](', 'CCC(=O)N1', '[Zn+2]', 'C(OC', 'c2cccc', 'c3ccc(OCC', '[SiH2]', 'ccc(F)', '/C=N/NC(', 'c23)', 'C(F)(F)F)cc(C(F)(F)F)', 'CCc1n', 'c1cc(Cl)c(', 'Cc1cccc(C)c1', 'C[n+]1', 'C(F)F)', 'c4ccc(', '/N=C2', 'c1ccncc1)', 'OC2', '2)CC1', '[N:10]', '/C1=C\\', 'CC)c1', 'C(=O)c2c(', 'c2c(c1', 'n1-', 'c2cccnc2', 'c(C#N)', 'CCc1ccccc1)', 'c1c(-c2ccc(', 'F)cc3)', 'CC[C@H]3', 'c2ncc(', 'F)c(-', 'c3)CC2)', 'CCN', 'c1cc(O)', 'c2nc(N3', 'cc12', 'Cl)cc1', 'S(C)(=O)=O)', '[n+](C)', '3)cc2)cc1', 'c1cccc(NC(=O)', 'nc32)', 'c2=O)', '(F)(F)', '[15n]', 'c2ccc3n', 'c3ccc(C(F)(F)F)cc3)', 'NC(=S)N', 'CCCN(C(=O)', 'c1cccc2ccccc12', 'N1CCCC', ')cc3)', 'c3c(N', 'O=C(Nc1cccc(', 'C(=O)OCC', 'C/C(', 'F)c1', 'c(NC(=O)', '[nH]c(-', 'C(=O)N(', 'c2ncc(-', 'CCOC', 'nc21', 'COC(=O)c1ccc(', 'c2)C1', '=[N+]=[N-])', 'c-3', 'nc2c1', 'nnc1', 'cccc2)c1', '2)cc1', 'CC[C@]2(C)', 'Nc1ncn', 'n1cc(', '(C)C', 'c(OCC)', 'c3cc(OC)', '[C@@H](N', 'cncc1', 'c2no', 'c3cccnc3', 'Cl', 'COc1ccc(C2', '-c2s', 'n2)CC1', 'N(Cc1ccccc1)', '/C=C2\\', 'O[C@H]1', 'c(=O)c1', 'c1c(C)cccc1', 'cnc1)', 'c1c2cccc', 'C(=O)O', 'c3)cc2)', 'c4)cc3)', 'O2)', 'c2ccc(-c3n', 'C)c1', '(N', 'c1ccc2nc(', 'c3cccc(OC)', 'c2cc(OC)', '[C@H](O)[C@H](O)', 'CS(=O)(=O)', 'o2)', 'CC[C@H]2', 'CCC2', 'Cl)cc(Cl)', '[nH]2)cc1', 'C34', 'c1ccccc1Cl)', 'Cl)c(Cl)', 'O)cc1)', '(C(F)(F)F)', 'C(S', 'c1ccc(OC', 'CCCCCCC)', 'c2ccc3c(c2)OCO3)', 's1', 'C#N)cc2', 'c1cccc(F)c1)', 'c(', '[Tm+3]', 'c1ccccc12', 'c1cc(OC)', 'c2cc3c(', '/C(C#N)', 'cnn1', 'C3', '/C(O)', '[Ag+3]', '[n+]1', 'C(CO', 'cccc1', 'c2nc(O)', 'NC(=O)CS', 'c4ccc(F)cc', 'C[C@@H](C)', '5CCCC', 'CCCC3)', 'c3ccc(C)cc3)', 'C(N2', '3', 'c7', 'ccc(Cl)c1', 'C(=S)', 'CCn1c(=O)', 'c1c(-c2ccccc2)', '[C@H]2CC', 'C(=O)N', 'OC)c(OC)c3)', 'c6cccc', 'Cc1c[nH]c2ccccc12)', '[nH]2)', 'CCN)', 'c1cccs1', 'OCCCC', 'S(=O)', 'c1nc(N)',
    'cn2)', 'c1c(Cl)cc(', '[C@@H]4[C@@]5', '[C@H]2C', 'n2)cc1', 'CCCCC', 'c(N2', 'c3ccc(C(=O)O)', 'C(=O)N[C@@H](CC(C)C)', '3C)', 'c2ccc(OC)cc2)', 'c3ccc(C(=O)N', 'c1ccccn1)', 'C(CC)', 'c4ccccc4)cc3)', '[n+](C', 'CCCn1', '[n-]', 'nn2)cc1', '=C(C#N)', 'n12', 'CCN(C)', 'OC', '3)ccc21', 'c(Cl)c1', 'F)', 'Br)c3)', 'CC[C@@H]3', 'NC(=N)', 'c1cccc(F)', '[Ag]', 'CC3)cc1', 'C=CC(=O)', 'CC(=O)', 'c3cc(Cl)c(', 'c3F)', 'C(=N', 'CCN2C(=O)', 'c4ccccc34)', 'c4ccc(C)', 'CCCO)', 'Cc1ccc(O)cc1)', 'OCO2)', 'c2cc3ccccc3', 'Cc1[nH]', 'c3c(F)cccc3', 'C(=O)C(C)', 'CCC(O)', 'CCC(C)C)', 'c1nc(', '[C@H]2O)', 'nc2n(', 'c1ccc(Cl)cc1)', 'c2)cc1)', 'C(N3', 'c2ccc(Cl)', 'NC(=O)C2', 'O[C@H](CO)[C@@H](O)', 'c2cc(S(=O)(=O)N', 'c4ccc(O)', '=N/N', 'C4)cc3)', 'nn', 'CC(CC(C3)', 'c1=O', 'c1ccc(OC)cc1', 'c1ccc(C)cc1', 'c1ccc(-c2ccc(', '[N@+]', 'c2sccc2', 'c2=O', 'Br', '\\', 'c(F)c3)', 'c3cco', 'c1ccc(N2', 'c3c(C)', 'CC[C@@]4', 'c1cn(', 'c2cccc(-', 'OCC(=O)O)', 'c3[nH]', 'c(=O)c(C(=O)O)', 'B', 'nnn2', 'CCOCC2)', 'c3c4ccccc4', 'CN(C)S(=O)(=O)', 'C(O', '[U+]', '[C@@H]1O)', 'n3)cc2)', 'c(=O)[nH]c(=O)', 'C(OC(=O)', '[Au+3]', 'O=[N+]([O-])', 'CCCN1CCN(', 'nc(N)', '[Ce+3]', 'C)cc2', '(C)CC', 'c2cc(C(F)(F)F)cc', 'c3ccn', 'OC[C@@H]1', 'c2ccc(Cl)cc', 'S(=O)(=O)c1ccccc1', 'OCO4)', 'C(=O)/C(=C/', '4)', '=O)(', ')cc1', '[13c]', '/C(C)=C/', 'c(F)c2)', 'c1ncc(-', 'c1ccn', 'ccc1O', '[C@](O)(', 'oc2c1', 'C3CCCCC3)', 'Cc1ccc(S(=O)(=O)N', 'CCCN2', 'N2)', '[Ru+3]', 'C2CCN(C(=O)', 'OCO3)', '(C)', 'CCCS', '4)cc2', 'n1', 'CCC#N)', '[C@]3(O)', 'c1c(', 'P(=O)(OCC)', 'c2cc(C)', '[C@@H](C(=O)O)', 'O=C(N1', '[C@H]3', 'nc(OC)', 'n(C)c(=O)', 'Cc1ccc(S(=O)(=O)', 'CC3)cc2)', 'c(=O)', 'CC1CCCCC1)', 'C1=O', 'OC(', 'c1O', 'c1c(N)', 'O=C1NC(=O)', '4', 'c3cc(O)ccc3', 'nc(N3', 'CC(=O)Nc1ccc(', 'c1cc(C)', 'c2c(N', 'C(OCC)', 'c2ccccc2F)', 'C(=O)N2CC', 'CC(=O)OC)', '=O)cccc1', 'CCC[C@H](', 'C(c1ccc(', 'N(CC)', 'n', 'c3ccc(Cl)cc', 'c2ccccc2n1', '=O)c1', 'c2ccc(C)', 'nc2)', '[C:10]', 'c1ccco1)', 'CN(C)C)', 'C(OC)', 'NO', 'c(=O)n(C', '3)c1', '[C@@H]2', 'ncc1', 'c2c(C)', 'c2cccc3cccc', 'COc1ccc(N', 'nc(N3CCOCC3)', 'CC3CC3)', 'C(=O)NC', 'Cn1c(', 'c(OC)', '%13', 'c2cs', 'N=C2', 'ccc2', '=S', '[C@@H]2C', '[Hf+4]', 'oc12', 'C1=CC(=O)', 'C2(C)C)', ')cc3', '[NH2+]', 'O[C@@H](', 'COc1c(', '=N1', 'c1cc', 'N1CCN(C(=O)', '-c1ccccc1', 'c4ncn', 'CCCCCC3)', 'CN(S(=O)(=O)', 'n(C', 'C1CCCC1', 'Cc4ccccc4)', 'COc1cc2c(cc1', '*', 'c1c(F)cc(', 'c3c(F)cccc3F)', '(O)(', 'c(-c3ccc(', 'c5ccc(', 'OCCN', 'CC1(C)C', 'c3ncc', 'c(=O)n(C)c(=O)', '=C1\\', 'CC1=C(', 'CN1', 'C23CC4CC(CC(C4)', 'c1C', 'c4cccc', 'cc3)', '[o:9]', '[C@H]1CC[C@H](', 'c1c(O)', 'C[C@H](C)', 'S(', 'c(SCC(=O)N', '[N+](C)(', 'c2)CC1', 'c4ccccc43)', 'C(=O)NS(=O)(=O)', 'ncnc(N', 'n(CC)', 'C(NCC', 'CCCCN', 'c4cc(F)cc', 'c2nnc(-', 'c1c2c(cc(', 'c2cccc(N', 'c2cc(F)cc', 'Cc1c(', 'NC(=O)[C@@H](', 'n2c3c(', 'c1ccccc1O', 'CC(C)(C)', '[n:8]', 'CN(C)c1ccc(', 'C#N)cc3)', 'n2cc', 'Cc3ccccc3', '[Ga]', 'c2ccc(Cl)cc2', 'CN2', 'c2nc(N)', 'C3CCN(', 'c1[nH]', 'c1c(C(=O)N', 'c1cc(O', 'cc2)', 'c2[nH]', 'c2F)', 'c1cc(F)ccc1', 'c3nc(C)', 'c2nc(-c3ccccc3)', 'C(=O)[C@H](', 'CC(c1ccccc1)', 'O=C(CN1', 'Cc3c(', '[C@H](O)[C@@H](O)[C@H](O)', '[CH2:10]', 'Cc2ccc(', 'c3c(N)', 'O=C1c2ccccc2', 'cccc3', 'OC(CO)', 'c2cc(NC(=O)', 'CC(C)N', 'CCCC(', '/C=N/N', 'N#Cc1c(', 'Oc2ccc(', 'c2c3c(', ')cc12', '-c1cc', 'nn(-', 'c(OC)c(OC)', '[C@@]3(O)', 'c(=S)', '[C@@]2(C)CC', 'c(SC', 'CCC(=O)O', 'CNCC', 'CCN(C)C', '[N+](', 'C[C@H](O)', 'Cl)c3)', 'c4cccc(F)', '=[N-]', '=C(N)', 'S(=O)(=O)c2ccc(', 'ccc(Cl)', 'c2c(Cl)cc(', 'C(c1ccccc1)', '[Pt+2]', '/c(', 'N(', 'c2cccc(OC)c2)', 'C(O)(', '[cH:8]', '[n+]', 'c1ccc(-n2', 'c2c(n1)', '#N', '[B-2]', '3)cc2)', '[C@H](CO', 'c3ccccc3OC)', 'c1ccc(-', 'F)c(', 'C2)C1', '=O)c(', 'N(CCCl)', 'c1cc2', 'c2ccccc2)C1', 'c4ccc(Cl)cc4)', 'SC(=S)', '[C@@H](', 'CCN(C)CC3)', 'N(C(C)C)', 'nc2', 'c1ccc(Cl)c(Cl)c1', 'nc2)c1', 'Cc1ccc(S(=O)(=O)N2', 'C2=O)cc1', 'ccc(O)', 'CCc1ccc(', 'n(Cc2ccccc2)', 'CS)', 'C(N)=N', 'NC(=N)N', '3)ccc1', 'S(N)(=O)=O)', 'c1ccc(F)cc1)', '[CH3:1]', '%14', 'c5cccc', 'C#N)cc(', 'CC(C)(C)O', 'CC[C@@H](C)', 'cccc1)', '[C@H](OC(=O)', '[TeH3]', 'CN=C(', 'c4ccccc4', '[n+](', 'C(=O)/C=C/', 'c2ncn', '4CCCCC4)', 'n2)cc1)', 'C(CC', '[C@@]12', 'C[C@@H](N)', 'N(Cc2ccccc2)', '_', '[UNK]', '[MASK]']

tokenizer1 = CustomBart_Atomic_Tokenizer(vocab=atomic_vocab)
tokenizer1.pad_token = '_'
tokenizer1.pad_token_id = tokenizer1.convert_tokens_to_ids(tokenizer1.pad_token)

tokenizer2 = CustomBart_FG_Tokenizer(vocab=fg_vocab)
tokenizer2.pad_token = '_'
tokenizer2.pad_token_id = tokenizer2.convert_tokens_to_ids(tokenizer2.pad_token)

max_length = 250
batch_size = 64





