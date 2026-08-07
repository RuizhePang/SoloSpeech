import os
import pandas as pd
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset
from asteroid.data import LibriMix
import random
import torch
import numpy as np

def read_enrollment_csv(csv_path):
    data = defaultdict(dict)
    with open(csv_path, 'r') as f:
        f.readline() # csv header

        for line in f:
            mix_id, utt_id, *aux = line.strip().split(',')
            aux_it = iter(aux)
            aux = [(auxpath,int(float(length))) for auxpath, length in zip(aux_it, aux_it)]
            data[mix_id][utt_id] = aux
    return data

class TSEDataset(Dataset):
    def __init__(
            self, 
            csv_dir, 
            base_dir, 
            vae_dir, 
            task="sep_noisy", 
            sample_rate=16000, 
            vae_rate=50,
            n_src=2, 
            min_length=3, 
            debug=False,
            training=False,
        ):
        self.base_dataset = LibriMix(csv_dir, task, sample_rate, n_src, None)
        self.data_aux = read_enrollment_csv(Path(csv_dir) / 'mixture2enrollment.csv')
        self.seg_len = self.base_dataset.seg_len
        self.data_aux_list = [(m,u) for m in self.data_aux 
                                    for u in self.data_aux[m]]
        self.debug = debug
        self.sample_rate = sample_rate
        self.base_dir = base_dir
        self.vae_dir = vae_dir
        self.vae_rate = vae_rate
        self.min_length = int(min_length*vae_rate)
        self.training = training
        
    def __len__(self):
        return len(self.data_aux_list) if not self.debug else len(self.data_aux_list) // 400


    def __getitem__(self, idx):
        mix_id, utt_id = self.data_aux_list[idx]
        row = self.base_dataset.df[self.base_dataset.df['mixture_ID'] == mix_id].squeeze()
        mixture_path = row['mixture_path']
        tgt_spk_idx = mix_id.split('_').index(utt_id)

        # read mixture
        mixture = torch.load(mixture_path.replace(self.base_dir, self.vae_dir).replace('.wav','.pt'))
        source_path = row[f'source_{tgt_spk_idx+1}_path']
        source = torch.load(source_path.replace(self.base_dir, self.vae_dir).replace('.wav','.pt'))
        exclude_path = source_path.replace('/s1/', '/s2/') if '/s1/' in source_path else source_path.replace('/s2/', '/s1/')
        exclude = torch.load(exclude_path.replace(self.base_dir, self.vae_dir).replace('.wav','.pt'))
        assert mixture.shape == source.shape, mixture.shape
        assert source.shape == exclude.shape, exclude.shape
        mixture = mixture.transpose(1,0)
        source = source.transpose(1,0)
        exclude = exclude.transpose(1,0)
        
        # read enrollment
        reference_path, _ = random.choice(self.data_aux[mix_id][utt_id])
        reference = torch.load(reference_path.replace(self.base_dir, self.vae_dir).replace('.wav','.pt'))
        reference = reference.transpose(1,0)

        if self.training:
            if mixture.shape[0] > self.min_length:
                new_length = random.randint(self.min_length, mixture.shape[0])
                start = random.randint(0, mixture.shape[0]-new_length)
                mixture = mixture[start:start+new_length]
                source = source[start:start+new_length]
                exclude = exclude[start:start+new_length]
            if reference.shape[0] > self.min_length:
                new_length = random.randint(self.min_length, reference.shape[0])
                start = random.randint(0, reference.shape[0]-new_length)
                reference = reference[start:start+new_length]

            
        return {
            'mixture_vae': mixture,
            'source_vae': source,
            'reference_vae': reference,
            'exclude_vae': exclude,
            'length': mixture.shape[0],
            'reference_length': reference.shape[0],
            'id': mix_id,
            'mixture_path': mixture_path,
            'source_path': source_path,
            'reference_path': reference_path,
            'exclude_path': exclude_path
        }
    
    
    
    def collate(self, batch):
        out = {key:[] for key in batch[0]}
        for item in batch:
            for key, val in item.items():
                out[key].append(val)
                
        out["length"] = torch.LongTensor(out["length"])
        out['mixture_vae'] = torch.nn.utils.rnn.pad_sequence(out['mixture_vae'], batch_first=True, padding_value=0.0)
        out['source_vae'] = torch.nn.utils.rnn.pad_sequence(out['source_vae'], batch_first=True, padding_value=0.0)
        out['exclude_vae'] = torch.nn.utils.rnn.pad_sequence(out['exclude_vae'], batch_first=True, padding_value=0.0)
        out["reference_length"] = torch.LongTensor(out["reference_length"])
        out['reference_vae'] = torch.nn.utils.rnn.pad_sequence(out['reference_vae'], batch_first=True, padding_value=0.0)
        return out

    def get_infos(self):
        return self.base_dataset.get_infos()

class TSRDataset(Dataset):
    def __init__(
            self, 
            csv_dir, 
            base_dir, 
            vae_dir, 
            task="sep_noisy", 
            sample_rate=16000, 
            vae_rate=50,
            n_src=2, 
            min_length=3, 
            debug=False,
            training=False,
        ):
        self.base_dataset = LibriMix(csv_dir, task, sample_rate, n_src, None)
        self.data_aux = read_enrollment_csv(Path(csv_dir) / 'mixture2enrollment.csv')
        self.seg_len = self.base_dataset.seg_len
        self.data_aux_list = [(m,u) for m in self.data_aux 
                                    for u in self.data_aux[m]]
        self.debug = debug
        self.sample_rate = sample_rate
        self.base_dir = base_dir
        self.vae_dir = vae_dir
        self.vae_rate = vae_rate
        self.min_length = int(min_length*vae_rate)
        self.training = training
        
    def __len__(self):
        return len(self.data_aux_list) if not self.debug else len(self.data_aux_list) // 400


    def __getitem__(self, idx):
        mix_id, utt_id = self.data_aux_list[idx]
        row = self.base_dataset.df[self.base_dataset.df['mixture_ID'] == mix_id].squeeze()
        mixture_path = row['mixture_path']
        tgt_spk_idx = mix_id.split('_').index(utt_id)

        # read mixture
        mixture = torch.load(mixture_path.replace(self.base_dir, self.vae_dir).replace('.wav','.pt'))
        source_path = row[f'source_{tgt_spk_idx+1}_path']
        source = torch.load(source_path.replace(self.base_dir, self.vae_dir).replace('.wav','.pt'))
        exclude_path = source_path.replace('/s1/', '/s2/') if '/s1/' in source_path else source_path.replace('/s2/', '/s1/')
        exclude = torch.load(exclude_path.replace(self.base_dir, self.vae_dir).replace('.wav','.pt'))
        assert mixture.shape == source.shape, mixture.shape
        assert source.shape == exclude.shape, exclude.shape
        mixture = mixture.transpose(1,0)
        source = source.transpose(1,0)
        exclude = exclude.transpose(1,0)

        if self.training:
            if mixture.shape[0] > self.min_length:
                new_length = random.randint(self.min_length, mixture.shape[0])
                start = random.randint(0, mixture.shape[0]-new_length)
                mixture = mixture[start:start+new_length]
                source = source[start:start+new_length]
                exclude = exclude[start:start+new_length]

            
        return {
            'mixture_vae': mixture,
            'source_vae': source,
            'exclude_vae': exclude,
            'length': mixture.shape[0],
            'id': mix_id,
            'mixture_path': mixture_path,
            'source_path': source_path,
            'exclude_path': exclude_path
        }
    
    
    
    def collate(self, batch):
        out = {key:[] for key in batch[0]}
        for item in batch:
            for key, val in item.items():
                out[key].append(val)
                
        out["length"] = torch.LongTensor(out["length"])
        out['mixture_vae'] = torch.nn.utils.rnn.pad_sequence(out['mixture_vae'], batch_first=True, padding_value=0.0)
        out['source_vae'] = torch.nn.utils.rnn.pad_sequence(out['source_vae'], batch_first=True, padding_value=0.0)
        out['exclude_vae'] = torch.nn.utils.rnn.pad_sequence(out['exclude_vae'], batch_first=True, padding_value=0.0)
        return out

    def get_infos(self):
        return self.base_dataset.get_infos()

if __name__ == "__main__":
    pass