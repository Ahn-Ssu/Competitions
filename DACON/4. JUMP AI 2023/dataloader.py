from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import KFold

import torch 
from torch.utils.data import Dataset

from torch_geometric.data import Data
from torch_geometric.data.dataset import IndexType
from torch_geometric.loader import DataLoader

import pytorch_lightning as pl

from utils.chem import Chemical_feature_generator

feature_label = ['AlogP', 'Molecular_Weight',
                    'Num_H_Acceptors', 'Num_H_Donors', 'Num_RotatableBonds', 'LogD',
                    'Molecular_PolarSurfaceArea', 'MolWt', 'HeavyAtomMolWt',
                    'NumValenceElectrons', 'FractionCSP3', 'HeavyAtomCount', 'NHOHCount',
                    'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
                    'NumAliphaticRings', 'NumAromaticCarbocycles',
                    'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors',
                    'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds', 'RingCount',
                    'MolMR', 'CalcNumBridgeheadAtom', 'ExactMolWt', 'NumRadicalElectrons',
                    'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles',
                    'NumSaturatedRings', 'MolLogP', 'CalcNumAmideBonds',
                    'CalcNumSpiroAtoms', 'num_carboxyl_groups', 'num_amion_groups',
                    'num_ammonium_groups', 'num_sulfonic_acid_groups', 'num_alkoxy_groups']


class Chemcial_dataset(Dataset):
    def __init__(self, 
                 data_frame:pd.DataFrame,
                 transform=None,
                 is_train=True):
        super().__init__()
        self.df = data_frame
        self.transform = transform
        self.generator = Chemical_feature_generator()
        
        self.is_train = is_train
            
        
    def __getitem__(self, idx: IndexType | int ):
        return self.get_chem_prop(idx)
        
    def __len__(self) -> int:
        return self.df.shape[0]
    
    def get_chem_prop(self, idx):
        
        sample = self.df.iloc[idx]
        smiles = sample['SMILES']
        
        edge_index, edge_attr = self.generator.get_adj_matrix(smiles=smiles)
        atomic_feature = self.generator.generate_mol_atomic_features(smiles=smiles)
        molecular_feature = sample[feature_label]

        if self.is_train:
            MLM = sample['MLM']
            HLM = sample['HLM']
        else:
            MLM = None
            HLM = None
            
        atomic_feature = torch.tensor(atomic_feature, dtype=torch.long)
        molecular_feature = torch.tensor(molecular_feature, dtype=torch.long)
        MLM = torch.tensor(MLM, dtype=torch.long)
        HLM = torch.tensor(HLM, dtype=torch.long)
        
        return Data(x=atomic_feature, mol_f=molecular_feature,
                    edge_index=edge_index, edge_attr=edge_attr,
                    MLM = MLM, HLM=HLM)
            
        
        
        
        
class KFold_pl_DataModule(pl.LightningDataModule):
    def __init__(self,
                 train_df: str = '/root/Competitions/DACON/4. JUMP AI 2023/data/new_train.csv',
                 k_idx: int =1, # fold index
                 num_split: int = 5, # fold number, if k=1 then return the whole data
                 split_seed: int = 41,
                 batch_size: int = 2, 
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 persistent_workers: bool=True,
                 train_transform=None,
                 val_transform =None
                 ) -> None:
        super().__init__()
        persistent_workers = True if num_workers > 0 else False
        self.save_hyperparameters(logger=False)

        self.train_data = None
        self.val_data = None
        self.num_cls = 0

        self.setup()

    def setup(self, stage=None) -> None:
        if not self.train_data and not self.val_data:
            train_df = pd.read_csv(self.hparams.train_df, index_col=0)
            
            mask = train_df['AlogP'] != train_df['AlogP']
            train_df.loc[mask, 'AlogP'] = train_df.loc[mask, 'MolLogP']

            scaler = preprocessing.StandardScaler()
            features = train_df[feature_label].to_numpy()
            features = scaler.fit_transform(features)
            train_df[feature_label] = features
            


            kf = KFold(n_splits=self.hparams.num_split,
                       shuffle=True,
                       random_state=self.hparams.split_seed)
            all_splits = [k for k in kf.split(train_df)]
            train_idx, val_idx = all_splits[self.hparams.k_idx]
            train_idx, val_idx = train_idx.tolist(), val_idx.tolist()

            train = train_df.iloc[train_idx]
            val = train_df.iloc[val_idx]
            
            self.train_data = Chemcial_dataset(data_frame=train, transform=None, is_train=True)
            self.val_data = Chemcial_dataset(data_frame=val, transform=None, is_train=True)

    def train_dataloader(self):
        return DataLoader(self.train_data,
                          batch_size=self.hparams.batch_size,
                          shuffle=True,
                          num_workers=self.hparams.num_workers,
                          persistent_workers=self.hparams.persistent_workers,
                          pin_memory=self.hparams.pin_memory,
                          drop_last=True)
                          
    def val_dataloader(self):
        return DataLoader(self.val_data,
                          batch_size=self.hparams.batch_size,
                          shuffle=False,
                          num_workers=self.hparams.num_workers,
                          persistent_workers=self.hparams.persistent_workers,
                          pin_memory=self.hparams.pin_memory)
        

if __name__ == '__main__':
    data = KFold_pl_DataModule()
    
    train_lodaer = data.train_dataloader()
    
    for batch in train_lodaer:
        print(batch)
        break
    