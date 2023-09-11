from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold

import torch 
from torch.utils.data import Dataset

from torch_geometric.data import Data
from torch_geometric.data.dataset import IndexType
from torch_geometric.loader import DataLoader

import pytorch_lightning as pl

from utils.chem import Chemical_feature_generator
from rdkit.Chem import PandasTools

# ['AlogP', 'Molecular_Weight', 'Num_H_Acceptors', 'Num_H_Donors',
#                  'Num_RotatableBonds', 'LogD', 'Molecular_PolarSurfaceArea']
                 
feature_label = [ 'MolWt', 'HeavyAtomMolWt',
                    'NumValenceElectrons', 'FractionCSP3', 'HeavyAtomCount', 'NHOHCount',
                    'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
                    'NumAliphaticRings', 'NumAromaticCarbocycles',
                    'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors',
                    'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds', 'RingCount',
                    'MolMR', 'CalcNumBridgeheadAtom', 'ExactMolWt', 
                    'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles',
                    'NumSaturatedRings', 'MolLogP', 'CalcNumAmideBonds',
                    'CalcNumSpiroAtoms',  
                    'num_ammonium_groups',  'num_alkoxy_groups'] # 29 

# NumRadicalElectrons, num_carboxyl_groups, num_amion_groups, num_sulfonic_acid_groups --> train zero

given_features = ['AlogP','Molecular_Weight','Num_H_Acceptors','Num_H_Donors','Num_RotatableBonds','LogD','Molecular_PolarSurfaceArea'] # 7 

generator = Chemical_feature_generator()

class Chemcial_dataset(Dataset):
    def __init__(self, 
                 data_frame:pd.DataFrame,
                 fps,
                 mol_f,
                 transform=None,
                 is_train=True):
        super().__init__()
        self.df = data_frame
        self.fps = fps
        self.mol_f = mol_f
        self.transform = transform
        
        self.is_train = is_train
            
        
    def __getitem__(self, idx: IndexType | int ):
        return self.get_chem_prop(idx)
        
    def __len__(self) -> int:
        return self.df.shape[0]
    
    def get_chem_prop(self, idx):
        
        sample = self.df.iloc[idx]
        fingerprint = self.fps[idx]
        molecular_feature = self.mol_f[idx]
        smiles = sample['SMILES']
        
        edge_index, edge_attr = generator.get_adj_matrix(smiles=smiles)
        atomic_feature = generator.generate_mol_atomic_features(smiles=smiles)
        input_ids = generator.encoder_smiles(smiles) # 384
        # ChemBERTa = ChemBERTa.detach()
        # molecular_feature = sample[feature_label] # if we use VarianceThreshold, then block this code

        if self.is_train:
            MLM = sample['MLM']
            HLM = sample['HLM']
        else:
            MLM = -99.
            HLM = -99.
            
        atomic_feature = torch.tensor(atomic_feature, dtype=torch.float)
        molecular_feature = torch.tensor(molecular_feature, dtype=torch.float).view(1, -1)
        fingerprint = torch.tensor(fingerprint, dtype=torch.float).view(1, -1)
        MLM = torch.tensor(MLM, dtype=torch.float).view(1, -1)
        HLM = torch.tensor(HLM, dtype=torch.float).view(1, -1)
        y = torch.concat([MLM, HLM], dim=1)
        
        return Data(x=atomic_feature, mol_f=molecular_feature, fp=fingerprint,
                    edge_index=edge_index, edge_attr=edge_attr, input_ids=input_ids,
                    y=y, MLM = MLM, HLM=HLM)
            
        
        
        
        
class KFold_pl_DataModule(pl.LightningDataModule):
    def __init__(self,
                 train_df: str = '/root/Competitions/DACON/4. JUMP AI 2023/data/new_train.csv',
                 k_idx: int =1, # fold index
                 num_split: int = 5, # fold number, if k=1 then return the whole data
                 split_seed: int = 41,
                 batch_size: int = 1, 
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
            df = pd.read_csv(self.hparams.train_df, index_col=0)
            
            mask = df['AlogP'] != df['AlogP']
            df.loc[mask, 'AlogP'] = df.loc[mask, 'MolLogP']
            
            # if we use rdkit fingerprint generators 
            # PandasTools.AddMoleculeColumnToFrame(df,'SMILES','Molecule')
            # df["FPs"] = df.Molecule.apply(generator.get_molecule_fingerprints)
            # train_fps = np.stack(df["FPs"])
            mol2vec = []
            
            for smiles in df.SMILES:
                vec = generator.get_mol_feature_from_deepchem(smiles=smiles)
                mol2vec.append(vec)
                
            mol2vec = np.concatenate(mol2vec, axis=0)

            scaler = preprocessing.StandardScaler()
            # craft_mol_f = df[feature_label].to_numpy()
            rdkit_df = pd.read_csv('/root/Competitions/DACON/4. JUMP AI 2023/data/rdkit_train.csv').iloc[:, 1:].to_numpy()
            # craft_mol_f = np.concatenate([craft_mol_f, rdkit_df], axis=1)
            craft_mol_f = scaler.fit_transform(rdkit_df)
            # print(df.columns)
            
            # df[feature_label] = craft_mol_f
            
            # print(df.columns)
            # feature_selector = VarianceThreshold(threshold=0.05)
            
            # mol_f = feature_selector.fit_transform(df[feature_label])
            # fps = feature_selector.fit_transform(train_fps)
            

            kf = KFold(n_splits=self.hparams.num_split,
                       shuffle=True,
                       random_state=self.hparams.split_seed)
            all_splits = [k for k in kf.split(df)]
            train_idx, val_idx = all_splits[self.hparams.k_idx]
            train_idx, val_idx = train_idx.tolist(), val_idx.tolist()

            train_df = df.iloc[train_idx]
            train_fp = mol2vec[train_idx]
            train_mol_f = craft_mol_f[train_idx]
            
            val_df = df.iloc[val_idx]
            val_fp = mol2vec[val_idx]
            val_mol_f = craft_mol_f[val_idx]
            
            self.train_data = Chemcial_dataset(data_frame=train_df, fps=train_fp, mol_f=train_mol_f, transform=None, is_train=True)
            self.val_data = Chemcial_dataset(data_frame=val_df, fps=val_fp, mol_f=val_mol_f, transform=None, is_train=True)

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
        # DataBatch(x=[29, 34], edge_index=[2, 62], edge_attr=[62], mol_f=[1, 36], fp=[5235], MLM=[1], HLM=[1], batch=[29], ptr=[2])
        print(batch)
        
    