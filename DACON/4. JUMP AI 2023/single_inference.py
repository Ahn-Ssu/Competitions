import numpy as np
import pandas as pd

import yaml
from easydict import EasyDict
from sklearn import preprocessing
from tqdm import tqdm

from sklearn.feature_selection import VarianceThreshold
from rdkit.Chem import PandasTools

import dataloader
from torch_geometric.loader import DataLoader

from model import molp, ChemBERTa2
from reg_pl import Regression_network

generator = dataloader.Chemical_feature_generator()

df = pd.read_csv('/root/Competitions/DACON/4. JUMP AI 2023/data/new_test.csv', index_col=0)
train_df = pd.read_csv('/root/Competitions/DACON/4. JUMP AI 2023/data/new_train.csv', index_col=0)
mask = df['AlogP'] != df['AlogP']
df.loc[mask, 'AlogP'] = df.loc[mask, 'MolLogP']

mask = train_df['AlogP'] != train_df['AlogP']
train_df.loc[mask, 'AlogP'] = train_df.loc[mask, 'MolLogP']

PandasTools.AddMoleculeColumnToFrame(df,'SMILES','Molecule')
PandasTools.AddMoleculeColumnToFrame(train_df,'SMILES','Molecule')
df["FPs"] = df.Molecule.apply(generator.get_molecule_fingerprints)
fps = np.stack(df["FPs"])
train_df["FPs"] = train_df.Molecule.apply(generator.get_molecule_fingerprints)
train_fps = np.stack(train_df["FPs"])

scaler = preprocessing.StandardScaler()
features = train_df[dataloader.feature_label].to_numpy()
features = scaler.fit_transform(features)
train_df[dataloader.feature_label] = features

features = df[dataloader.feature_label].to_numpy()
features = scaler.transform(features)
df[dataloader.feature_label] = features

feature_selector = VarianceThreshold(threshold=0.05)

feature_selector = feature_selector.fit(train_df[dataloader.feature_label])
mol_f = feature_selector.transform(df[dataloader.feature_label])
feature_selector = feature_selector.fit(train_fps)
fps = feature_selector.transform(fps)


test_dataset = dataloader.Chemcial_dataset(data_frame=df, fps=fps, mol_f=mol_f, transform=None, is_train=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)


hparams_path = '/root/Competitions/DACON/4. JUMP AI 2023/lightning_logs/2.ChemBERT/2023-09-06/ChemBERT only/0/hparams.yaml'
with open(hparams_path) as f:
    config = yaml.load(f, Loader=yaml.Loader)
args = EasyDict(config).args

model = ChemBERTa2.ChemBERT(out_dim=2)

ckpt = '/root/Competitions/DACON/4. JUMP AI 2023/lightning_logs/2.ChemBERT/2023-09-06/ChemBERT only/0/checkpoints/ChemBERT-epoch=015-train_loss=20.4771-val_loss=27.0577.ckpt'
submit = pd.read_csv('/root/Competitions/DACON/4. JUMP AI 2023/data/sample_submission.csv')

    
pl_runner = Regression_network.load_from_checkpoint(ckpt, network=model, args=args)

DEVICE = 'cuda:1'
pl_runner.to(DEVICE)
pl_runner.eval()

HLM = []
MLM = []
for x in tqdm(test_loader):
    x = x.to(DEVICE)
    output = pl_runner.model(x) * 100.
    MLM.extend(output[..., 0].detach().cpu().numpy().flatten().tolist())
    HLM.extend(output[..., 1].detach().cpu().numpy().flatten().tolist())
    del x

submit['MLM'] = MLM
submit['HLM'] = HLM

output_path = '/root/Competitions/DACON/4. JUMP AI 2023/out'
submit.to_csv(f'{output_path}/ChemBERTa.csv', index=False)