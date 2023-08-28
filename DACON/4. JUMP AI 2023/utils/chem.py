from mendeleev.fetch import fetch_table
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors, MolFromSmiles

from torch import Tensor
from torch_geometric.utils.sparse import dense_to_sparse

import numpy as np


class Chemical_feature_generator():
    def __init__(self) -> None:
        mendeleev_atomic_f = ['atomic_radius', 'atomic_radius_rahm', 'atomic_volume', 'atomic_weight', 'c6', 'c6_gb', 
                        'covalent_radius_cordero', 'covalent_radius_pyykko', 'covalent_radius_pyykko_double', 'covalent_radius_pyykko_triple', 
                        'density', 'dipole_polarizability', 'dipole_polarizability_unc', 'electron_affinity', 'en_allen', 'en_ghosh', 'en_pauling', 
                        'heat_of_formation', 'is_radioactive', 'molar_heat_capacity', 'specific_heat_capacity', 'vdw_radius']
                        # Others are fine
                        # Heat of Formation: This reflects the energy associated with the formation of a molecule and might indirectly impact metabolic reactions.
                        # Is Radioactive: This binary property may not be directly relevant to metabolic stability.
                        # Molar Heat Capacity, Specific Heat Capacity: These properties relate to heat transfer but might not be directly tied to metabolic stability.
        self.mendeleev_atomic_f_table = fetch_table('elements')[mendeleev_atomic_f]
        
    

    def get_atomic_features(self,atom):

        atomic_num = atom.GetAtomicNum() - 1 # -1 is offset
        mendel_atom_f = self.mendeleev_atomic_f_table.loc[atomic_num]
        mendel_atom_f.is_radioactive = mendel_atom_f.is_radioactive.astype(int)
        mendel_atom_f = mendel_atom_f.to_numpy().astype(np.float32)

        rdkit_atom_f = [atom.GetDegree(),
                atom.GetTotalDegree(),
                atom.GetFormalCharge(),
                atom.GetIsAromatic()*1.,
                atom.GetNumImplicitHs(),
                atom.GetNumExplicitHs(),
                atom.GetTotalNumHs(),
                atom.GetNumRadicalElectrons(),
                atom.GetImplicitValence(),
                atom.GetExplicitValence(),
                atom.GetTotalValence(),
                atom.IsInRing()*1.]
        
        return mendel_atom_f, rdkit_atom_f
    
    def get_molecular_features(self, mol):
        ## 1. Molecular Descriptors 5
        MolWt = Descriptors.MolWt(mol)
        HeavyAtomMolWt = Descriptors.HeavyAtomMolWt(mol)
        NumValenceElectrons = Descriptors.NumValenceElectrons(mol)
        MolMR = Crippen.MolMR(mol)
        MolLogP = Crippen.MolLogP(mol)
 
        ## 2. Lipinski's Rule of Five 16
        FractionCSP3 = Lipinski.FractionCSP3(mol)
        HeavyAtomCount = Lipinski.HeavyAtomCount(mol)
        NHOHCount = Lipinski.NHOHCount(mol)
        NOCount = Lipinski.NOCount(mol)
        NumAliphaticCarbocycles = Lipinski.NumAliphaticCarbocycles(mol)
        NumAliphaticHeterocycles = Lipinski.NumAliphaticHeterocycles(mol)
        NumAliphaticRings = Lipinski.NumAliphaticRings(mol)
        NumAromaticCarbocycles = Lipinski.NumAromaticCarbocycles(mol)
        NumAromaticHeterocycles = Lipinski.NumAromaticHeterocycles(mol)
        NumAromaticRings = Lipinski.NumAromaticRings(mol)
        NumHAcceptors = Lipinski.NumHAcceptors(mol)
        NumHDonors = Lipinski.NumHDonors(mol)
        NumHeteroatoms = Lipinski.NumHeteroatoms(mol)
        NumRotatableBonds = Lipinski.NumRotatableBonds(mol)
        RingCount = Lipinski.RingCount(mol)
        CalcNumBridgeheadAtom = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)

        ## 3. Additional Features 11
        ExactMolWt = Descriptors.ExactMolWt(mol)
        NumRadicalElectrons = Descriptors.NumRadicalElectrons(mol)
        # MaxPartialCharge = Descriptors.MaxPartialCharge(mol) 
        # MinPartialCharge = Descriptors.MinPartialCharge(mol) 
        # MaxAbsPartialCharge = Descriptors.MaxAbsPartialCharge(mol) 
        # MinAbsPartialCharge = Descriptors.MinAbsPartialCharge(mol)  
        NumSaturatedCarbocycles = Lipinski.NumSaturatedCarbocycles(mol)
        NumSaturatedHeterocycles = Lipinski.NumSaturatedHeterocycles(mol)
        NumSaturatedRings = Lipinski.NumSaturatedRings(mol)
        CalcNumAmideBonds = rdMolDescriptors.CalcNumAmideBonds(mol)
        CalcNumSpiroAtoms = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        
        num_carboxyl_groups = len(mol.GetSubstructMatches(MolFromSmiles("[C](=O)[OH]"))) # "[C;X3](=O)[OH1]" not working
        num_amion_groups = len(mol.GetSubstructMatches(MolFromSmiles("[NH2]")))
        num_ammonium_groups = len(mol.GetSubstructMatches(MolFromSmiles("[NH4+]")))
        num_sulfonic_acid_groups = len(mol.GetSubstructMatches(MolFromSmiles("[S](=O)(=O)[O-]")))
        num_alkoxy_groups = len(mol.GetSubstructMatches(MolFromSmiles('CO'))) # "[*]-O-[*]" not working
        
        return [MolWt,
                HeavyAtomMolWt,
                NumValenceElectrons,
                FractionCSP3,
                HeavyAtomCount,
                NHOHCount,
                NOCount,
                NumAliphaticCarbocycles,
                NumAliphaticHeterocycles,
                NumAliphaticRings,
                NumAromaticCarbocycles,
                NumAromaticHeterocycles,
                NumAromaticRings,
                NumHAcceptors,
                NumHDonors,
                NumHeteroatoms,
                NumRotatableBonds,
                RingCount,
                MolMR,
                CalcNumBridgeheadAtom,
                ExactMolWt,
                NumRadicalElectrons,
                # MaxPartialCharge,
                # MinPartialCharge,
                # MaxAbsPartialCharge,
                # MinAbsPartialCharge,
                NumSaturatedCarbocycles,
                NumSaturatedHeterocycles,
                NumSaturatedRings,
                MolLogP,
                CalcNumAmideBonds,
                CalcNumSpiroAtoms,
                num_carboxyl_groups,
                num_amion_groups,
                num_ammonium_groups,
                num_sulfonic_acid_groups,
                num_alkoxy_groups]

    

    def generate_mol_atomic_features(self, smiles):

        mol = Chem.MolFromSmiles(smiles)

        # gathering atomic feature 
        mendel_atom_features = [] 
        rdkit_atom_features = [] 
        for atom in mol.GetAtoms():
            mendel_atom_f, rdkit_atom_f = self.get_atomic_features(atom)

            mendel_atom_features.append(mendel_atom_f)
            rdkit_atom_features.append(rdkit_atom_f)

        atomic_features = np.concatenate([mendel_atom_features, rdkit_atom_features], axis=1, dtype=np.float32)
        
        return atomic_features
    
    def get_adj_matrix(self, smiles):
        
        mol = MolFromSmiles(smiles)
        adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
        
        edge_index, edge_attr = dense_to_sparse(Tensor(adj))
        
        bonds = []
        for i in range(0, mol.GetNumAtoms()):
            for j in range(0, mol.GetNumAtoms()):
                if adj[i, j] == 1:
                    bonds.append([i, j])
            
        return edge_index, edge_attr


if __name__ == '__main__' : 
    import pandas as pd 
    from tqdm import tqdm
    
    train = pd.read_csv('/root/Competitions/DACON/4. JUMP AI 2023/data/new_train.csv')
    test  = pd.read_csv('/root/Competitions/DACON/4. JUMP AI 2023/data/new_test.csv')
    
    generator = Chemical_feature_generator()
    
    def process(df):
        molecular_f = [] 
        for sample in tqdm(df.SMILES):
            _, molecular_features = generator.generate_molecular_features(smiles=sample)
            generator.get_adj_matrix(sample)
            molecular_f.append(molecular_features)
            break
    #     molecular_f = np.concatenate([molecular_f], axis=0)
    #     print(molecular_f.shape)
        
    #     return pd.DataFrame(data=molecular_f, columns=['MolWt','HeavyAtomMolWt','NumValenceElectrons','FractionCSP3','HeavyAtomCount','NHOHCount','NOCount','NumAliphaticCarbocycles','NumAliphaticHeterocycles','NumAliphaticRings','NumAromaticCarbocycles','NumAromaticHeterocycles','NumAromaticRings','NumHAcceptors','NumHDonors','NumHeteroatoms','NumRotatableBonds','RingCount','MolMR','CalcNumBridgeheadAtom','ExactMolWt','NumRadicalElectrons','NumSaturatedCarbocycles','NumSaturatedHeterocycles','NumSaturatedRings','MolLogP','CalcNumAmideBonds','CalcNumSpiroAtoms','num_carboxyl_groups','num_amion_groups','num_ammonium_groups','num_sulfonic_acid_groups','num_alkoxy_groups'])
    
    print(train.iloc[1, :])
    print(test.columns)
    # train_molecular_f = process(train)
    # train_merged = pd.concat([train, train_molecular_f], axis=1)
    
    # test_molecular_f = process(test)
    # test_merged = pd.concat([test, test_molecular_f], axis=1)
    
    # print(train_merged.shape)
    # print(test_merged.shape)
    
    # train_merged.to_csv('/root/Competitions/DACON/4. JUMP AI 2023/data/new_train.csv')
    # test_merged.to_csv('/root/Competitions/DACON/4. JUMP AI 2023/data/new_test.csv')