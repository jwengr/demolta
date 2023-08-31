import random
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, AutoConfig
from dgllife.utils import (
    atom_type_one_hot, 
    atom_formal_charge_one_hot, 
    atom_total_degree_one_hot, 
    atom_explicit_valence_one_hot, 
    atom_implicit_valence_one_hot, 
    atom_is_aromatic_one_hot, 
    atom_hybridization_one_hot, 
    atom_total_num_H_one_hot, 
    atom_is_in_ring_one_hot, 
)
from dgllife.utils import (
    bond_type_one_hot,
    bond_is_conjugated_one_hot,
    bond_is_in_ring_one_hot,
    bond_stereo_one_hot
)
from rdkit import Chem
from rdkit.Chem import AllChem
from pytorch_metric_learning.losses import NTXentLoss, SelfSupervisedLoss


class DeMOLTaConfig:
    def __init__(
            self,
            num_atom=64+1,
            num_atom_charge=5+1,
            num_degree=11+1,
            num_explicit_valence=7+1,
            num_implicit_valence=6+1,
            num_aromatic=2+1,
            num_hybridization=5+1,
            num_total_num_H=5+1,    
            num_is_in_ring=2+1,
            num_bond_type=4+1,
            num_conjugated=2+1,
            num_ring=2+1,
            num_stereo=6+1,
            num_shortest_path=6+1,
            node_hidden_dim=768,
            edge_hidden_dim=256,
            num_heads=32,
            node_ff_dim=768,
            edge_ff_dim=256,
            num_layers=12,
            dropout=0.1,
        ):
        self.num_atom = num_atom
        self.num_atom_charge = num_atom_charge
        self.num_degree = num_degree
        self.num_explicit_valence = num_explicit_valence
        self.num_implicit_valence = num_implicit_valence
        self.num_aromatic = num_aromatic
        self.num_hybridization = num_hybridization
        self.num_total_num_H = num_total_num_H
        self.num_is_in_ring = num_is_in_ring
        self.num_bond_type = num_bond_type
        self.num_conjugated = num_conjugated
        self.num_ring = num_ring
        self.num_stereo = num_stereo
        self.num_shortest_path = num_shortest_path
        self.node_hidden_dim = node_hidden_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.num_heads = num_heads
        self.node_ff_dim = node_ff_dim
        self.edge_ff_dim = edge_ff_dim
        self.num_layers = num_layers
        self.dropout = dropout

def find_index(lst, index, default):
    if index in lst:
        return lst.index(index)
    else:
        return default

class DeMOLTaFeaturizer:
    def __init__(self):
        self.atom_symbols = [
            'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
            'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag',
            'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni',
            'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb',
            'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os',
            'Ir', 'Ce', 'Gd', 'Ga', 'Cs', '*', 'UNK'
        ]
        self.bond_shortest_paths_max = 5

    def _get_atom_feats(self, mol: Chem.Mol, conformer: Chem.Conformer = None):
        atoms = mol.GetAtoms()
        atom_numbers = []
        atom_charges = []
        atom_degrees = []
        atom_explicit_valences = []
        atom_implicit_valences = []
        atom_aromatics = []
        atom_hybridizations = []
        atom_total_num_Hs = []
        atom_is_in_rings = []
        atom_positions = []

        for atom in atoms:
            atom_number = atom_type_one_hot(atom, self.atom_symbols)
            atom_number = find_index(atom_number, 1, -1)
            atom_numbers.append(atom_number)
            atom_charge = atom_formal_charge_one_hot(atom)
            atom_charge = find_index(atom_charge, 1, -1)
            atom_charges.append(atom_charge)
            atom_degree = atom_total_degree_one_hot(atom, list(range(11)))
            atom_degree = find_index(atom_degree, 1, -1)
            atom_degrees.append(atom_degree)
            atom_explicit_valence = atom_explicit_valence_one_hot(atom, list(range(7)))
            atom_explicit_valence = find_index(atom_explicit_valence, 1, -1)
            atom_explicit_valences.append(atom_explicit_valence)
            atom_implicit_valence = atom_implicit_valence_one_hot(atom, list(range(6)))
            atom_implicit_valence = find_index(atom_implicit_valence, 1, -1)
            atom_implicit_valences.append(atom_implicit_valence)
            atom_aromatic = atom_is_aromatic_one_hot(atom)
            atom_aromatic = find_index(atom_aromatic, 1, -1)
            atom_aromatics.append(atom_aromatic)
            atom_hybridization = atom_hybridization_one_hot(atom)
            atom_hybridization = find_index(atom_hybridization, 1, -1)
            atom_hybridizations.append(atom_hybridization)
            atom_total_num_H = atom_total_num_H_one_hot(atom, list(range(5)))
            atom_total_num_H = find_index(atom_total_num_H, 1, -1)
            atom_total_num_Hs.append(atom_total_num_H)
            atom_is_in_ring = atom_is_in_ring_one_hot(atom)
            atom_is_in_ring = find_index(atom_is_in_ring, 1, -1)
            atom_is_in_rings.append(atom_is_in_ring)
            atom_position = conformer.GetAtomPosition(atom.GetIdx()) if conformer else np.zeros(3)
            atom_positions.append(atom_position)

        atom_feats = {
            'atomic_number': torch.LongTensor(atom_numbers)+1,
            'formal_charge': torch.LongTensor(atom_charges)+1,
            'degree': torch.LongTensor(atom_degrees)+1,
            'explicit_valence': torch.LongTensor(atom_explicit_valences)+1,
            'implicit_valence': torch.LongTensor(atom_implicit_valences)+1,
            'aromatic': torch.LongTensor(atom_aromatics)+1,
            'hybridization': torch.LongTensor(atom_hybridizations)+1,
            'total_num_H': torch.LongTensor(atom_total_num_Hs)+1,
            'is_in_ring': torch.LongTensor(atom_is_in_rings)+1,
            'position': torch.FloatTensor(atom_positions)
        }
        return atom_feats
    
    def _get_bond_feats(self, mol: Chem.Mol, conformer: Chem.Conformer = None):
        mol_len = mol.GetNumAtoms()
        bond_types = np.zeros((mol_len, mol_len), dtype=int)-1
        bond_conjugateds = np.zeros((mol_len, mol_len), dtype=int)-1
        bond_rings = np.zeros((mol_len, mol_len), dtype=int)-1
        bond_stereos = np.zeros((mol_len, mol_len), dtype=int)-1
        bond_shortest_paths = np.zeros((mol_len, mol_len), dtype=int)-1
        bond_relative_distances = np.zeros((mol_len, mol_len), dtype=float)

        graph = nx.Graph() 
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = bond_type_one_hot(bond)
            bond_type = find_index(bond_type, 1, -1)
            bond_types[i, j] = bond_types[j, i] = bond_type
            bond_conjugated = bond_is_conjugated_one_hot(bond)
            bond_conjugated = find_index(bond_conjugated, 1, -1)
            bond_conjugateds[i, j] = bond_conjugateds[j, i] = bond_conjugated
            bond_ring = bond_is_in_ring_one_hot(bond)
            bond_ring = find_index(bond_ring, 1, -1)
            bond_rings[i, j] = bond_rings[j, i] = bond_ring
            bond_stereo = bond_stereo_one_hot(bond)
            bond_stereo = find_index(bond_stereo, 1, -1)
            bond_stereos[i, j] = bond_stereos[j, i] = bond_stereo
            graph.add_edge(i, j)
        
        shortest_paths = dict(nx.all_pairs_shortest_path_length(graph))
        for i in shortest_paths:
            for j in shortest_paths[i]:
                bond_shortest_paths[i][j] = min(shortest_paths[i][j], self.bond_shortest_paths_max)

        for i in range(mol_len):
            for j in range(i, mol_len):
                bond_relative_distance = 0
                if conformer is not None:
                    atom1_position = conformer.GetAtomPosition(i)
                    atom2_position = conformer.GetAtomPosition(j)
                    bond_relative_distance = np.linalg.norm(atom1_position - atom2_position)
                bond_relative_distances[i, j] = bond_relative_distances[j, i] = bond_relative_distance

        bond_feats = {
            'bond_type': torch.LongTensor(bond_types)+1,
            'conjugated': torch.LongTensor(bond_conjugateds)+1,
            'ring': torch.LongTensor(bond_rings)+1,
            'stereo': torch.LongTensor(bond_stereos)+1,
            'shortest_path': torch.LongTensor(bond_shortest_paths)+1,
            'relative_distance': torch.FloatTensor(bond_relative_distances)
        }
        return bond_feats
    
    def __call__(self, mol: Chem.Mol = None, smiles: str = None):
        if mol is None:
            mol = Chem.MolFromSmiles(smiles)
        conformer = None
        if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) == 0:
            conformer = mol.GetConformer()

        atom_feats = self._get_atom_feats(mol, conformer)
        bond_feats = self._get_bond_feats(mol, conformer)
        sample = {
            'atom_feats': atom_feats,
            'bond_feats': bond_feats
        }
        return sample
    
class DeMOLTaCollateFn:
    def __init__(self):
        pass

    def __call__(self, samples):
        atom_feats = [sample['atom_feats'] for sample in samples]
        bond_feats = [sample['bond_feats'] for sample in samples]
        max_len = max([len(atom_feat['atomic_number']) for atom_feat in atom_feats])

        batch_atom_feats = {}
        for atom_feat in atom_feats:
            for key in atom_feat:
                batch_atom_feats.setdefault(key, [])
                atom_feat_type = atom_feat[key].dtype
                if len(atom_feat[key].shape) == 1:
                    temp = torch.zeros(max_len, dtype=atom_feat_type)
                    temp[:len(atom_feat[key])] = atom_feat[key]
                else :
                    temp = torch.zeros((max_len, atom_feat[key].shape[1]), dtype=atom_feat_type)
                    temp[:atom_feat[key].shape[0] , :atom_feat[key].shape[1]] = atom_feat[key]
                batch_atom_feats[key].append(temp)

        batch_bond_feats = {}
        for bond_feat in bond_feats:
            for key in bond_feat:
                bond_feat_type = bond_feat[key].dtype
                batch_bond_feats.setdefault(key, [])
                temp = torch.zeros((max_len, max_len), dtype=bond_feat_type)
                temp[:len(bond_feat[key]), :len(bond_feat[key])] = bond_feat[key]
                batch_bond_feats[key].append(temp)

        for key in batch_atom_feats:
            batch_atom_feats[key] = torch.stack(batch_atom_feats[key], dim=0)
        
        for key in batch_bond_feats:
            batch_bond_feats[key] = torch.stack(batch_bond_feats[key], dim=0)
        
        attention_mask = (batch_bond_feats['shortest_path'] != 0) * 1

        batch = {
            'atom_feats': batch_atom_feats,
            'bond_feats': batch_bond_feats,
            'attention_mask': attention_mask
        }
        return batch

    
class DeMOLTaAtomEmbedding(nn.Module):
    def __init__(
            self, 
            num_atom, 
            num_atom_charge, 
            num_degree,
            num_explicit_valence,
            num_implicit_valence,
            num_aromatic,
            num_hybridization,
            num_total_num_H,
            num_is_in_ring,
            hidden_dim,
            dropout=0.1
        ):
        super(DeMOLTaAtomEmbedding, self).__init__()
        self.atom_number_embedding = nn.Embedding(num_atom, hidden_dim, padding_idx=0)
        self.atom_charge_embedding = nn.Embedding(num_atom_charge, hidden_dim, padding_idx=0)
        self.degree_embedding = nn.Embedding(num_degree, hidden_dim, padding_idx=0)
        self.explicit_valence_embedding = nn.Embedding(num_explicit_valence, hidden_dim, padding_idx=0)
        self.implicit_valence_embedding = nn.Embedding(num_implicit_valence, hidden_dim, padding_idx=0)
        self.aromatic_embedding = nn.Embedding(num_aromatic, hidden_dim, padding_idx=0)
        self.hybridization_embedding = nn.Embedding(num_hybridization, hidden_dim, padding_idx=0)
        self.total_num_H_embedding = nn.Embedding(num_total_num_H, hidden_dim, padding_idx=0)
        self.is_in_ring_embedding = nn.Embedding(num_is_in_ring, hidden_dim, padding_idx=0)
        self.position_weights = nn.Parameter(torch.empty(3, hidden_dim).uniform_(-1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, atom_feats):
        atom_number = self.atom_number_embedding(atom_feats['atomic_number'])
        atom_charge = self.atom_charge_embedding(atom_feats['formal_charge'])
        degree = self.degree_embedding(atom_feats['degree'])
        explicit_valence = self.explicit_valence_embedding(atom_feats['explicit_valence'])
        implicit_valence = self.implicit_valence_embedding(atom_feats['implicit_valence'])
        aromatic = self.aromatic_embedding(atom_feats['aromatic'])
        hybridization = self.hybridization_embedding(atom_feats['hybridization'])
        total_num_H = self.total_num_H_embedding(atom_feats['total_num_H'])
        is_in_ring = self.is_in_ring_embedding(atom_feats['is_in_ring'])
        position = torch.matmul(atom_feats['position'], self.position_weights)
        output = self.dropout(atom_number + atom_charge + degree + explicit_valence + implicit_valence + aromatic + hybridization + total_num_H + is_in_ring + position)
        return output
    
class DeMOLTaBondEmbedding(nn.Module):
    def __init__(
            self, 
            num_bond_type, 
            num_conjugated,
            num_ring,
            num_stereo,
            num_shortest_path,
            hidden_dim, 
            dropout=0.1
        ):
        super(DeMOLTaBondEmbedding, self).__init__()
        self.bond_type_embedding = nn.Embedding(num_bond_type, hidden_dim, padding_idx=0)
        self.conjugated_embedding = nn.Embedding(num_conjugated, hidden_dim, padding_idx=0)
        self.ring_embedding = nn.Embedding(num_ring, hidden_dim, padding_idx=0)
        self.stereo_embedding = nn.Embedding(num_stereo, hidden_dim, padding_idx=0)
        self.shortest_path_embedding = nn.Embedding(num_shortest_path, hidden_dim, padding_idx=0)
        self.relative_distance_weights = nn.Parameter(torch.empty(1, hidden_dim).uniform_(-1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, bond_feats):
        bond_type = self.bond_type_embedding(bond_feats['bond_type'])
        conjugated = self.conjugated_embedding(bond_feats['conjugated'])
        ring = self.ring_embedding(bond_feats['ring'])
        stereo = self.stereo_embedding(bond_feats['stereo'])
        shortest_path = self.shortest_path_embedding(bond_feats['shortest_path'])
        relative_distance = bond_feats['relative_distance'].unsqueeze(-1) * self.relative_distance_weights
        output = self.dropout(bond_type + conjugated + ring + stereo + shortest_path + relative_distance)
        return output

class DeMOLTaEmbedding(nn.Module):
    def __init__(
            self, 
            node_hidden_dim,
            edge_hidden_dim,
            num_atom=64+1, 
            num_atom_charge=5+1, 
            num_degree=11+1,
            num_explicit_valence=7+1,
            num_implicit_valence=6+1,
            num_aromatic=2+1,
            num_hybridization=5+1,
            num_total_num_H=5+1,
            num_is_in_ring=2+1,
            num_bond_type=4+1, 
            num_conjugated=2+1,
            num_ring=2+1,
            num_stereo=6+1,
            num_shortest_path=6+1,
            dropout=0.1
        ):
        super(DeMOLTaEmbedding, self).__init__()
        self.atom_embedding = DeMOLTaAtomEmbedding(
            num_atom, 
            num_atom_charge, 
            num_degree,
            num_explicit_valence,
            num_implicit_valence,
            num_aromatic,
            num_hybridization,
            num_total_num_H,
            num_is_in_ring,
            node_hidden_dim, 
            dropout
        )
        self.bond_embedding = DeMOLTaBondEmbedding(
            num_bond_type, 
            num_conjugated,
            num_ring,
            num_stereo,
            num_shortest_path,
            edge_hidden_dim, 
            dropout
        )
    
    def forward(self, atom_feats, bond_feats):
        atom_embedding = self.atom_embedding(atom_feats)
        bond_embedding = self.bond_embedding(bond_feats)
        return atom_embedding, bond_embedding

    
class DeMOLTaAttention(nn.Module):
    def __init__(self, node_hidden_dim, edge_hidden_dim, num_heads, dropout=0.1):
        super(DeMOLTaAttention, self).__init__()
        self.qkv = nn.Linear(node_hidden_dim, node_hidden_dim*3)
        self.rqk = nn.Linear(edge_hidden_dim, num_heads*2)
        self.num_heads = num_heads
        self.head_dim = node_hidden_dim // num_heads
        self.scale = 1/ ( (3*self.head_dim) ** 0.5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, p, attention_matrix_mask):
        batch_size, seq_len, hidden_dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, self.num_heads, 3 * self.head_dim).permute(0, 2, 1, 3) # b, h, l, d
        q, k, v = qkv.chunk(3, dim=-1)
        rqk = self.rqk(p)
        rqk = rqk.view(batch_size, seq_len, seq_len, self.num_heads, 2).permute(0, 3, 1, 2, 4) # b, h, l, l, 2
        rq, rk = rqk.chunk(2, dim=-1)
        attention = torch.matmul(q, k.transpose(-2, -1)) + (rq * k.unsqueeze(-2)).sum(dim=-1) + (rk * q.unsqueeze(-2)).sum(dim=-1)
        attention.masked_fill_(attention_matrix_mask.unsqueeze(1)==0, -1e4)
        attention_probs = F.softmax(attention * self.scale, dim=-1)
        attention_probs = self.dropout(attention_probs)
        output = torch.matmul(attention_probs, v).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, hidden_dim)
         
        return output
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_dim, ff_dim, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.dropout(F.gelu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
class NodeEncoderLayerWithPreLayerNorm(nn.Module):
    def __init__(self, node_hidden_dim, edge_hidden_dim, num_heads, ff_dim, dropout=0.1):
        super(NodeEncoderLayerWithPreLayerNorm, self).__init__()
        self.attention = DeMOLTaAttention(node_hidden_dim, edge_hidden_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(node_hidden_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(node_hidden_dim)
        self.norm2 = nn.LayerNorm(node_hidden_dim)
        
    def forward(self, x, p, attention_matrix_mask):
        x + self.attention(self.norm1(x), p, attention_matrix_mask)
        x = x + self.ffn(self.norm2(x))
        return x
    
class OuterProduct(nn.Module):
    def __init__(self, node_hidden_dim, edge_hidden_dim, num_heads, dropout=0.1):
        super(OuterProduct, self).__init__()
        self.o12 = nn.Linear(node_hidden_dim, num_heads*2)
        self.o3 = nn.Linear(num_heads**2, edge_hidden_dim)
        
    def forward(self, x):
        o1, o2 = self.o12(x).chunk(2, dim=-1)
        a, b = o1.unsqueeze(2).unsqueeze(4), o2.unsqueeze(1).unsqueeze(3)
        o = torch.matmul(a,b).flatten(3)
        output = self.o3(o)
        return output
    
class TriangularUpdate(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super(TriangularUpdate, self).__init__()
        self.t = nn.Linear(hidden_dim, num_heads*4)
        self.t5 = nn.Linear(hidden_dim, hidden_dim)
        self.t6 = nn.Linear(num_heads, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, p, attention_matrix_mask):
        t1, t2, t3, t4= self.t(p).chunk(5, dim=-1)
        t5 = self.t5(p)
        t12 = t1 * t2
        t12.masked_fill_(attention_matrix_mask.unsqueeze(-1)==0, -1e4)
        t34 = t3 * t4
        t34.masked_fill_(attention_matrix_mask.unsqueeze(-1)==0, -1e4)
        a = torch.sigmoid(t12).sum(dim=1).unsqueeze(1)
        b = torch.sigmoid(t34).sum(dim=2).unsqueeze(2)
        o = a + b
        o = self.dropout(o)
        t56 = t5 * self.t6(o)
        t56.masked_fill_(attention_matrix_mask.unsqueeze(-1)==0, -1e4)
        output = torch.sigmoid(t56)
        return output

class EdgeEncoderLayerWithPreLayerNorm(nn.Module):
    def __init__(self, node_hidden_dim, edge_hidden_dim, num_heads, ff_dim, dropout=0.1):
        super(EdgeEncoderLayerWithPreLayerNorm, self).__init__()
        self.outer_product = OuterProduct(node_hidden_dim, edge_hidden_dim, num_heads, dropout)
        self.triangular_update = TriangularUpdate(edge_hidden_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(edge_hidden_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(node_hidden_dim)
        self.norm2 = nn.LayerNorm(edge_hidden_dim)
        self.norm3 = nn.LayerNorm(edge_hidden_dim)
        
    def forward(self, x, p, attention_matrix_mask):
        p = p + self.outer_product(self.norm1(x))
        p = p + self.triangular_update(self.norm2(p), attention_matrix_mask)
        p = p + self.ffn(self.norm3(p))
        return p
    
class EncoderLayer(nn.Module):
    def __init__(self, node_hidden_dim, edge_hidden_dim, num_heads, node_ff_dim, edge_ff_dim, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.node_encoder_layer = NodeEncoderLayerWithPreLayerNorm(node_hidden_dim, edge_hidden_dim, num_heads, node_ff_dim, dropout)
        self.edge_encoder_layer = EdgeEncoderLayerWithPreLayerNorm(node_hidden_dim, edge_hidden_dim, num_heads, edge_ff_dim, dropout)
        
    def forward(self, x, p, attention_matrix_mask):
        x = self.node_encoder_layer(x, p, attention_matrix_mask)
        p = self.edge_encoder_layer(x, p, attention_matrix_mask)
        return x, p
    
class DeMOLTaEncoder(nn.Module):
    def __init__(self, node_hidden_dim, edge_hidden_dim, num_heads, node_ff_dim, edge_ff_dim, num_layers, dropout=0.1):
        super(DeMOLTaEncoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(node_hidden_dim, edge_hidden_dim, num_heads, node_ff_dim, edge_ff_dim, dropout) for _ in range(num_layers)])
        
    def forward(self, x, p, attention_matrix_mask):
        for layer in self.layers:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            
            x, p = torch.utils.checkpoint.checkpoint(
                create_custom_forward(layer), x, p, attention_matrix_mask
            )

        return x, p
    
class DeMOLTaModel(nn.Module):
    def __init__(self, config):
        super(DeMOLTaModel, self).__init__()
        self.config = config
        self.embedding = DeMOLTaEmbedding(
            config.node_hidden_dim,
            config.edge_hidden_dim,
            config.num_atom,
            config.num_atom_charge,
            config.num_degree,
            config.num_explicit_valence,
            config.num_implicit_valence,
            config.num_aromatic,
            config.num_hybridization,
            config.num_total_num_H,
            config.num_is_in_ring,
            config.num_bond_type,
            config.num_conjugated,
            config.num_ring,
            config.num_stereo,
            config.num_shortest_path,
            config.dropout
        )
        self.encoder = DeMOLTaEncoder(config.node_hidden_dim, config.edge_hidden_dim, config.num_heads, config.node_ff_dim, config.edge_ff_dim, config.num_layers, config.dropout)

        
    def forward(self, atom_feats, bond_feats, attention_matrix_mask):
        x, p = self.embedding(atom_feats, bond_feats)
        x, p = self.encoder(x, p, attention_matrix_mask)
        return x, p
    

class MOLLACollateFn:
    def __init__(self, tokenizer):
        self.mol_collate_fn = DeMOLTaCollateFn()
        self.tokenizer = tokenizer

    def __call__(self, samples):
        mols = self.mol_collate_fn([sample['mol_feats'] for sample in samples])
        labels = [sample['query'] + ' ' + sample['answer'] for sample in samples]
        encode = self.tokenizer(labels, padding=True, truncation=True, return_tensors='pt')
        input_ids = encode['input_ids']
        attention_mask = encode['attention_mask']
        label_ids = encode['input_ids'].clone()

        batch = {
            'mols': mols,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label_ids,
        }
        return batch
    
class FineTuneCollateFn:
    def __init__(self):
        self.mol_collate_fn = DeMOLTaCollateFn()

    def __call__(self, samples):
        mols = self.mol_collate_fn([sample['mol_feats'] for sample in samples])
        labels = torch.FloatTensor([sample['label'] for sample in samples])
        batch = {
            'mols': mols,
            'labels': labels,
        }
        return batch

class MOLLA(nn.Module):
    def __init__(self, mol_config, text_model_name, hf_token=None):
        super(MOLLA, self).__init__()
        self.mol_model = DeMOLTaModel(mol_config)
        if hf_token:
            self.language_model = AutoModelForCausalLM.from_pretrained(text_model_name, use_auth_token=hf_token, torch_dtype = "auto")
        else:
            self.language_model = AutoModelForCausalLM.from_pretrained(text_model_name)
        self.freeze_language_model()
        self.vocab_size = self.language_model.config.vocab_size
        self.language_projection = nn.Linear(mol_config.node_hidden_dim, self.language_model.config.hidden_size)

    def forward(self, input_ids, input_attention_mask, atom_feats, bond_feats, attention_matrix_mask, labels=None):
        atom_outputs, bond_outputs = self.mol_model(atom_feats, bond_feats, attention_matrix_mask)
        mol_embeds = torch.cat([atom_outputs.mean(dim=1).unsqueeze(1),atom_outputs.max(dim=1).unsqueeze(1)])
        mol_embeds = self.language_projection(mol_embeds)
        mol_attention_mask = torch.ones(
            mol_embeds.size()[:-1], dtype=torch.long, device=mol_embeds.device
        )

        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([mol_embeds, input_embeds], dim=1)
        attention_mask = torch.cat([mol_attention_mask , input_attention_mask], dim=1)
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        logits = outputs.logits
        last_hidden_state = outputs.hidden_states[-1]
        ref_emb = last_hidden_state[:, -1, :]
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            logits = logits[:, -labels.size(1) :, :]
            loss_contrastive_fn = SelfSupervisedLoss(NTXentLoss(temperature=0.1), symmetric=True)
            loss_contrastive = loss_contrastive_fn(mol_embeds.squeeze(1), ref_emb)
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(logits.device)
            loss_language_model_loss_fn = nn.CrossEntropyLoss(reduction="mean")
            loss_language_model_loss = loss_language_model_loss_fn(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
            loss = loss_contrastive + loss_language_model_loss * 5.0
        return loss, logits
        
    def freeze_language_model(self):
        for param in self.language_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def generate(self, input_ids, input_attention_mask, atom_feats, bond_feats, attention_matrix_mask, **generate_kwargs):
        atom_outputs, bond_outputs = self.mol_model(atom_feats, bond_feats, attention_matrix_mask)
        mol_embeds = atom_outputs.mean(dim=1).unsqueeze(1)
        mol_embeds = self.language_projection(mol_embeds)
        mol_attention_mask = torch.ones(
            mol_embeds.size()[:-1], dtype=torch.long, device=mol_embeds.device
        )

        batch_size = mol_embeds.size(0)
        if input_ids is None:
            input_ids = (
                torch.LongTensor([[self.config.text_config.bos_token_id]])
                .repeat(batch_size, 1)
                .to(mol_embeds.device)
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([mol_embeds, input_embeds], dim=1)
        attention_mask = torch.cat([mol_attention_mask , input_attention_mask], dim=1)

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        return outputs
    

class MOLAForMolculeRegression(nn.Module):
    def __init__(self, mol_config, text_model_name, n_class):
        super(MOLAForMolculeRegression, self).__init__()
        self.mol_model = DeMOLTaModel(mol_config)
        self.language_model_config = AutoConfig.from_pretrained(text_model_name)
        self.language_projection = nn.Linear(mol_config.hidden_dim, self.language_model_config.hidden_size)
        self.dropout = nn.Dropout(mol_config.dropout)
        self.regressor = nn.Sequential(
            nn.Linear(mol_config.hidden_dim, mol_config.hidden_dim),
            nn.LayerNorm(mol_config.hidden_dim),
            nn.GELU(),
            nn.Linear(mol_config.hidden_dim, n_class)
        )

    def forward(self, atom_feats, bond_feats, attention_matrix_mask, labels=None):
        atom_outputs, bond_outputs = self.mol_model(atom_feats, bond_feats, attention_matrix_mask)
        mol_embeds = atom_outputs.mean(dim=1)
        mol_embeds = self.dropout(mol_embeds)
        logits = self.regressor(mol_embeds)
        loss1, loss2 = None, None
        if labels is not None:
            loss1 = F.mse_loss(logits[:,0].flatten(), labels[:,0].flatten())
            loss2 = F.mse_loss(logits[:,1].flatten(), labels[:,1].flatten())
        loss = (loss1, loss2)
        return loss, logits
