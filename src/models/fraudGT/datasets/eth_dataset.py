import json, itertools
import sys, os
import os.path as osp
import pandas as pd
import numpy as np
import datatable as dt
from datetime import datetime
from datatable import f,join,sort
from collections import defaultdict
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)

from torch_geometric.utils import index_to_mask
from .utils import download_dataset
from .temporal_dataset import TemporalDataset

def z_norm(data):
    std = data.std(0).unsqueeze(0)
    std = torch.where(std == 0, torch.tensor(1, dtype=torch.float32).cpu(), std)
    return (data - data.mean(0).unsqueeze(0)) / std

def to_adj_nodes_with_times(data):
    num_nodes = data.num_nodes
    timestamps = torch.zeros((data.edge_index.shape[1], 1)) if data['node', 'to', 'node'].timestamps is None else data['node', 'to', 'node'].timestamps.reshape((-1,1))
    edges = torch.cat((data.edge_index.T, timestamps), dim=1) if not isinstance(data, HeteroData) else torch.cat((data['node', 'to', 'node'].edge_index.T, timestamps), dim=1)
    adj_list_out = dict([(i, []) for i in range(num_nodes)])
    adj_list_in = dict([(i, []) for i in range(num_nodes)])
    for u,v,t in edges:
        u,v,t = int(u), int(v), int(t)
        adj_list_out[u] += [(v, t)]
        adj_list_in[v] += [(u, t)]
    return adj_list_in, adj_list_out

def ports(edge_index, adj_list):
    ports = torch.zeros(edge_index.shape[1], 1)
    ports_dict = {}
    for v, nbs in adj_list.items():
        if len(nbs) < 1: continue
        a = np.array(nbs)
        a = a[a[:, -1].argsort()]
        _, idx = np.unique(a[:,[0]],return_index=True,axis=0)
        nbs_unique = a[np.sort(idx)][:,0]
        for i, u in enumerate(nbs_unique):
            ports_dict[(u,v)] = i
    for i, e in enumerate(edge_index.T):
        ports[i] = ports_dict[tuple(e.numpy())]
    return ports

class ETHDataset(TemporalDataset):

    url = 'https://drive.google.com/file/d/1GuGgGaJRDXLsOkkHaRfemvtrfYzCPh9q/view?usp=drive_link'

    def __init__(self, root: str, reverse_mp: bool = False,
                 add_ports: bool = False,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = 'ETH'
        self.reverse_mp = reverse_mp
        self.add_ports = add_ports
        super().__init__(root, transform, pre_transform)
        self.data_dict = torch.load(self.processed_paths[0])
        # del self._data['node'].x
        if not reverse_mp:
            for split in ['train', 'val', 'test']:
                del self.data_dict[split]['node', 'rev_to', 'node']
            # del self.slices['node', 'rev_to', 'node']
        if add_ports:
            self.ports_dict = torch.load(self.processed_paths[1])
            for split in ['train', 'val', 'test']:
                self.data_dict[split] = self.add_ports_func(self.data_dict[split], self.ports_dict[split])

    def add_ports_func(self, data, ports):
        reverse_ports = True
        if not self.reverse_mp:
            # adj_list_in, adj_list_out = to_adj_nodes_with_times(data)
            # in_ports = ports(data['node', 'to', 'node'].edge_index, adj_list_in)
            # out_ports = [ports(data['node', 'to', 'node'].edge_index.flipud(), adj_list_out)] if reverse_ports else []
            in_ports, out_ports = ports
            out_ports = [out_ports]
            data['node', 'to', 'node'].edge_attr = \
                torch.cat([data['node', 'to', 'node'].edge_attr, in_ports] + out_ports, dim=1)
            # return data

        else:
            '''Adds port numberings to the edge features'''
            # adj_list_in, adj_list_out = to_adj_nodes_with_times(data)
            # in_ports = ports(data['node', 'to', 'node'].edge_index, adj_list_in)
            # out_ports = ports(data['node', 'rev_to', 'node'].edge_index, adj_list_out)
            in_ports, out_ports = ports
            data['node', 'to', 'node'].edge_attr = torch.cat([data['node', 'to', 'node'].edge_attr, in_ports], dim=1)
            data['node', 'rev_to', 'node'].edge_attr = torch.cat([data['node', 'rev_to', 'node'].edge_attr, out_ports], dim=1)
        return data

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        file_names = ['eth_transactions_all.csv', 'node_labels.csv']
        return file_names

    @property
    def processed_file_names(self) -> str:
        return ['data.pt', 'ports.pt']

    def download(self):
        url = self.url
        download_dataset(url, self.root)

    def process(self):
        # eth_transactions_all.csv: contains ETH transactions
        df_edges = pd.read_csv(osp.join(self.raw_dir, 'eth_transactions_all.csv'))
        # node_labels.csv: contains label for each ETH account: 0 - no phishing, 1 - phishing
        df_nodes = pd.read_csv(osp.join(self.raw_dir, 'node_labels.csv'))

        print(f'Available Edge Features: {df_edges.columns.tolist()}')

        df_edges['Timestamp'] = df_edges['Timestamp'] - df_edges['Timestamp'].min()
        # The original timestamps are already sorted, no need to sort again
        df_nodes['Feature'] = np.ones(len(df_nodes))
        timestamps = torch.Tensor(df_edges['Timestamp'].to_numpy())

        y = torch.LongTensor(df_nodes['Is Phishing'].to_numpy())
        print(f"Illicit ratio = {sum(y)} / {len(y)} = {sum(y) / len(y) * 100:.2f}%")
        print(f"Number of nodes (holdings doing transcations) = {df_nodes.shape[0]}")
        print(f"Number of transactions = {df_edges.shape[0]}")

        edge_features = ['Timestamp', 'Value', 'Nonce', 'Block Nr', 'Gas', 'Gas Price', 'Transaction Type']
        node_features = ['Feature']

        print(f'Edge features being used: {edge_features}')
        print(f'Node features being used: {node_features} ("Feature" is a placeholder feature of all 1s)')

        x = torch.Tensor(df_nodes.loc[:, node_features].to_numpy())
        edge_index = torch.LongTensor(df_edges.loc[:, ['Source Node', 'Destination Node']].to_numpy().T)
        edge_attr = torch.Tensor(df_edges.loc[:, edge_features].to_numpy())

        n_days = int(timestamps.max() / (3600 * 24) + 1)
        n_samples = y.shape[0]
        print(f'number of days and transactions in the data: {n_days} days, {n_samples} accounts')
        
        source_timestamps = df_edges.groupby('Source Node')['Timestamp'].first().reset_index()
        destination_timestamps = df_edges.groupby('Destination Node')['Timestamp'].first().reset_index()
        source_timestamps.columns = ['Node', 'Timestamp']
        destination_timestamps.columns = ['Node', 'Timestamp']
        first_timestamps = pd.concat([source_timestamps, destination_timestamps], ignore_index=True)
        first_timestamps = first_timestamps.groupby('Node')['Timestamp'].min().reset_index()
        sorted_nodes = first_timestamps.sort_values(by='Timestamp').reset_index(drop=True)
        mask_inds = torch.LongTensor(sorted_nodes['Node'].to_numpy())

        split_per = [0.65, 0.15, 0.2]
        train_inds = mask_inds[:int(n_samples * split_per[0])]
        val_inds = mask_inds[int(n_samples * split_per[0]) : int(n_samples * sum(split_per[:2]))]
        test_inds = mask_inds[int(n_samples * sum(split_per[:2])):]

        node_train = train_inds
        node_val = torch.cat([train_inds, val_inds])
        node_test = torch.cat([train_inds, val_inds, test_inds])
        e_train = torch.isin(edge_index[0], node_train) & torch.isin(edge_index[1], node_train)
        e_val = torch.isin(edge_index[0], node_val) & torch.isin(edge_index[1], node_val)
        e_test = torch.isin(edge_index[0], node_test) & torch.isin(edge_index[1], node_test)

        self.ports_dict = {}
        self.data_dict = {}
        for split in ['train', 'val', 'test']:
            inds = eval(f'{split}_inds')
            e_mask = eval(f'e_{split}')

            masked_edge_index = edge_index[:, e_mask]
            masked_edge_attr = z_norm(edge_attr[e_mask])
            masked_y = y[inds]
            masked_timestamps = timestamps[e_mask]

            data = HeteroData()
            data['node'].x = x # z_norm(x) will render all x be 0
            data['node'].y = y # masked_y
            data['node'].num_nodes = int(x.shape[0])
            data['node', 'to', 'node'].edge_index = masked_edge_index
            data['node', 'to', 'node'].edge_attr = masked_edge_attr
            # We use "y" here so LinkNeighborLoader won't mess up the edge label
            data['node', 'to', 'node'].timestamps = masked_timestamps
            # if args.ports:
            #     #swap the in- and outgoing port numberings for the reverse edges
            #     data['node', 'rev_to', 'node'].edge_attr[:, [-1, -2]] = data['node', 'rev_to', 'node'].edge_attr[:, [-2, -1]]

            data['node', 'rev_to', 'node'].edge_index = masked_edge_index.flipud()
            data['node', 'rev_to', 'node'].edge_attr = masked_edge_attr

            # Define the labels in the training/validation/test sets
            data['node'].train_mask = index_to_mask(train_inds, size=data['node'].num_nodes)
            data['node'].val_mask = index_to_mask(val_inds, size=data['node'].num_nodes)
            data['node'].test_mask = index_to_mask(test_inds, size=data['node'].num_nodes)
            data['node'].split_mask = index_to_mask(inds, size=data['node'].num_nodes)

            adj_list_in, adj_list_out = to_adj_nodes_with_times(data)
            in_ports = ports(data['node', 'to', 'node'].edge_index, adj_list_in)
            out_ports = ports(data['node', 'to', 'node'].edge_index.flipud(), adj_list_out)
            self.ports_dict[split] = [in_ports, out_ports]
            self.data_dict[split] = data

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.data_dict, self.processed_paths[0])
        torch.save(self.ports_dict, self.processed_paths[1])

    def __repr__(self) -> str:
        return 'ETH_Dataset()'