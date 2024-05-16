from pathlib import Path
from typing import TypeAlias

import torch
from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset
from torch_geometric.data import Data, TemporalData

TimeSpan: TypeAlias = list[int, int]


class TemporalOGBDataset:
    def __init__(
        self,
        root=f"{Path.home()}/Datasets/OGB/",
        name: str = "ogbn-arxiv",
        msg_size: int = 1,
        batch_size: int = 512,
        num_workers: int = 8,
    ) -> None:
        super().__init__()

        self.msg_size = msg_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.name = name
        self.root = root

        self.eval_metric = "acc"
        self.transform = None

        self.setup()

    def prepare_data(self) -> None:
        PygNodePropPredDataset(name=self.name, root=self.root, transform=self.transform)

    def setup(self, stage=None) -> None:
        self.dataset = PygNodePropPredDataset(
            name=self.name, root=self.root, transform=self.transform
        )
        self.data = self.dataset[0]

        split_idx = self.dataset.get_idx_split()
        train_idx = split_idx["train"]
        val_idx = split_idx["valid"]
        test_idx = split_idx["test"]

        self.full_data = self.convert_data_to_temporal_data(self.data)
        self.train_data = self.convert_data_to_temporal_data(
            self.data.subgraph(train_idx)
        )
        self.val_data = self.convert_data_to_temporal_data(self.data.subgraph(val_idx))
        self.test_data = self.convert_data_to_temporal_data(
            self.data.subgraph(test_idx)
        )
        self.train_time = [self.train_data.t.min(), self.train_data.t.max()]
        self.val_time = [self.val_data.t.min(), self.val_data.t.max()]
        self.test_time = [self.test_data.t.min(), self.test_data.t.max()]

        self.train_mask, self.val_mask, self.test_mask = self.make_masks(
            self.full_data, self.train_time, self.val_time, self.test_time
        )

    @property
    def num_nodes(self) -> int:
        return self.data.num_nodes

    @property
    def num_features(self) -> int:
        return self.data.num_features

    @property
    def num_classes(self) -> int:
        return 40

    def convert_data_to_temporal_data(self, data: Data) -> TemporalData:
        # selects the maximum timestamp for each edge
        timestamps = torch.max(data.node_year[data.edge_index].squeeze(), dim=0)[0]

        from_edges = data.edge_index[0, :]
        to_edges = data.edge_index[1, :]

        assert len(from_edges) == len(to_edges) == len(timestamps)

        # Messages: from_year, to_year
        messages = data.node_year[data.edge_index].view(2, -1).T.float()

        return TemporalData(
            src=from_edges,
            dst=to_edges,
            t=timestamps,
            msg=messages,
        )

    def make_masks(
        self,
        data: TemporalData,
        train_time: TimeSpan,
        val_time: TimeSpan,
        test_time: TimeSpan,
    ):
        train_mask = data.t <= train_time[-1]
        val_mask = (data.t > train_time[-1]) & (data.t <= val_time[-1])
        test_mask = data.t >= test_time[0]
        assert (train_mask + val_mask + test_mask).sum() == data.t.shape[0]
        return train_mask, val_mask, test_mask
