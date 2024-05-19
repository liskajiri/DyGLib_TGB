import torch.nn as nn
from models.modules import MLPClassifier, MergeLayer

from models.CAWN import CAWN
from models.DyGFormer import DyGFormer
from models.GraphMixer import GraphMixer
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.TCL import TCL
from models.TGAT import TGAT


class TemporalNodeClassifier(nn.Module):
    def __init__(self, dynamic_backbone, args, num_classes: int):
        super(TemporalNodeClassifier, self).__init__()
        self.dynamic_backbone = dynamic_backbone
        self.node_classifier = MLPClassifier(
            input_dim=args.output_dim, output_dim=num_classes, dropout=args.dropout
        )

        self.model = nn.Sequential(self.dynamic_backbone, self.node_classifier)

    def forward(self, x):
        return self.model(x)


class TemporalLinkModule(nn.Module):
    def __init__(self, dynamic_backbone, args):
        super(TemporalLinkModule, self).__init__()
        self.dynamic_backbone = dynamic_backbone
        self.link_predictor = MergeLayer(
            input_dim1=args.output_dim,
            input_dim2=args.output_dim,
            hidden_dim=args.output_dim,
            output_dim=1,
        )
        self.model = nn.Sequential(self.dynamic_backbone, self.link_predictor)

    def forward(self, x):
        return self.model(x)


def create_dynamic_model(
    args, train_data, node_raw_features, edge_raw_features, train_neighbor_sampler
) -> TGAT | MemoryModel | CAWN | TCL | GraphMixer | DyGFormer:
    # create model
    if args.model_name == "TGAT":
        dynamic_backbone = TGAT(
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            neighbor_sampler=train_neighbor_sampler,
            time_feat_dim=args.time_feat_dim,
            output_dim=args.output_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            device=args.device,
        )
    elif args.model_name in ["JODIE", "DyRep", "TGN"]:
        # four floats that represent the mean and standard deviation of source and destination node time shifts in the training data, which is used for JODIE
        (
            src_node_mean_time_shift,
            src_node_std_time_shift,
            dst_node_mean_time_shift_dst,
            dst_node_std_time_shift,
        ) = compute_src_dst_node_time_shifts(
            train_data.src_node_ids,
            train_data.dst_node_ids,
            train_data.node_interact_times,
        )
        dynamic_backbone = MemoryModel(
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            neighbor_sampler=train_neighbor_sampler,
            time_feat_dim=args.time_feat_dim,
            output_dim=args.output_dim,
            model_name=args.model_name,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            src_node_mean_time_shift=src_node_mean_time_shift,
            src_node_std_time_shift=src_node_std_time_shift,
            dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst,
            dst_node_std_time_shift=dst_node_std_time_shift,
            device=args.device,
        )
    elif args.model_name == "CAWN":
        dynamic_backbone = CAWN(
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            neighbor_sampler=train_neighbor_sampler,
            time_feat_dim=args.time_feat_dim,
            position_feat_dim=args.position_feat_dim,
            output_dim=args.output_dim,
            walk_length=args.walk_length,
            num_walk_heads=args.num_walk_heads,
            dropout=args.dropout,
            device=args.device,
        )
    elif args.model_name == "TCL":
        dynamic_backbone = TCL(
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            neighbor_sampler=train_neighbor_sampler,
            time_feat_dim=args.time_feat_dim,
            output_dim=args.output_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            num_depths=args.num_neighbors + 1,
            dropout=args.dropout,
            device=args.device,
        )
    elif args.model_name == "GraphMixer":
        dynamic_backbone = GraphMixer(
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            neighbor_sampler=train_neighbor_sampler,
            time_feat_dim=args.time_feat_dim,
            output_dim=args.output_dim,
            num_tokens=args.num_neighbors,
            num_layers=args.num_layers,
            dropout=args.dropout,
            device=args.device,
        )
    elif args.model_name == "DyGFormer":
        dynamic_backbone = DyGFormer(
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            neighbor_sampler=train_neighbor_sampler,
            time_feat_dim=args.time_feat_dim,
            channel_embedding_dim=args.channel_embedding_dim,
            output_dim=args.output_dim,
            patch_size=args.patch_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            max_input_sequence_length=args.max_input_sequence_length,
            device=args.device,
        )
    else:
        raise ValueError(f"Wrong value for model_name {args.model_name}!")
    return dynamic_backbone
