import random

import dgl
import torch
from functools import reduce
from dgl.dataloading import DataLoader, MultiLayerNeighborSampler


def sampling_layer(snapshots, views, span, strategy):
    T = []
    if strategy == 'random':
        T = [random.uniform(0, span * (snapshots - 1) / snapshots) for _ in range(views)]
    elif strategy == 'low_overlap':
        if (0.75 * views + 0.25) > snapshots:
            return "The number of sampled views exceeds the maximum value of the current policy."
        start = random.uniform(0, span - (0.75 * views + 0.25) * span /  snapshots)
        T = [start + (0.75 * i * span) / snapshots for i in range(views)]
    elif strategy == 'high_overlap':
        if (0.25 * views + 0.75) > snapshots:
            return "The number of sampled views exceeds the maximum value of the current policy."
        start = random.uniform(0, span - (0.25 * views + 0.75) * span /  snapshots)
        T = [start + (0.25 * i * span) / snapshots for i in range(views)]
    elif strategy == 'mid_overlap':
        if (0.5 * views + 0.5) > snapshots:
            return "The number of sampled views exceeds the maximum value of the current policy."
        start = random.uniform(0, span - (0.5 * views + 0.5) * span /  snapshots)
        T = [start + (0.5 * i * span) / snapshots for i in range(views)]
    elif strategy == 'sequential':
        T = [span * i / snapshots for i in range(snapshots)]
        if views > snapshots:
            return "The number of sampled views exceeds the maximum value of the current policy."
        T = random.sample(T, views)

    return T
def augment_graph(graph, sampler, args, device):

    edges_time = graph.edata['edge_time'].tolist()
    max_time, min_time, span = max(edges_time), min(edges_time), max(edges_time) - min(edges_time)
    temporal_subgraphs, nids, train_dataloader_list = [], [], []
    T = sampling_layer(args.snapshots, args.views, span, args.strategy)
    # add the start time
    T = [t + min_time for t in T]
    for start in T:
        end = min(start + span / args.snapshots, max_time)
        start = max(start, min_time)
        sample_time = (graph.edata['edge_time'] >= start) & (graph.edata['edge_time'] <= end)

        temporal_subgraph = dgl.edge_subgraph(graph, sample_time, relabel_nodes=False)

        # temporal_subgraph = dgl.to_simple(temporal_subgraph)
        # temporal_subgraph = dgl.to_bidirected(temporal_subgraph, copy_ndata=True)
        # temporal_subgraph = dgl.add_self_loop(temporal_subgraph)
        nids.append(torch.unique(temporal_subgraph.edges()[0]))
        temporal_subgraphs.append(temporal_subgraph)

    train_nid_per_gpu = list(reduce(lambda x, y: x & y, [set(nids[sg_id].tolist()) for sg_id in range(args.views)]))
    train_nid_per_gpu = random.sample(train_nid_per_gpu, args.nid_batch_size)
    random.shuffle(train_nid_per_gpu)
    train_nid_per_gpu = torch.tensor(train_nid_per_gpu).to(device)

    for sg_id in range(args.views):
        train_dataloader = DataLoader(temporal_subgraphs[sg_id],
                                          train_nid_per_gpu,
                                          sampler,
                                          batch_size=train_nid_per_gpu.shape[0],
                                          shuffle=False,
                                          drop_last=False,
                                          num_workers=0,
                                          )
        train_dataloader_list.append(train_dataloader)
    return train_dataloader_list