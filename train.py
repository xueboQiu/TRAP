import os
import random
import time

import psutil
import torch.nn.functional as F
import dgl
import torch
import warnings
from tqdm import tqdm

from utils.data_aug import augment_graph
from utils.loaddata import load_batch_level_dataset, load_entity_level_dataset, load_metadata
from model.autoencoder import build_model
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader, MultiLayerNeighborSampler
from model.train import batch_level_train
from utils.utils import set_random_seed, create_optimizer
from utils.config import build_args
warnings.filterwarnings('ignore')


def extract_dataloaders(entries, batch_size):
    random.shuffle(entries)
    train_idx = torch.arange(len(entries))
    train_sampler = SubsetRandomSampler(train_idx)
    train_loader = GraphDataLoader(entries, batch_size=batch_size, sampler=train_sampler)
    return train_loader


def sample_graphs(subgraph, new_seed_id, hop):

    neis = set([new_seed_id])
    for i in range(hop):
        if i == 0:
            tmp_neis = [new_seed_id]
        for nei in tmp_neis:

            cur_neis = set(subgraph.successors(nei).tolist() + subgraph.predecessors(nei).tolist())
            neis = neis.union(cur_neis)

            tmp_neis = cur_neis

    return torch.tensor(list(neis))

def main(main_args):
    device = f'cuda:{main_args.device}' if main_args.device >= 0 else "cpu"
    dataset_name = main_args.dataset
    if dataset_name == 'streamspot':
        main_args.num_hidden = 256
        main_args.max_epoch = 5
        main_args.num_layers = 4
        main_args.alpha = 0.8
    elif dataset_name == 'wget':
        main_args.num_hidden = 256
        main_args.max_epoch = 2
        main_args.num_layers = 4
        main_args.alpha = 0.7
    else:
        main_args.max_epoch = 50
        main_args.num_hidden = main_args.d
        main_args.num_layers = main_args.l
        main_args.alpha = 0.3

    set_random_seed(0)

    if dataset_name == 'streamspot' or dataset_name == 'wget':
        if dataset_name == 'streamspot':
            batch_size = 12
        else:
            batch_size = 1
        dataset = load_batch_level_dataset(dataset_name)
        n_node_feat = dataset['n_feat']
        n_edge_feat = dataset['e_feat']
        graphs = dataset['dataset']
        train_index = dataset['train_index']
        main_args.n_dim = n_node_feat
        main_args.e_dim = n_edge_feat
        model = build_model(main_args)
        model = model.to(device)
        optimizer = create_optimizer(main_args.optimizer, model, main_args.lr, main_args.weight_decay)
        model = batch_level_train(model, main_args, graphs, (extract_dataloaders(train_index, batch_size)),
                                  optimizer, main_args.max_epoch, device, main_args.n_dim, main_args.e_dim)
        torch.save(model.state_dict(), "./checkpoints/checkpoint-{}_fr{}_sr{}_gcl{}_d{}_l{}.pt".format(
            dataset_name,main_args.fr,main_args.sr,main_args.gcl, main_args.num_hidden,main_args.num_layers))
    else:
        metadata = load_metadata(dataset_name)
        main_args.n_dim = metadata['node_feature_dim']
        main_args.e_dim = metadata['edge_feature_dim']
        model = build_model(main_args)
        model = model.to(device)
        model.train()
        optimizer = create_optimizer(main_args.optimizer, model, main_args.lr, main_args.weight_decay)
        epoch_iter = tqdm(range(main_args.max_epoch))
        n_train = metadata['n_train']
        sampler = MultiLayerNeighborSampler([int(i) for i in args.fanout.split(',')])
        contrast_loss_fn = torch.nn.CrossEntropyLoss().to(device)

        start_time = time.time()

        for epoch in epoch_iter:
            epoch_loss = 0.0
            for i in range(n_train):
                model.train()

                g = load_entity_level_dataset(dataset_name, 'train', i).to(device)

                train_contrastive_loss = torch.tensor([0]).to(device)
                if args.gcl == 1:
                    train_dataloader_list = augment_graph(g, sampler, args, device)
                    seeds_emb = torch.tensor([]).to(device)
                    sgs_emb = torch.tensor([]).to(device)
                    for train_dataloader in train_dataloader_list:
                        for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):

                            # batch_inputs = node_feats[input_nodes].to(device)
                            block = blocks[0].to(device)
                            src, dst = block.edges()
                            nid_maps = {v.item():i for i, v in enumerate(block.srcdata[dgl.NID])}
                            subgraph = dgl.graph((src, dst), num_nodes=block.num_src_nodes())
                            subgraph.ndata['attr'] = block.srcdata['attr']
                            subgraph.edata['attr'] = block.edata['attr']

                            sg_emb = model.embed(subgraph)

                            new_seed_ids = torch.tensor([nid_maps[seed.item()] for seed in seeds], device=device)
                            seeds_embds = sg_emb[new_seed_ids]
                            seeds_emb = torch.cat([seeds_emb, seeds_embds.unsqueeze(0)], dim=0)

                            sg_nodes_list = [sample_graphs(subgraph, new_seed_id, hop=2) for new_seed_id in new_seed_ids]
                            all_sg_nodes = torch.cat(sg_nodes_list)
                            all_sg_embs = sg_emb[all_sg_nodes]
                            tmp_sg_embs = []
                            offset = 0
                            for sg_nodes in sg_nodes_list:
                                num_nodes = len(sg_nodes)
                                tmp_sg_emb = torch.mean(all_sg_embs[offset:offset + num_nodes], dim=0)
                                tmp_sg_embs.append(tmp_sg_emb)
                                offset += num_nodes
                            sgs_emb = torch.cat([sgs_emb, torch.stack(tmp_sg_embs).unsqueeze(0)], dim=0)

                    for idx in range(seeds_emb.shape[0]):
                        for idy in range(idx + 1, seeds_emb.shape[0]):

                            # # perform node-node contrast
                            # z1 = seeds_emb[idx]
                            # z2 = seeds_emb[(idx + 1) % seeds_emb.shape[0]]  # 正样本对

                            # pred1 = torch.mm(z1, z2.T).to(device)
                            # pred2 = torch.mm(z2, z1.T).to(device)

                            # labels = torch.arange(pred1.shape[0]).to(device)
                            # if args.ncl == 1 and args.gcl == 0:
                            #     train_contrastive_loss = train_contrastive_loss + (contrast_loss_fn(pred1, labels) + contrast_loss_fn(pred2, labels)) / 2
                            # elif args.ncl == 1:
                            #     train_contrastive_loss = train_contrastive_loss + (1-args.alpha) * (contrast_loss_fn(pred1, labels) + contrast_loss_fn(pred2, labels)) / 2
                            # train_contrastive_loss = train_contrastive_loss + (contrast_loss_fn(pred1 / args.temp, labels) + contrast_loss_fn(pred2 / args.temp, labels)) / 2

                            # perform subgraph-subgraph contrast
                            z3 = sgs_emb[idx]
                            z4 = sgs_emb[idy]
                            pred3 = torch.mm(z3, z4.T).to(device)
                            pred4 = torch.mm(z4, z3.T).to(device)
                            labels = torch.arange(pred3.shape[0]).to(device)
                            # if args.gcl == 1 and args.ncl == 0:
                            #     train_contrastive_loss = train_contrastive_loss + contrast_loss_fn(pred3, labels) + contrast_loss_fn(pred4, labels) / 2
                            # elif args.gcl == 1:
                                # train_contrastive_loss = train_contrastive_loss + args.alpha * (contrast_loss_fn(pred3, labels) + contrast_loss_fn(pred4, labels)) / 2

                            train_contrastive_loss = train_contrastive_loss + (contrast_loss_fn(pred3, labels) + contrast_loss_fn(pred4, labels)) / 2

                print("train_contrastive_loss: ", train_contrastive_loss)
                loss = model(g)
                loss /= n_train
                # total_loss = args.beta * loss + train_contrastive_loss
                if args.gcl == 1 and (args.fr == 1 or args.sr == 1):
                    total_loss = (1 - args.alpha) * loss + args.alpha * train_contrastive_loss
                elif args.gcl == 1:
                    total_loss = train_contrastive_loss
                elif args.gcl == 0:
                    total_loss = loss
                optimizer.zero_grad()
                epoch_loss += total_loss.item()
                total_loss.backward()
                optimizer.step()
                del g
            epoch_iter.set_description(f"Epoch {epoch} | train_loss: {epoch_loss:.4f}")
        torch.save(model.state_dict(), "./checkpoints/checkpoint-{}_fr{}_sr{}_gcl{}_d{}_l{}_v{}_s{}_n{}.pt".format(dataset_name,main_args.fr,main_args.sr,
                                                                                           main_args.gcl, main_args.num_hidden,main_args.num_layers,
                                                                                        main_args.views,main_args.strategy,main_args.nid_batch_size))
        end_time = time.time()

        # output trianing time in seconds
        print(f"Training time: {end_time - start_time} seconds")
        save_dict_path = './eval_result/distance_save_{}.pkl'.format(dataset_name)
        os.unlink(save_dict_path)

    return

if __name__ == '__main__':
    args = build_args()
    main(args)