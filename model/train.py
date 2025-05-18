import dgl
import numpy as np
import torch.nn as nn
import torch
from tqdm import tqdm
from utils.loaddata import transform_graph
import torch.nn.functional as F

from utils.utils import dgl_graph_copy

def batch_level_train(model, args, graphs, train_loader, optimizer, max_epoch, device, n_dim=0, e_dim=0):
    epoch_iter = tqdm(range(max_epoch))
    sampler = dgl.dataloading.MultiLayerNeighborSampler([8, 8])

    for epoch in epoch_iter:
        model.train()
        loss_list = []
        train_contrastive_loss = torch.tensor([0]).to(device)
        for _, batch in enumerate(train_loader):
            args.nid_batch_size = 128
            g = [transform_graph(graphs[idx][0], n_dim, e_dim).to(device) for idx in batch]
            g = dgl.batch(g)
            model.train()
            if args.gcl == 1:

                dataloader = dgl.dataloading.DataLoader(
                    g,
                    torch.arange(g.num_nodes()).to(g.device),
                    sampler,
                    batch_size=args.nid_batch_size,
                    shuffle=True,
                    drop_last=False,
                    num_workers=0,
                )
                graphs_v1 = []
                # 使用示例
                for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                    # blocks是采样得到的子图列表, blocks[-1]是最后一层的子图
                    block = blocks[0].to(args.device)  # 2跳邻居子图
                    src, dst = block.edges()
                    subgraph = dgl.graph((src, dst), num_nodes=block.num_src_nodes())
                    subgraph.ndata['attr'] = block.srcdata['attr']
                    subgraph.edata['attr'] = block.edata['attr']
                    graphs_v1.append(subgraph)

                    if len(graphs_v1) == args.nid_batch_size: break

                gm_augmentations = [
                    dgl.transforms.FeatMask(0.3, ['attr']),
                    dgl.transforms.DropEdge(0.1)
                ]
                aug_type = np.random.choice(len(gm_augmentations),
                                            args.nid_batch_size,
                                            replace=True)
                aug_type = {k: aug_type[k] for k in range(args.nid_batch_size)}

                graphs_v2 = [dgl_graph_copy(g) for g in graphs_v1]
                for i, g in enumerate(graphs_v2): graphs_v2[i] = gm_augmentations[aug_type[i]](g)

                # 批处理所有图
                bg1 = dgl.batch(graphs_v1).to(args.device)
                bg2 = dgl.batch(graphs_v2).to(args.device)
                # 提取嵌入
                bg1.ndata['h'] = model.embed(bg1)
                bg2.ndata['h'] = model.embed(bg2)
                # 使用均值池化
                z1 = dgl.mean_nodes(bg1, 'h')
                z2 = dgl.mean_nodes(bg2, 'h')
                # 对特征进行L2归一化
                z1_norm = F.normalize(z1, dim=1)
                z2_norm = F.normalize(z2, dim=1)

                # 计算相似度矩阵 [B, B]
                logits_12 = torch.mm(z1_norm, z2_norm.t())
                logits_21 = torch.mm(z2_norm, z1_norm.t())
                # 创建标签：对角线位置为正样本
                labels = torch.arange(z1_norm.shape[0], device=z1_norm.device)
                # 使用CrossEntropyLoss
                train_contrastive_loss = (nn.CrossEntropyLoss()(logits_12, labels) + nn.CrossEntropyLoss()(logits_21, labels)) / 2

                # print(f"train_contrastive_loss: {train_contrastive_loss.item()}")
            loss = model(g)

            if args.gcl == 1 and (args.fr == 1 or args.sr == 1):
                total_loss = (1 - args.alpha) * loss + args.alpha * train_contrastive_loss
            elif args.gcl == 1:
                total_loss = train_contrastive_loss
            elif args.gcl == 0:
                total_loss = loss

            # print(f"total_loss:{total_loss}")
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            loss_list.append(total_loss.item())
            del g
        epoch_iter.set_description(f"Epoch {epoch} | train_loss: {np.mean(loss_list):.4f}")
    return model