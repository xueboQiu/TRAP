import argparse


def build_args():
    parser = argparse.ArgumentParser(description="MAGIC")
    parser.add_argument("--dataset", type=str, default="wget")
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument("--negative_slope", type=float, default=0.2,
                        help="the negative slope of leaky relu for GAT")
    parser.add_argument("--mask_rate", type=float, default=0.5)
    parser.add_argument("--alpha_l", type=float, default=3,
                        help="`pow`inddex for `sce` loss")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--loss_fn", type=str, default='sce')
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument('--snapshots', type=int, default=3)
    parser.add_argument('--views', type=int, default=3)
    parser.add_argument('--strategy', type=str, default='mid_overlap')
    parser.add_argument('--nid_batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument("--fanout", type=str, required=False,
                        help="fanout numbers", default='8,8')
    parser.add_argument("--temp", type=float, required=False, default=0.07)
    parser.add_argument("--alpha", type=float, required=False, default=0.3)
    parser.add_argument("--d", type=int, required=False, default=64)
    parser.add_argument("--l", type=int, required=False, default=3)
    parser.add_argument("--gcl", type=int, required=False, default=1)
    parser.add_argument("--fr", type=int, required=False, default=0)
    parser.add_argument("--sr", type=int, required=False, default=0)

    args = parser.parse_args()
    return args
