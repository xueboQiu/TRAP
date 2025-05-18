import os.path
import time

import psutil
import torch
import warnings
from utils.loaddata import load_batch_level_dataset, load_entity_level_dataset, load_metadata
from model.autoencoder import build_model
from utils.poolers import Pooling
from utils.utils import set_random_seed
import numpy as np
from model.eval import batch_level_evaluation, evaluate_entity_level_using_knn
from utils.config import build_args
warnings.filterwarnings('ignore')


def main(main_args):
    device = main_args.device if main_args.device >= 0 else "cpu"
    device = torch.device(device)
    set_random_seed(0)
    dataset_name = main_args.dataset
    if dataset_name in ['streamspot', 'wget']:
        main_args.num_hidden = 256
        main_args.num_layers = 4
    else:
        main_args.num_hidden = main_args.d
        main_args.num_layers = main_args.l

    if dataset_name == 'streamspot' or dataset_name == 'wget':
        dataset = load_batch_level_dataset(dataset_name)
        n_node_feat = dataset['n_feat']
        n_edge_feat = dataset['e_feat']
        main_args.n_dim = n_node_feat
        main_args.e_dim = n_edge_feat
        model = build_model(main_args)
        ck_path = "./checkpoints/checkpoint-{}_fr{}_sr{}_gcl{}_d{}_l{}.pt".format(
            dataset_name,main_args.fr,main_args.sr,main_args.gcl, main_args.num_hidden,main_args.num_layers)
        if os.path.exists(ck_path):
            print(f"load checkpoint file from {ck_path}.")
            model.load_state_dict(torch.load(ck_path, map_location=device))
        else:
            print(f"not checkpoint file found.")
            # exit(-1)

        model = model.to(device)
        pooler = Pooling(main_args.pooling)
        test_auc, test_std = batch_level_evaluation(model, pooler, device, ['knn'], args.dataset, main_args.n_dim,
                                                    main_args.e_dim)
    else:
        metadata = load_metadata(dataset_name)
        main_args.n_dim = metadata['node_feature_dim']
        main_args.e_dim = metadata['edge_feature_dim']

        print("hidden dim: ", main_args.num_hidden)
        print("num layers: ", main_args.num_layers)
        print("gcl: ", main_args.gcl)
        print("sr: ", main_args.sr)
        print("fr: ", main_args.fr)

        model = build_model(main_args)
        ck_path = "./checkpoints/checkpoint-{}_fr{}_sr{}_gcl{}_d{}_l{}_v{}_s{}_n{}.pt".format(dataset_name,main_args.fr,main_args.sr,
                                                                                           main_args.gcl, main_args.num_hidden,main_args.num_layers,
                                                                                        main_args.views,main_args.strategy,main_args.nid_batch_size)
        model.load_state_dict(torch.load(ck_path, map_location=device))
        print("Model loaded from: ", ck_path)
        model = model.to(device)
        model.eval()
        malicious, _ = metadata['malicious']
        n_train = metadata['n_train']
        n_test = metadata['n_test']

        start_time = time.time()

        with torch.no_grad():
            x_train = []
            for i in range(n_train):
                g = load_entity_level_dataset(dataset_name, 'train', i).to(device)
                x_train.append(model.embed(g).cpu().numpy())
                del g
            x_train = np.concatenate(x_train, axis=0)
            skip_benign = 0
            x_test = []
            for i in range(n_test):
                g = load_entity_level_dataset(dataset_name, 'test', i).to(device)
                # Exclude training samples from the test set
                if i != n_test - 1:
                    skip_benign += g.number_of_nodes()
                x_test.append(model.embed(g).cpu().numpy())
                del g
            x_test = np.concatenate(x_test, axis=0)

            n = x_test.shape[0]
            y_test = np.zeros(n)
            y_test[malicious] = 1.0
            malicious_dict = {}
            for i, m in enumerate(malicious):
                malicious_dict[m] = i

            # Exclude training samples from the test set
            test_idx = []
            for i in range(x_test.shape[0]):
                if i >= skip_benign or y_test[i] == 1.0:
                    test_idx.append(i)
            result_x_test = x_test[test_idx]
            result_y_test = y_test[test_idx]
            del x_test, y_test
            test_auc, test_std, _, _ = evaluate_entity_level_using_knn(dataset_name, x_train, result_x_test,
                                                                       result_y_test)
    print(f"#Test_AUC: {test_auc:.4f}±{test_std:.4f}")
    end_time = time.time()

    print(f"Testing time: {end_time - start_time} seconds")
    return

if __name__ == '__main__':
    args = build_args()
    main(args)
