import sys
sys.path.append('/mnt/storage01/szalata/autoencoders')

import argparse
import mlflow
import torch
import numpy as np
from sklearn.decomposition import TruncatedSVD
from utils import load_dataset, evaluate_solution, set_seed, scores_mean_std

EXPERIMENT_NAME="joint_embeddings"


def method_evaluate(args, seed, dataset):
    ad_mod1, ad_mod2, ad_solution = dataset
    n_dim = args.n_dim
    gex_dim = args.gex_dim
    embedder_mod1 = TruncatedSVD(n_components=n_dim if args.gex_only else gex_dim, random_state=seed)
    mod1_pca = embedder_mod1.fit_transform(ad_mod1.X)
    # 'Performing dimensionality reduction on modality 2 values...'
    embedder_mod1 = TruncatedSVD(n_components=n_dim - gex_dim, random_state=seed)
    mod2_pca = embedder_mod1.fit_transform(ad_mod2.X)

    # 'Concatenating datasets'
    if args.gex_only:
        pca_combined = mod1_pca
    else:
        pca_combined = np.concatenate([mod1_pca, mod2_pca], axis=1)
    scores = evaluate_solution(ad_solution, pca_combined, args.run_name)
    return scores


def main():
    parser = argparse.ArgumentParser()

    # Other parameters
    parser.add_argument("--dataset_path", default="output/datasets_phase2/joint_embedding/"
                                                  "openproblems_bmmc_multiome_phase2/"
                                                  "openproblems_bmmc_multiome_phase2.censor_dataset"
                                                  ".output_", type=str,
                        help="Path to the dataset.")
    parser.add_argument("--use_sample_data", action='store_true')
    parser.add_argument("--gex_only", action='store_true')
    parser.add_argument("--n_dim", type=int, default=100)
    parser.add_argument("--gex_dim", type=int, default=50)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--init_seed", type=int, default=42)
    parser.add_argument("--run_name", default=None, type=str, help="name of the mlflow run")
    parser.add_argument("--minmax_norm", action="store_true")
    parser.add_argument("--std_norm", action="store_true")

    args = parser.parse_args()
    if args.use_sample_data:
        args.dataset_path = "sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter."

    set_seed(args.init_seed)
    dataset = load_dataset(args.dataset_path, args.minmax_norm, args.std_norm)
    all_scores = []
    for seed in range(args.seeds):
        all_scores.append(method_evaluate(args, seed, dataset))
    all_scores = scores_mean_std(all_scores)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({
            "pytorch version": torch.__version__,
            "cuda version": torch.version.cuda,
            "device name": torch.cuda.get_device_name(0)
        })
        mlflow.log_params(vars(args))

        for key, value in all_scores.items():
            mlflow.log_metric(key, value)

if __name__ == "__main__":
    main()
