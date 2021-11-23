import sys
sys.path.append('/mnt/storage01/szalata/autoencoders')

import argparse
import mlflow
import torch
import numpy as np
import umap
from utils import load_dataset, evaluate_solution


EXPERIMENT_NAME="joint_embeddings"


def main():
    parser = argparse.ArgumentParser()

    # Other parameters
    parser.add_argument("--dataset_path", default="output/datasets_phase2/joint_embedding/"
                                                  "openproblems_bmmc_multiome_phase2/"
                                                  "openproblems_bmmc_multiome_phase2.censor_dataset"
                                                  ".output_", type=str,
                        help="Path to the dataset.")
    parser.add_argument("--use_sample_data", action='store_true')
    parser.add_argument("--n_dim", type=int, default=100)
    parser.add_argument("--run_name", default=None, type=str, help="name of the mlflow run")

    args = parser.parse_args()
    if args.use_sample_data:
        args.dataset_path = "sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter."


    ad_mod1, ad_mod2, ad_solution = load_dataset(path=args.dataset_path)
    n_dim = args.n_dim

    embedder_mod1 = umap.UMAP(n_components=n_dim // 2)
    mod1_pca = embedder_mod1.fit_transform(ad_mod1.X)
    # 'Performing dimensionality reduction on modality 2 values...'
    embedder_mod1 = umap.UMAP(n_components=n_dim // 2)
    mod2_pca = embedder_mod1.fit_transform(ad_mod2.X)
    del ad_mod2
    del ad_mod1

    # 'Concatenating datasets'
    pca_combined = np.concatenate([mod1_pca, mod2_pca], axis=1)
    scores = evaluate_solution(ad_solution, pca_combined, args.run_name)
    print(scores)

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({
            "pytorch version": torch.__version__,
            "cuda version": torch.version.cuda,
            "device name": torch.cuda.get_device_name(0)
        })
        mlflow.log_params(vars(args))

        for key, value in scores.items():
            mlflow.log_metric(key, value)

if __name__ == "__main__":
    main()
