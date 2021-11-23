import sys

from scipy.sparse import csr_matrix

sys.path.append('/mnt/storage01/szalata/autoencoders')

import argparse
import mlflow
import torch
import scvi
import anndata as ad
from utils import load_dataset, evaluate_solution, set_seed, scores_mean_std

EXPERIMENT_NAME="joint_embeddings"


def method_evaluate(args, seed, dataset):
    set_seed(seed)
    ad_mod1, ad_mod2, ad_solution = dataset
    ad_mod12 = ad.concat((ad_mod1, ad_mod2), axis=1)
    if not args.use_normalized_counts:
        ad_mod12.X = ad_mod12.layers["counts"]
        ad_mod12.X = csr_matrix(ad_mod12.X)
    ad_mod12.obs["batch_id"] = 1
    ad_mod12.obs["site_donor"] = ad_mod2.obs.batch
    organized_anndata = scvi.data.organize_multiome_anndatas(ad_mod12, modality_key="feature_types")
    del ad_mod12
    scvi.model.MULTIVI.setup_anndata(organized_anndata, batch_key="feature_types",
                                     categorical_covariate_keys=["site_donor"])
    n_genes = (organized_anndata.var.feature_types == "GEX").sum()
    vae = scvi.model.MULTIVI(organized_anndata, n_genes=n_genes,
                             n_regions=organized_anndata.shape[1] - n_genes, n_latent=args.n_dim)
    vae.train(batch_size=1280, max_epochs=args.epochs, early_stopping=True, check_val_every_n_epoch=10,
              early_stopping_min_delta=0.01, early_stopping_patience=10)
    latent_vector = vae.get_latent_representation()
    return evaluate_solution(ad_solution, latent_vector, args.run_name)


def main():
    parser = argparse.ArgumentParser()

    # Other parameters
    parser.add_argument("--dataset_path", default="output/datasets_phase2/joint_embedding/"
                                                  "openproblems_bmmc_multiome_phase2/"
                                                  "openproblems_bmmc_multiome_phase2.censor_dataset"
                                                  ".output_", type=str,
                        help="Path to the dataset.")
    parser.add_argument("--use_sample_data", action='store_true')
    parser.add_argument("--use_normalized_counts", action='store_true')
    parser.add_argument("--n_dim", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--init_seed", type=int, default=42)
    parser.add_argument("--run_name", default=None, type=str, help="name of the mlflow run")

    args = parser.parse_args()
    if args.use_sample_data:
        args.dataset_path = "sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter."

    set_seed(args.init_seed)
    dataset = load_dataset(path=args.dataset_path)
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
