import sys
sys.path.append('/mnt/storage01/szalata/autoencoders')

import argparse
import mlflow
import torch
import scvi
import anndata as ad
from utils import load_dataset, evaluate_solution


EXPERIMENT_NAME="joint_embeddings"


def main():
    parser = argparse.ArgumentParser()

    # Other parameters
    parser.add_argument("--dataset_path", default="output/datasets/joint_embedding/"
                                                  "openproblems_bmmc_multiome_phase1/"
                                                  "openproblems_bmmc_multiome_phase1.censor_dataset"
                                                  ".output_", type=str,
                        help="Path to the dataset.")
    parser.add_argument("--use_sample_data", action='store_true')
    parser.add_argument("--n_dim", type=int, default=100)
    parser.add_argument("--run_name", default=None, type=str, help="name of the mlflow run")

    args = parser.parse_args()
    if args.use_sample_data:
        args.dataset_path = "sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter."


    ad_mod1, ad_mod2, ad_solution = load_dataset(path=args.dataset_path)
    ad_mod12 = ad.concat((ad_mod1, ad_mod2), axis=1)
    ad_mod12.obs["batch_id"] = 1
    del ad_mod1
    del ad_mod2
    organized_anndata = scvi.data.organize_multiome_anndatas(ad_mod12, modality_key="feature_types")
    del ad_mod12
    scvi.model.MULTIVI.setup_anndata(organized_anndata, batch_key="feature_types")
    n_genes = (organized_anndata.var.feature_types == "GEX").sum()
    vae = scvi.model.MULTIVI(organized_anndata, n_genes=n_genes,
                             n_regions=organized_anndata.shape[1] - n_genes, n_latent=args.n_dim)
    vae.train(batch_size=1280, max_epochs=500)
    latent_vector = vae.get_latent_representation()
    scores = evaluate_solution(ad_solution, latent_vector)
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
