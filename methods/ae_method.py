import sys

import os
from pathlib import Path
import pytorch_lightning as pl

import json
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

sys.path.append('/mnt/storage01/szalata/autoencoders')
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import argparse
import mlflow
import torch
from utils import evaluate_solution, set_seed, scores_mean_std
from autoencoder.ae_model import AEModel
from dataloaders.multiome_datamodule import MultiomeDataModule

EXPERIMENT_NAME="joint_embeddings"


def method_evaluate(args, seed, dm, solution_ad):
    set_seed(seed)
    args.run_name = f"{args.mlf_run_name}_s{seed}"
    mlf_logger = MLFlowLogger(experiment_name=EXPERIMENT_NAME, run_name=args.run_name)
    mlf_logger.log_hyperparams(vars(args))
    model = AEModel(args, solution_ad)
    if args.save_model:
        checkpoint_path = os.path.join(args.output_dir, args.mlf_experiment_name, args.run_name)
        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(monitor="best_mse_val", dirpath=checkpoint_path)

        with open(os.path.join(checkpoint_path, "args.json"), "w") as fp:
            json.dump(vars(args), fp)

    trainer = pl.Trainer(logger=mlf_logger, max_epochs=args.epochs, gpus=torch.cuda.device_count(),
                         callbacks=[checkpoint_callback] if args.save_model else None,
                         check_val_every_n_epoch=args.eval_epochs, num_sanity_val_steps=0)
    trainer.fit(model, dm)
    latent_vector = trainer.predict(model, datamodule=dm)
    latent_vector = torch.cat(latent_vector).cpu().numpy()
    return evaluate_solution(solution_ad, latent_vector, args.run_name)


def main():
    parser = argparse.ArgumentParser()

    # datamodule params
    parser.add_argument("--dataset_path", default="output/datasets_phase2/joint_embedding/"
                                                  "openproblems_bmmc_multiome_phase2/"
                                                  "openproblems_bmmc_multiome_phase2.censor_dataset"
                                                  ".output_", type=str,
                        help="Path to the dataset.")
    parser.add_argument("--output_dir", default="output/models")
    parser.add_argument("--use_sample_data", action='store_true')
    parser.add_argument("--use_raw_counts", action='store_true')

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--minmax_norm", action="store_true")
    parser.add_argument("--std_norm", action="store_true")

    # model params
    parser.add_argument("--model", default="ae_model", choices=["ae_model"])
    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--batchnorm", action="store_true")
    parser.add_argument("--dropout_p", type=float, default=0)
    parser.add_argument("--layer_size_multiplier", type=float, default=2)
    parser.add_argument("--n_hidden", type=int, default=2)

    # training params
    parser.add_argument("--mlf_run_name", type=str, default="test_run")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--eval_epochs", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--init_seed", type=int, default=42)
    parser.add_argument("--save_model", action="store_true")

    args = parser.parse_args()

    if args.use_sample_data:
        args.dataset_path = "sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter."
    set_seed(args.init_seed)

    dm = MultiomeDataModule(args.dataset_path, args.batch_size, args.minmax_norm, args.std_norm)
    _, _, solution_ad = dm.dataset
    args.expression_dim = dm.dataset_params["expression"].shape[1]
    args.batch_input_dim = dm.dataset_params["batch"].shape[1]

    all_scores = []
    for seed in range(args.seeds):
        all_scores.append(method_evaluate(args, seed, dm, solution_ad))
    all_scores = scores_mean_std(all_scores)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=args.mlf_run_name):
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
