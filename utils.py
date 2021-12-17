import anndata as ad
import scanpy as sc
import torch
import random
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from evaluation.eval import evaluate
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def set_seed(seed):
    torch.random.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def scores_mean_std(all_scores):
    mean_scores = pd.DataFrame(all_scores).mean()
    mean_scores = pd.Series(mean_scores.values, index=mean_scores.index.map(lambda x: x + "_mean"))
    std_scores = pd.DataFrame(all_scores).std()
    std_scores = pd.Series(std_scores.values, index=std_scores.index.map(lambda x: x + "_std"))
    all_scores = pd.concat((mean_scores, std_scores), axis=0)
    return all_scores.to_dict()


def load_dataset(path='output/datasets_phase2/joint_embedding/openproblems_bmmc_multiome_phase2/openproblems_bmmc_multiome_phase2.censor_dataset.output_',
                 minmax_norm=False, std_norm=False):
    solution_path = path + "solution.h5ad"
    adata_solution = ad.read_h5ad(solution_path)
    ad_mod1 = ad.read_h5ad(path + 'mod1.h5ad')
    ad_mod2 = ad.read_h5ad(path + 'mod2.h5ad')
    if minmax_norm:
        ad_mod1.X = MinMaxScaler().fit_transform(ad_mod1.X.todense())
        # ad_mod2.X = csr_matrix(MinMaxScaler().fit_transform(ad_mod2.X.todense()))
    if std_norm:
        ad_mod1.X = StandardScaler().fit_transform(ad_mod1.X.todense())
        # ad_mod2.X = StandardScaler().fit_transform(ad_mod2.X)
    ad_mod1.X = csr_matrix(ad_mod1.X)
    # ad_mod2.X = csr_matrix(ad_mod2.X)

    ad_mod1.obs['cell_type'] = adata_solution.obs['cell_type'][ad_mod1.obs_names]
    return ad_mod1, ad_mod2, adata_solution

def evaluate_solution(ad_solution, embedding, run_name):
    # put into anndata
    adata = ad.AnnData(
        X=embedding,
        obs=ad_solution.obs
    )
    # Transfer obs annotations
    obs_names = adata.obs_names

    adata.obs['batch'] = ad_solution.obs['batch'][obs_names]
    adata.obs['cell_type'] = ad_solution.obs['cell_type'][obs_names]

    # Preprocessing
    adata.obsm['X_emb'] = adata.X
    sc.pp.neighbors(adata, use_rep="X")
    sc.tl.umap(adata)
    sc.pl.umap(adata, color='cell_type', save=f"/{run_name}_celltype_plot.png")
    sc.pl.umap(adata, color='batch', save=f"/{run_name}_batch_plot.png")
    return evaluate(ad_solution, adata)
