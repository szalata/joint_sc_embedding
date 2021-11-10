import anndata as ad
import scanpy as sc

from evaluation.eval import evaluate


def load_dataset(path='output/datasets/joint_embedding/openproblems_bmmc_multiome_phase1/openproblems_bmmc_multiome_phase1.censor_dataset.output_'):
    solution_path = path + "solution.h5ad"
    adata_solution = ad.read_h5ad(solution_path)
    ad_mod1 = ad.read_h5ad(path + 'mod1.h5ad')
    ad_mod2 = ad.read_h5ad(path + 'mod2.h5ad')
    
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
    adata.write(f"output/embeddings/{run_name}.h5ad")
    sc.pp.neighbors(adata, use_rep="X")
    sc.tl.umap(adata)
    sc.pl.umap(adata, color='cell_type', save=f"output/embeddings/{run_name}_celltype_plot.png")
    sc.pl.umap(adata, color='batch', save=f"output/embeddings/{run_name}_batch_plot.png")
    return evaluate(ad_solution, adata)
