import anndata as ad

from evaluation.eval import evaluate


def load_dataset(path='output/datasets/joint_embedding/openproblems_bmmc_multiome_phase1/openproblems_bmmc_multiome_phase1.censor_dataset.output_'):
    solution_path = path + "solution.h5ad"
    adata_solution = ad.read_h5ad(solution_path)
    ad_mod1 = ad.read_h5ad(path + 'mod1.h5ad')
    ad_mod2 = ad.read_h5ad(path + 'mod2.h5ad')
    
    ad_mod1.obs['cell_type'] = adata_solution.obs['cell_type'][ad_mod1.obs_names]
    return ad_mod1, ad_mod2, adata_solution

def evaluate_solution(ad_mod1, ad_solution, embedding):
    # put into anndata
    adata = ad.AnnData(
        X=embedding,
        obs=ad_mod1.obs,
        uns={
            'dataset_id': ad_mod1.uns['dataset_id']
        },
    )
    # Transfer obs annotations
    adata.obs['batch'] = ad_solution.obs['batch'][adata.obs_names]
    adata.obs['cell_type'] = ad_solution.obs['cell_type'][adata.obs_names]

    # Preprocessing
    adata.obsm['X_emb'] = adata.X
    del ad_mod1
    return evaluate(ad_solution, adata)
