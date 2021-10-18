from statistics import mean

import anndata as ad
import numpy as np
import scanpy as sc
from scIB.metrics import silhouette_batch, graph_connectivity, nmi, silhouette, cell_cycle, trajectory_conservation
from scIB.clustering import opt_louvain

from sklearn.decomposition import TruncatedSVD


def evaluate(adata_solution, adata_joint, bio_metrics_weight=0.6, batch_metrics_weight=0.4,
             chosen_metrics=["nmi_ATAC", "asw_label_ATAC", "cc_cons_ATAC", "ti_cons_mean_ATAC", "cc_cons_ATAC", "ti_cons_mean_ATAC"]):
    scores = {}
    chosen_metrics = set(chosen_metrics)
    
    organism = adata_solution.uns['organism']
    obs_keys = adata_solution.obs_keys()
    
    recompute_cc = 'S_score' not in adata_solution.obs_keys() or \
               'G2M_score' not in adata_solution.obs_keys()
    adt_atac_trajectory = 'pseudotime_order_ATAC' if 'pseudotime_order_ATAC' in adata_solution.obs else 'pseudotime_order_ADT'
    
    # preprocessing for graph connectivity
    sc.pp.neighbors(adata_joint, use_rep='X_emb')
    # clustering for nmi
    opt_louvain(
        adata_joint,
        label_key='cell_type',
        cluster_key='cluster',
        plot=False,
        inplace=True,
        force=True
    )

    # Compute score
    scores["asw_batch_ATAC"] = silhouette_batch(
        adata_joint,
        batch_key='batch',
        group_key='cell_type',
        embed='X_emb',
        verbose=False
    )
    
    scores["graph_conn_ATAC"] = graph_connectivity(adata_joint, label_key='cell_type')
    scores["nmi_ATAC"] = nmi(adata_joint, group1='cluster', group2='cell_type')
    scores["asw_label_ATAC"] = silhouette(adata_joint, group_key='cell_type', embed='X_emb')
    scores["cc_cons_ATAC"] = cell_cycle(
        adata_pre=adata_solution,
        adata_post=adata_joint,
        batch_key='batch',
        embed='X_emb',
        recompute_cc=recompute_cc,
        organism=organism
    )
    if 'pseudotime_order_GEX' in obs_keys:
        score_rna = trajectory_conservation(
            adata_pre=adata_solution,
            adata_post=adata_joint,
            label_key='cell_type',
            pseudotime_key='pseudotime_order_GEX'
        )
    else:
        score_rna = np.nan

    if adt_atac_trajectory in obs_keys:
        score_adt_atac = trajectory_conservation(
            adata_pre=adata_solution,
            adata_post=adata_joint,
            label_key='cell_type',
            pseudotime_key=adt_atac_trajectory
        )
    else:
        score_adt_atac = np.nan

    scores["ti_cons_mean_ATAC"] = (score_rna + score_adt_atac) / 2
    
    scores["overall"] = bio_metrics_weight * mean([scores[bio_metric] for bio_metric in ["nmi_ATAC", "asw_label_ATAC", "cc_cons_ATAC", "ti_cons_mean_ATAC"] if bio_metric in chosen_metrics]) + \
        batch_metrics_weight * mean([scores[batch_metric] for batch_metric in ["cc_cons_ATAC", "ti_cons_mean_ATAC"] if batch_metric in chosen_metrics])
    return scores
