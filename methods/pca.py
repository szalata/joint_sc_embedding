import sys
sys.path.append('/mnt/storage01/szalata/autoencoders')

import numpy as np
from sklearn.decomposition import TruncatedSVD
from utils import load_dataset, evaluate_solution

ad_mod1, ad_mod2, ad_solution = load_dataset(path='sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.')
# TODO: implement your own method
n_dim = 50
embedder_mod1 = TruncatedSVD(n_components=n_dim//2)
mod1_pca = embedder_mod1.fit_transform(ad_mod1.X)
# 'Performing dimensionality reduction on modality 2 values...'
embedder_mod1 = TruncatedSVD(n_components=n_dim//2)
mod2_pca = embedder_mod1.fit_transform(ad_mod2.X)
del ad_mod2

# 'Concatenating datasets'
pca_combined = np.concatenate([mod1_pca, mod2_pca], axis=1)
scores = evaluate_solution(ad_mod1, ad_solution, pca_combined)
print(scores)

mlflow.set_experiment(args.mlflow.experiment_name)
with mlflow.start_run(run_name=args.mlflow.run_name):
    mlflow.log_params({
        "pytorch version": torch.__version__,
        "cuda version": torch.version.cuda,
        "device name": torch.cuda.get_device_name(0)
    })
    mlflow.log_params(vars(args))