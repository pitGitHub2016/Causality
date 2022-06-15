import sys,os,glob,time
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.io
from scipy.io import savemat
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import manifold
import numpy as np
import datafold.dynfold as dfold
import datafold.pcfold as pfold
from datafold.dynfold import (
    GeometricHarmonicsInterpolator as GHI, LaplacianPyramidsInterpolator as LPI, TSCRadialBasis,
    LocalRegressionSelection, TSCTakensEmbedding
)
from datafold.dynfold import LocalRegressionSelection

"/////////////////////////////////////////////////////////////////////////////////////////////////////////////"
def Embed(method, X_train_local, **kwargs):
    """
    Function to embed input data using a specific embedding method

    :param method: DM, LLE or PCA
    :param X_train_local: X to embed
    :param kwargs: LLE_neighbors, dm_epsilon : either a specific value, or if zero it is internally optimized
    :return [target_mapping, parsimoniousEigs, X_pcm.kernel.epsilon, eigValsOut]:
    target_mapping --> the mapped data
    parsimoniousEigs (str) --> the indexes of the parsimonious coordinates
    X_pcm.kernel.epsilon --> optimized epsilon value
    eigValsOut --> corresponding eigenvalues
    """

    if "DM" in method:
        X_pcm = pfold.PCManifold(X_train_local)

        cut_off = np.inf

        n_eigenpairsIn = 10

        dmap_local = dfold.DiffusionMaps(
            kernel=pfold.GaussianKernel(epsilon=dm_epsilon),
            n_eigenpairs=n_eigenpairsIn,
            dist_kwargs=dict(cut_off=cut_off))
        dmap_local = dmap_local.fit(X_pcm)
        # evecs_raw, evals_raw = dmap.eigenvectors_, dmap.eigenvalues_

        if "ComputeParsimonious" in method:
            n_subsampleIn = 500

            selection = LocalRegressionSelection(
                intrinsic_dim=target_intrinsic_dim, n_subsample=n_subsampleIn, strategy="dim"
            ).fit(dmap_local.eigenvectors_)

            target_mapping = selection.transform(dmap_local.eigenvectors_)

        else:
            target_mapping = dmap_local.eigenvectors_[:,1:target_intrinsic_dim+1]

    elif "LLE" in method:

        lle = manifold.LocallyLinearEmbedding(n_neighbors=50, n_components=target_intrinsic_dim,
                                              method="standard", n_jobs=-1)
        target_mapping = lle.fit_transform(X_train_local)

    elif "PCA" in method:
        pca = PCA(n_components=target_intrinsic_dim)
        evecs = pca.fit_transform(X_train_local.T)
        evals = pca.singular_values_
        explainedVarianceRatio = pca.explained_variance_ratio_
        target_mapping = pca.components_.T

    return target_mapping
"/////////////////////////////////////////////////////////////////////////////////////////////////////////////"

plotData = 1
random_state = 0
target_intrinsic_dim = 1
embedMethod = "DMComputeParsimonious"#"LLE", "DMComputeParsimonious", "DM"
trainLen = 4500 # 4500

for f in tqdm(glob.glob("Datasets/*.mat")):
    try:
        print(f)
        original_space_data = scipy.io.loadmat(f)['Ytrain'][:trainLen]
        for dm_epsilon in [1.25]: #DM epsilon = 0.003 siettos
            print(dm_epsilon)
            embedded_space_data = Embed(embedMethod, original_space_data)
            savemat("Embeddings/"+embedMethod+"_"+str(dm_epsilon)+"_"+str(f.split("_")[1].replace(".mat",""))+"_"+str(target_intrinsic_dim)+"_"+str(trainLen)+".mat", {"embedded_space_data": embedded_space_data})

        if plotData == 1:
            fig, ax = plt.subplots(sharex=True, nrows=2, ncols=1)
            pd.DataFrame(original_space_data).plot(ax=ax[0], legend=None)
            pd.DataFrame(embedded_space_data).plot(ax=ax[1], legend=None)

            #pd.DataFrame(embedded_space_data).plot()
            #plt.show()

        #time.sleep(3000)

    except Exception as e:
        print(e)