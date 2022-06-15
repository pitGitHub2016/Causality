import sys,os,glob,time
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.io
from scipy.io import savemat
from scipy.interpolate import NearestNDInterpolator
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import manifold
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel, RationalQuadratic, ExpSineSquared, Matern, \
    ConstantKernel
import datafold.dynfold as dfold
import datafold.pcfold as pfold
from datafold.dynfold import (
    GeometricHarmonicsInterpolator as GHI, LaplacianPyramidsInterpolator as LPI, TSCRadialBasis,
    LocalRegressionSelection, TSCTakensEmbedding
)
from scipy.spatial.distance import squareform, pdist
from scipy.interpolate import RBFInterpolator
from datafold.dynfold import LocalRegressionSelection

"/////////////////////////////////////////////////////////////////////////////////////////////////////////////"
def Lift(method, X_trainingSet, eig_trainingSet, eig_Simulation):
    """
     Function to perform lifting

     :param method: available methods are
         'GH' : Geometric Harmonics
         'LP' : Laplacial Pyramids
         'KR' : Kriging (GPRs)
         'SI' : Simple knn interpolation
         'RBF' : Radial Basis Functions interpolation
     :param X_trainingSet: input high-dimensional space data (X), training set
     :param eig_trainingSet: low-dimensional (embedded) space parsimonious eigenvectors (Y), training set
     :param eig_Simulation: low-dimensional (embedded) space parsimonious eigenvectors (Y), predicted (by a specific forecasting methodogy, e.g. VAR(3)) set
     :param knn: input neighbors used for the lifting
     :return: Lifted_Preds : lifted data
     """
    if method == 'GH':
        #pcm = pfold.PCManifold(eig_trainingSet)
        #pcm.optimize_parameters(random_state=random_state, k=lift_optParams_knn)
        #print(5*np.mean(pd.DataFrame(squareform(pdist(pd.DataFrame(eig_trainingSet)))).values))
        #time.sleep(3000)
        GH_cut_off = np.inf #pcm.cut_off, np.inf
        opt_n_eigenpairs = 30 #eig_trainingSet.shape[0]-1
        gh_interpolant_psi_to_X = GHI(pfold.GaussianKernel(epsilon=GH_epsilon),
                                      n_eigenpairs=opt_n_eigenpairs, dist_kwargs=dict(cut_off=GH_cut_off))
        gh_interpolant_psi_to_X.fit(eig_trainingSet, X_trainingSet)
        Lifted_Preds = gh_interpolant_psi_to_X.predict(eig_Simulation)
    elif method == 'LP':
        lpyr_interpolant_psi_to_X = LPI(auto_adaptive=True)
        lpyr_interpolant_psi_to_X.fit(eig_trainingSet, X_trainingSet)
        Lifted_Preds = lpyr_interpolant_psi_to_X.predict(eig_Simulation)
    elif method == 'KR':
        mainKernel_Kriging_GP = 1 * ConstantKernel() + 1 * ExpSineSquared() + 1 * RBF() + 1 * WhiteKernel()  # Official (29/8/2021)
        #mainKernel_Kriging_GP = 1 * Matern(nu=1.5) + 1 * WhiteKernel()
        gpr_model = GaussianProcessRegressor(kernel=mainKernel_Kriging_GP, normalize_y=True)
        gpr_model_fit = gpr_model.fit(eig_trainingSet, X_trainingSet)
        Lifted_Preds = gpr_model_fit.predict(eig_Simulation)
    elif method == 'SI':  # Simple Linear ND Interpolator
        knn_interpolator = NearestNDInterpolator(eig_trainingSet, X_trainingSet)
        Lifted_Preds = knn_interpolator(eig_Simulation)
    elif method == "RBF":
        Lifted_Preds = RBFInterpolator(eig_trainingSet, X_trainingSet, kernel="linear", degree=1, neighbors=100, epsilon=1)(eig_Simulation)
    elif method == "RBFc":
        Lifted_Preds = RBFInterpolator(eig_trainingSet, X_trainingSet, kernel="cubic", neighbors=100, epsilon=1)(eig_Simulation)

    return Lifted_Preds
"/////////////////////////////////////////////////////////////////////////////////////////////////////////////"

random_state = 0

for f in tqdm(glob.glob("TrainModels/*.mat")):#"TrainModels/*.mat", "Embeddings/*.mat"
    print(f)
    if "DMComputeParsimonious" in f: #"LLE", "DMComputeParsimonious", "DM"
        for liftMethod in ["GH", "RBF", "RBFc"]:#"GH", "RBF", "RBFc"
            for GH_epsilon in [0.0052]:
                print(liftMethod)

                original_space_TrainData = scipy.io.loadmat("Datasets/diffusionPDE_0.mat")['Ytrain'][: 4500]

                "Out-of-sample"
                embedded_space_TrainData = scipy.io.loadmat(f)['Ytrain']
                embedded_space_PredictedData = scipy.io.loadmat(f)['ypred']
                "In-sample"
                #embedded_space_TrainData = scipy.io.loadmat(f)['embedded_space_data'][:4500]
                #embedded_space_PredictedData = scipy.io.loadmat(f)['embedded_space_data'][4500:]

                lifted_data = Lift(liftMethod, original_space_TrainData, embedded_space_TrainData, embedded_space_PredictedData)

                savemat("Lifted/"+liftMethod+"_"+str(GH_epsilon)+"_"+f.split("\\")[1], {"lifted_data": lifted_data})
