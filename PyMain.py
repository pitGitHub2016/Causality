from Slider import Slider as sl
import scipy.io, glob, os
from scipy.io import savemat
from scipy.interpolate import NearestNDInterpolator
import itertools, math
import numpy as np, investpy, time, pickle
from math import sqrt
import pandas as pd
from tqdm import tqdm
from statsmodels.tsa.ar_model import AutoReg, AR
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.api import VAR
import statsmodels.stats.api as sms
import matplotlib as mpl
import matplotlib.pyplot as plt
import multiprocessing as mp
from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel, RationalQuadratic, ExpSineSquared, Matern, \
    ConstantKernel
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.interpolate import RBFInterpolator
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import datafold.dynfold as dfold
import datafold.pcfold as pfold
from datafold.dynfold import (
    GeometricHarmonicsInterpolator as GHI, LaplacianPyramidsInterpolator as LPI, TSCRadialBasis,
    LocalRegressionSelection, TSCTakensEmbedding
)
from datafold.dynfold import LocalRegressionSelection

import warnings
warnings.filterwarnings("ignore")
mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 20
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 200)

"GLOBALLY READ THE DATA"
#df = pd.read_excel("data\\RES_all.xlsx", nrows=250).set_index('Dates', drop=True).dropna(axis=1, how='all')
#df.to_excel("data\\testRES.xlsx")

df = pd.read_excel("data\\testRES.xlsx").set_index('Dates', drop=True).diff().fillna(0)

def Run(params):

    PredsList = []
    embeddingList = []
    for i in tqdm(range(params['trainSetLength'], df.shape[0], 1)):

        intvDF = df.iloc[i-params['trainSetLength']:params['trainSetLength'],:]
        print(intvDF)

        if params['model'] == "MVAR":
            roll_forecasting_model = VAR(intvDF.values)
            roll_model_fit = roll_forecasting_model.fit(params['mvar_lag'])
            roll_target_mapping_Preds_All = roll_model_fit.forecast_interval(roll_model_fit.y, steps=24, alpha=0.05)
            row_Preds = roll_target_mapping_Preds_All[0]
            print(type(row_Preds))
            time.sleep(3000)

params = {"model":"MVAR", "trainSetLength":25, "mvar_lag": 5}
Run(params)