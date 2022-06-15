import sys,os,glob,time
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.io
from statsmodels.tsa.api import VAR
from scipy.io import savemat
from scipy.interpolate import NearestNDInterpolator
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import manifold
import numpy as np
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel, RationalQuadratic, ExpSineSquared, Matern, \
    ConstantKernel
import datafold.dynfold as dfold
import datafold.pcfold as pfold
from datafold.dynfold import (
    GeometricHarmonicsInterpolator as GHI, LaplacianPyramidsInterpolator as LPI, TSCRadialBasis,
    LocalRegressionSelection, TSCTakensEmbedding
)
from scipy.interpolate import RBFInterpolator
from datafold.dynfold import LocalRegressionSelection

random_state=0

def reframeData(dataIn, reframeStep, varSelect, **kwargs):
    """
    Function to reframe a dataset into lagged instances of the input matrix :
    ####################
    dataIn : the input matrix
    reframeStep : up to which lag to create the instances
    varSelect (int) : which variable to return as 'Y' for any potential regression using the x, or if 'all' return all vars

    return [X_all_gpr, Y_all_gpr, lastY_test_point_gpr]
    X_all_gpr : the predictors (lagged instances)
    Y_all_gpr : the targets Y (as per varselect)
    lastY_test_point_gpr : the last point to be the next input (test point) for an online ML rolling prediction framework
    """
    if "frameConstructor" in kwargs:
        frameConstructor = kwargs["frameConstructor"]
    else:
        frameConstructor = "ascending"

    if "returnMode" in kwargs:
        returnMode = kwargs["returnMode"]
    else:
        returnMode = "ML"

    baseDF = pd.DataFrame(dataIn)

    if frameConstructor == "ascending":
        looperRange = range(reframeStep + 1)
    elif frameConstructor == "descending":
        looperRange = range(reframeStep, -1, -1)

    df_List = []
    for i in looperRange:
        if i == 0:
            subDF_i0 = baseDF.copy()
            subDF_i0.columns = ["base_" + str(x) for x in subDF_i0.columns]
            df_List.append(subDF_i0)
        else:
            subDF = baseDF.shift(i)  # .fillna(0)
            subDF.columns = ["delay_" + str(i) + "_" + str(x) for x in subDF.columns]
            df_List.append(subDF)

    df = pd.concat(df_List, axis=1).dropna()

    if returnMode == "ML":

        if varSelect == "all":
            Y_DF = df.loc[:, [x for x in df.columns if "base_" in x]]
        else:
            Y_DF = df.loc[:, "base_" + str(varSelect)]
        X_DF = df.loc[:, [x for x in df.columns if "delay_" in x]]
        lastY_test_point = df.loc[df.index[-1], [x for x in df.columns if "base_" in x]]

        X_all_gpr = X_DF.values
        if isinstance(Y_DF, pd.Series) == 1:
            Y_all_gpr = Y_DF.values.reshape(-1, 1)
        else:
            Y_all_gpr = Y_DF.values

        lastY_test_point_gpr = lastY_test_point.values.reshape(1, -1)

        return [X_all_gpr, Y_all_gpr, lastY_test_point_gpr]

    elif returnMode == "Takens":

        return df

def get_ML_Predictions(MLmethod, predictorsData, forecastHorizon):

    MLmethodSplit = MLmethod.split(",")
    normaliseFlag = False
    #print("MLmethodSplit = ", MLmethodSplit)
    if MLmethodSplit[0] == "MVAR":
        forecasting_model = VAR(predictorsData)
        model_fit = forecasting_model.fit(int(MLmethodSplit[1]))
        target_mapping_Preds_All = model_fit.forecast_interval(model_fit.y, steps=forecastHorizon, alpha=0.05)
        Preds_List = target_mapping_Preds_All[0]
        print(Preds_List)
    else:
        if MLmethodSplit[0] == "GP":
            #mainKernel = 1 * ConstantKernel() + 1 * ExpSineSquared() + 1 * Matern() + 1 * WhiteKernel()
            mainKernel = 1 * Matern(nu=0.5) + 1 * WhiteKernel()
            model_List = [
                GaussianProcessRegressor(kernel=mainKernel, random_state=random_state) #alpha=0.01,, n_restarts_optimizer=2
                for var in range(predictorsData.shape[1])]
        elif MLmethodSplit[0] == "FFNN":
            normaliseFlag = True
            model_List = []
            for var in range(predictorsData.shape[1]):
                ANN_model = Sequential()
                ANN_model.add(Dense(7, input_dim=predictorsData.shape[1], activation='sigmoid'))
                ANN_model.add(Dense(1, activation='linear'))
                ANN_model.compile(loss='mse', optimizer='adam')
                model_List.append(ANN_model)
        if normaliseFlag == True:
            muTrain = pd.DataFrame(predictorsData).mean()
            stdTrain = pd.DataFrame(predictorsData).std()
            predictorsData = ((pd.DataFrame(predictorsData) - muTrain)/stdTrain).values

        Preds_List = []
        for step_i in tqdm(range(forecastHorizon)):

            models_preds_list = []
            for modelIn in range(len(model_List)):
                if step_i == 0:

                    roll_reframedData = reframeData(predictorsData, int(MLmethodSplit[1]), modelIn)

                    if MLmethodSplit[1] in ["FFNN"]:
                        model_List[modelIn].fit(roll_reframedData[0], roll_reframedData[1], validation_split=0.2, verbose=0, epochs=1000, callbacks=[EarlyStopping(patience=10, verbose=1)])
                    else:
                        model_List[modelIn].fit(roll_reframedData[0], roll_reframedData[1])
                    try:
                        print("model_List[", modelIn, "].kernel = ", model_List[modelIn].kernel_)
                    except:
                        pass
                    sub_row_Preds = model_List[modelIn].predict(roll_reframedData[2])

                else:
                    sub_row_Preds = model_List[modelIn].predict(total_row_subPred.reshape(roll_reframedData[2].shape))

                models_preds_list.append(sub_row_Preds[0][0])

            total_row_subPred = np.array(models_preds_list)

            #print("step_i = ", step_i, ", total_row_subPred = ", pd.Series(total_row_subPred))
            #time.sleep(3000)
            Preds_List.append(total_row_subPred)

        if normaliseFlag == True:
            normedPreds = pd.DataFrame(Preds_List)
            deNormedPreds = (normedPreds * stdTrain)+muTrain
            Preds_List = deNormedPreds.values.tolist()

    return Preds_List

plotData = 1
for f in tqdm(glob.glob("data/*.mat")):
    print(f)
    if 'RESonly' in f: #"RESonly", "RES_allData"
        Ytrain = scipy.io.loadmat(f)['Ytrain'][:20900]
        model = "GP,1" #"MVAR,1", "GP,1"
        ypred = get_ML_Predictions(model, Ytrain, 52)

        savemat("TrainModels/"+f.split("\\")[1].split(".")[0]+"_"+model+".mat", {"Ytrain": Ytrain, "ypred" : ypred})
