import numpy as np, time, pickle
import pandas as pd
from tqdm import tqdm
import math
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error, max_error, mean_absolute_error
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 20
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 200)

def metrix(y_actual_DF, y_predicted_DF, metricsLabels):
    y_actual = y_actual_DF.values
    y_predicted = y_predicted_DF.values

    metricList = []
    for choice in metricsLabels:
        if choice == "mean_squared_error":
            metricList.append(mean_squared_error(y_actual, y_predicted))
        elif choice == "root_mean_squared_error":
            MSE = mean_squared_error(y_actual, y_predicted)
            RMSE = math.sqrt(MSE)
            metricList.append(RMSE)
        elif choice == "mean_absolute_error":
            MAE = mean_absolute_error(y_actual, y_predicted)
            metricList.append(MAE)

    return [metricList, metricsLabels]

def Run(params):
    "READ THE DATA"
    #df = pd.read_excel("data\\RES_all.xlsx", nrows=250).set_index('Dates', drop=True).dropna(axis=1, how='all')
    #df.to_excel("data\\testRES.xlsx")
    #df = pd.read_excel("data\\testRES.xlsx").iloc[:, :20]

    df = pd.read_excel("data\\RES_all.xlsx").iloc[:, :20]

    df['Dates'] = df['Dates'].astype(str).str.split(".").str[0]
    df = df.set_index('Dates', drop=True).diff().fillna(0)
    df[df > 1e+10] = 0
    df.plot()
    plt.show()

    c=0
    PredsList = []
    for i in tqdm(range(params['trainSetLength'], df.shape[0], 1)):

        if params['predMode'] == "Expanding":
            intvDF = df.iloc[0:i,:]
        else:
            intvDF = df.iloc[i-params['trainSetLength']:i,:]

        next_out_of_sampleDF = df.iloc[i:i+params['forecastHorizon'],:].fillna(0)

        #if c == 0:
        #    print(intvDF)
        #    print(next_out_of_sampleDF)

        try:
            "Check nulls"
            nullCounterDF = intvDF.isna().sum()
            for c in nullCounterDF.index:
                if nullCounterDF.loc[c] >= intvDF.shape[0]:
                    intvDF[c] = np.random.random(intvDF.shape[0]) * 1e-5

            if params['normaliseFlag']:
                intvDF = (intvDF - intvDF.mean())/intvDF.std()

            if params['model'] == "MVAR":
                roll_forecasting_model = VAR(intvDF.values)
                roll_model_fit = roll_forecasting_model.fit(params['mvar_lag'])
                roll_target_mapping_Preds_All = roll_model_fit.forecast_interval(roll_model_fit.y, steps=params['forecastHorizon'], alpha=0.05)
                roll_Preds = roll_target_mapping_Preds_All[0]
                roll_PredsDF = pd.DataFrame(roll_Preds, columns=df.columns)
                if params['normaliseFlag']:
                    roll_PredsDF = (roll_PredsDF*roll_PredsDF.std())+roll_PredsDF.mean()

            try:
                roll_PredsDF[roll_PredsDF > 1e+10] = np.nan
            except:
                pass
            roll_PredsDF = roll_PredsDF.fillna(0)

        except Exception as e:
            print(e)
            roll_PredsDF = pd.DataFrame(np.zeros((params['forecastHorizon'], len(df.columns))), columns=df.columns)

        #print(intvDF)
        #print(next_out_of_sampleDF)
        #print(roll_PredsDF)
        #time.sleep(3000)

        metricsData = metrix(next_out_of_sampleDF, roll_PredsDF, params['metricsLabels'])
        metricsData[0].insert(0, intvDF.index[-1])
        #print(metricsData)

        PredsList.append([intvDF, next_out_of_sampleDF, roll_PredsDF, metricsData])

    PickledData = [params, PredsList]
    pickle.dump(PickledData, open("PredictionsPackage\\"+params['model']+"_PredsList.p", "wb"))

def Report(params):
    PredictionsPackage = pickle.load(open("PredictionsPackage\\"+params['model']+"_PredsList.p", "rb"))
    PackageParams = PredictionsPackage[0]
    PredictionsData = PredictionsPackage[1]

    reportMetricsList = []
    for elem in tqdm(PredictionsData):
        print(elem[3][0])
        reportMetricsList.append(elem[3][0])
        #print(elem[1])
        #elem[1].plot()
        #plt.show()

    reportMetricsDF = pd.DataFrame(reportMetricsList, columns=["Date"]+PackageParams['metricsLabels']).set_index("Date", drop=True).replace([np.inf, -np.inf], np.nan).dropna()
    reportMetricsDF[reportMetricsDF > 1e+6] = 0
    print("mean : ")
    print(reportMetricsDF.mean())
    print("min : ")
    print(reportMetricsDF.min())
    print("max : ")
    print(reportMetricsDF.max())
    reportMetricsDF["mean_absolute_error"].plot()
    plt.show()
    print(reportMetricsDF)

params = {"model": "MVAR", "predMode" : "Rolling", "trainSetLength":500, "mvar_lag": 24,
          "forecastHorizon" : 24, "normaliseFlag": False,
          "metricsLabels" : ["mean_squared_error", "root_mean_squared_error", "mean_absolute_error"]}

Run(params)
Report(params)