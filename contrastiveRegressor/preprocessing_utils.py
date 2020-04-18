import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, scale
import itertools
from sklearn import preprocessing
from scipy import stats

def get_MAD(input_array: 'Numpy array'):
    return stats.median_absolute_deviation(input_array)


def get_quantiles(input_DF: 'PD DF', _confidence_level = 99):

  confidence_level = _confidence_level
  qLB = 0.01*(100 - confidence_level)
  qUB = 1 - qLB

  var_quantiles = pd.DataFrame([input_DF.mean(),
  input_DF.quantile(qLB),
  input_DF.quantile(qUB)], index=['mean', 'CI_lb', 'CI_ub'])
  return var_quantiles


def oneHotEncoding(df, categoricalVars):
    blocks_OHE = []
    ohe_mapping  = {}

    blocks_OHE.append(df)
    for varName in categoricalVars:
        blocks_OHE.append(pd.get_dummies(df[varName], columns=varName, prefix=varName))
        ohe_mapping[varName] = blocks_OHE[-1].columns.tolist()

    df_OHE = pd.concat(blocks_OHE, axis=1)

    return df_OHE, ohe_mapping


def labelEncoding(df, categoricalVars):
    dfLE = pd.DataFrame.empty;
    # Apply label encoder
    leVarNames   = []
    encodersList = []
    for varName in categoricalVars:
        currentVarName = varName + 'CAT'
        leVarNames.append(currentVarName)
        currentLE = LabelEncoder()
        df[currentVarName] = currentLE.fit_transform(df[varName].astype(str))
        encodersList.append(currentLE)

    leSolver = dict(zip(categoricalVars, encodersList))
    return df, leSolver, leVarNames


def invertLabelEncoding(dfLE, leSolver, categoricalVarName, leVarName):
    currentLE = leSolver[categoricalVarName];
    leLabels  = currentLE.inverse_transform(dfLE[leVarName])
    return leLabels


def invertEncodedRow(inputRow, leSolver, categoricalVarNames):
    decodedValues = [];
    for idx, varName in enumerate(categoricalVarNames):
        currentLE    = leSolver[varName];
        currentValue = currentLE.inverse_transform(inputRow[idx])
        decodedValues.append(currentValue) 
    return decodedValues


def percentageBasedCutOff(inputDF, colName, cutOff = 95.0):
    totals = inputDF[colName].sum();
    percentage = 100.0*inputDF[colName]/totals;    
    # cutoff at p%
    idxCutOff = percentage.cumsum() < cutOff;
    return inputDF.loc[idxCutOff, :], idxCutOff


def column_transformation(df, columnNames, type_transformation='StandardScaler'):
    prc = [];
    exec('prc=preprocessing.{type_transformation}()')
    print(prc)
    for varName in columnNames:
        x_prime = df[f'{varName}'].values.reshape(-1,1)
        df[f'{varName}_{type_transformation}'] = prc.fit_transform(x_prime)

    return df    

def infer_variable_type(df):
    '''
        Infer the type of variable based on the type
    '''
    categoricalVars = df.select_dtypes(include=['object']).columns.tolist()
    datetimeVars    = df.select_dtypes(include=['datetime64']).columns.tolist()
    nonNumerical    = list(itertools.chain.from_iterable([categoricalVars, datetimeVars]))
    numericalVars   = sorted([*set(df.columns.tolist())-set(nonNumerical)])
    return {'categorical':categoricalVars,
    'datetime': datetimeVars,
    'numerical': numericalVars,
    'non_numerical': nonNumerical}