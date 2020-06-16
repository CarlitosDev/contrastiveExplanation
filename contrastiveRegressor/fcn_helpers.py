'''
  Helpers
  carlos.aguilar.palacios@gmail.com
'''

import pandas as pd
import os
import pickle
import subprocess
import sys
from datetime import datetime
from shutil import copyfile, move
import uuid
import matplotlib.pyplot as plt
import matplotlib
from os import path as _p
import numpy as np
import re
from sklearn.metrics import r2_score, explained_variance_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from ngboost import NGBRegressor




def get_current_folder():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)))


def to_excel_file(df_to_save, xls_filepath):
    '''
      Save to Excel
    '''
    datetimeVars = df_to_save.select_dtypes(
        include=['datetime64[ns, UTC]']).columns.tolist()

    for i_var in datetimeVars:
        df_to_save[i_var] = df_to_save[i_var].astype(str)

    dataFrameToXLSv2(df_to_save, xls_filepath)
    osOpenFile(xls_filepath)


def to_random_excel_file(df_to_save):
    '''
      Only use with the TF analysis data
    '''
    datetimeVars = df_to_save.select_dtypes(
        include=['datetime64[ns, UTC]']).columns.tolist()

    for i_var in datetimeVars:
        df_to_save[i_var] = df_to_save[i_var].astype(str)

    outputFolder = os.path.join(get_current_folder(), 'data', 'xls_to_delete')
    makeFolder(outputFolder)
    xls_filepath = os.path.join(outputFolder, str(uuid.uuid4()) + '.xlsx')
    dataFrameToXLSv2(df_to_save, xls_filepath)
    osOpenFile(xls_filepath)


def dataFrameToXLSv2(df, xlsFile, sheetName='DF',
                     writeIndex=False, float_format='%.2f', freezeTopRow=True,
                     remove_timezone=True):
    '''
    Write DF to Excel centering the columns and freezing the top row
    '''
    if not df.empty:
        xlsWriter = pd.ExcelWriter(xlsFile, engine='xlsxwriter',
                                   options={'remove_timezone': remove_timezone})

        if remove_timezone:
            for i_var in df.select_dtypes(include=['datetime64[ns, UTC]']).columns.tolist():
                df[i_var] = df[i_var].astype(str)

        df.to_excel(xlsWriter, sheetName,
                    index=writeIndex, float_format=float_format)

        # Get the xlsxwriter workbook and worksheet objects.
        workbook = xlsWriter.book
        worksheet = xlsWriter.sheets[sheetName]

        if freezeTopRow:
            worksheet.freeze_panes(1, 0)

        # set the format for the cells
        cell_format = workbook.add_format()
        cell_format.set_align('center')
        cell_format.set_align('vcenter')

        # set the col format (fake Autolimit)
        colNames_lenght = df.columns.str.len().values
        for col in range(0, df.shape[1]):
            maxWidth = 1 + max(colNames_lenght[col],
                               df.iloc[:, col].astype(str).str.len().max())
            worksheet.set_column(col, col, maxWidth, cell_format)

        xlsWriter.save()


def osOpenFile(filePath):
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.call([opener, filePath])


def makeFolder(thisPath):
    if not os.path.exists(thisPath):
        os.makedirs(thisPath)


# Wrap some plotting fuctions into Matlab's style
def figure():
    return plt.figure()


def plot(x=None, y=None):
    '''
    Missing Matlab so much
        x = np.linspace(0,30,26)
        y = np.random.rand(26)
        plot(x, y)
    '''
    plt.plot(x, y)
    plt.show()


def save_plot_to_pfg(plt, pfg_file_name):
    '''
    Save figure for LaTex pfg
        x = np.linspace(0,30,26)
        y = np.random.rand(26)
        plot(x, y)
    '''
    matplotlib.use('pgf')

    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
        "pgf.preamble": [
            r"\usepackage[utf8x]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{cmbright}",
        ]
    })
    [_, fName] = os.path.split(pfg_file_name)
    plt.savefig(pfg_file_name)

    instructions = '''\\usepackage{caption}
        \\usepackage{pgf}
        \\usepackage{import}
        ...
        \\begin{figure}
        \\begin{center}
        ''' + '\\input{' + fName + '.pgf}' + '''
        \\end{center}
        \\caption{Made with matplotlib's PGF backend.}
        \\end{figure}
        '''
    print(instructions)


# Pickle helpers
def toPickleFile(data, filePath):
    with open(filePath, 'wb') as output:
        pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)

# Binary data, not text


def readPickleFile(filePath):
    with open(filePath, 'rb') as fId:
        pickleData = pickle.load(fId)
    return pickleData

# Write text file


def writeTextFile(thisStr, thisFile):
    with open(thisFile, 'w') as f:
        f.write(thisStr)


def prepareTableLaTeX(df_latex):
    '''
        Prepare the table for the paper
    '''
    latex_prefix = '''\\begin{table*}[t]
    \centering
    \\resizebox{\\textwidth}{!}{'''

    latex_postfix = '''}
    \\caption{Results for}
    \\label{tab:real_forecast_}
    \\end{table*}
    '''
    str_latex = df_latex.to_latex(index=False, float_format='{:3.2f}'.format)
    # Remove the trailing whitespaces
    #str_latex = re.sub(r'(\s+)[\d|-]', ' ', str_latex)
    str_latex = " ".join(str_latex.split())
    # replace the formulas
    #str_latex = str_latex.replace('0.0', '')
    str_latex = str_latex.replace('y\_train', '''$\mathbf{y}$ (sales)''')
    str_latex = str_latex.replace(
        'delta\_y\_train', '''$\mathbf{y_{\Delta}}$''')
    str_latex = str_latex.replace(
        'y\_train\_plus\_delta', '''$\mathbf{y + y_{\Delta}}$''')
    str_latex = str_latex.replace('nan', '')
    # Add variable importance to the text
    str_latex = latex_prefix + str_latex + latex_postfix

    return str_latex


def promotions_reader(baseFolder, baseFile):
    dataFile = _p.expanduser(_p.join(baseFolder, baseFile))
    df_eng_all = pd.read_pickle(dataFile)
    return df_eng_all


def describe_bpns_from_file(baseFolder, baseFile, bpns):
  df_eng_all = BPNS_reader(baseFolder, baseFile, bpns)
  idx_product = df_eng_all.base_product_number_std == bpns
  print(df_eng_all[idx_product].iloc[0])

def describe_bpns(df_eng_all, bpns):
  idx_product = df_eng_all.base_product_number_std == bpns
  print(df_eng_all[idx_product].iloc[0])

def BPNS_reader(baseFolder, baseFile, bpns):
    '''
        Speed things up
        Look for the file with the BPNS. If it does not exist,
        filter by the PSGC and save the file.
    '''
    bpns_filepath = _p.join(get_current_folder(), 'data', bpns + '.pickle')
    if os.path.exists(bpns_filepath):
        df_eng_all = pd.read_pickle(bpns_filepath)
    else:
        dataFile = _p.expanduser(_p.join(baseFolder, baseFile))
        df_temp = pd.read_pickle(dataFile)
        idx_product = df_temp.base_product_number_std == bpns
        this_psgc = df_temp.loc[idx_product, 'product_sub_group_code'].iloc[0]
        idx_psgc = df_temp.product_sub_group_code.str.contains(this_psgc)
        df_eng_all = df_temp[idx_psgc].copy()
        print(f'Saving file {bpns_filepath}...')
        toPickleFile(df_eng_all, bpns_filepath)

    return df_eng_all

def frc_with_random_neighbours(X_train, X_test, num_neighbours, contrastiveReg):
    '''
        Benchmark the Euclidean search vs just random search
    '''

    trainingSize = X_train.shape[0]
    testSize = X_test.shape[0]

    y_k_all_list = []
    y_k_list = []
    y_k_weighted_list = []
    y_delta_list = []

    for idx_test in range(0, testSize):

        idxSorted = np.random.choice(
            trainingSize-1, num_neighbours, replace=False)
        x_A = X_train[idxSorted]
        x_B = np.tile(X_test[idx_test], (num_neighbours, 1))
        X_AB_test = np.concatenate([x_A, x_B], axis=1)

        # differences regarding the reference promotions
        xgb_frc = contrastiveReg.regressor.predict(X_AB_test)

        # Get the average
        y_delta_list.append(xgb_frc)
        y_k_hat_all = xgb_frc + contrastiveReg.y_train[idxSorted]
        y_k_hat = np.mean(y_k_hat_all)

        # overwrite for compatibility
        y_k_hat_distances = y_k_hat
        y_k_weighted_list.append(y_k_hat_distances)

        # Append to the list
        y_k_all_list.append(y_k_hat_all)
        y_k_list.append(y_k_hat)

    # Arrange the forecast as np-arrays
    y_hat_random = np.array(y_k_list)

    return y_hat_random


def frc_plain_CatBoost(num_neighbours, validation_test_size,
                       num_iterations, learning_rate, depth, X_train, y_train, X_test):

    # CatBoost
    cb_model = CatBoostRegressor(iterations=num_iterations, learning_rate=learning_rate,
                                 depth=depth, loss_function='RMSE', cat_features=None, silent=False)

    # Split into training and evaluation
    X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = \
        train_test_split(X_train, y_train, test_size=validation_test_size)

    #
    eval_set = [(X_val_xgb, y_val_xgb)]

    print(f'')
    cb_model.fit(X_train_xgb, y_train_xgb, eval_set=eval_set,
                 verbose=50)  # , logging_level='Info')

    # differences regarding the reference promotions
    xgb_frc = cb_model.predict(X_test)
    return xgb_frc



def frc_AutoGluon(df_train, df_test, 
    categoricalVars, experiment_label = 'grocery',
    responseVar = 'wk1_sales_all_stores'):
    
    import autogluon as ag
    from autogluon import TabularPrediction as task
    # autogluon.task.tabular_prediction.TabularPredictor

    for varName in categoricalVars:
        df_train[varName] = df_train[varName].astype(str)
        df_test[varName] = df_test[varName].astype(str)

    # AutoGluon format
    train_data = task.Dataset(df=df_train)
    test_data = task.Dataset(df=df_test)

    model = task.fit(train_data=train_data, 
    output_directory="auto_gluon/" + experiment_label, label=responseVar,
    hyperparameter_tune=False)


    # Forecast with the best model
    autogluon_frc = model.predict(test_data)
    
    # Forecast with all the models
    individual_frc = {'AG_'+model_to_use: model.predict(test_data, model=model_to_use) \
        for model_to_use in model.model_names}

    return {'autoGluon_frc': autogluon_frc, 'autoGluon_model': model, 'individual_frc': individual_frc}

def get_frc_errors(y, y_hat, verbose=True):
      '''
      Get forecast residuals as e_t = \hat{y} - y
          so e_t > 0 overforecast
          so e_t < 0 underforecast
          e_t = 0 : the dream
      '''
      var_explained = explained_variance_score(y, y_hat)
      e_t = y - y_hat
      abs_e_t = np.abs(e_t)
      
      frc_error = np.sum(abs_e_t)/np.sum(y)
      frc_bias = np.sum(e_t)/np.sum(y_hat)
      frc_acc  = 1.0 - frc_bias
      
      MAE = abs_e_t.mean()
      MSE = np.power(e_t, 2).mean()
      meanError = e_t.mean()
      MAPE = 100*(abs_e_t/np.abs(y)).mean()
      r2 = r2_score(y, y_hat)

      d = {'MAE': MAE,
      'RMSE': np.sqrt(MSE), 
      'meanError': meanError,
      'MAPE': MAPE,
      'R2': r2,
      'frc_error': frc_error,
      'frc_bias': frc_bias,
      'frc_acc': frc_acc,
      'Var explained': var_explained}
      #'MSE': MSE, 
      #,'residuals': e_t}
      if verbose:
        for k,v in d.items():
          print(f'{k}: {v:3.2f}')
      return d


def frc_plain_ngboost(num_iterations, learning_rate, validation_test_size, X_train, y_train, X_test):

    # ngboost
    ngb_model = NGBRegressor(learning_rate=learning_rate, n_estimators=num_iterations)

    # Split into training and evaluation
    X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = \
        train_test_split(X_train, y_train, test_size=validation_test_size)

    # Fit NGBoost
    ngb_model.fit(X_train_xgb, y_train_xgb, X_val=X_val_xgb, Y_val=y_val_xgb)

    # differences regarding the reference promotions
    ngb_frc = ngb_model.predict(X_test)
    return ngb_frc


def frc_plain_extratrees(num_iterations, depth, validation_test_size, X_train, y_train, X_test):
  # ExtraTrees
  xtt_model = ExtraTreesRegressor(n_estimators=num_iterations, 
  criterion='mse', max_depth=depth, n_jobs=-1)

  xtt_model.fit(X_train, y_train)

  xtt_frc = xtt_model.predict(X_test)
  return xtt_frc