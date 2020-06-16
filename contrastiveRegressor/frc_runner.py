'''
  Runner for the contrastiveRegressor class 

  This script runs the contrastiveRegressor for the Tesco data
  and arranges the results for the paper

  Updates:
  12.03.2020 - Add info
  05.06.2020 - Add test set as an option in the model agnostic runner

  carlos.aguilar.palacios@gmail.com

'''

import matplotlib
from mpl_toolkits import mplot3d
import category_encoders as ce
import pandas as pd
import numpy as np
from os import path as _p

# from regressor_contrastive_explanation import dm_regressor, arrange_regressor_results, get_feature_importances
from contrastiveRegressor import contrastiveRegressor
from common_metrics import get_frc_errors

from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool

import seaborn as sns
import matplotlib.pyplot as plt
import preprocessing_utils as pt
from generate_Gompertz_sales import generate_Gompertz_sales
import fcn_helpers as fhelp
import datetime as dt



def frc_runner(df_eng_all, base_product_number_std,
  categoricalVars, numerical_vars,
  num_neighbours, validation_test_size,
  num_iterations, learning_rate, depth,
  feat_importance_keyword = 'feature_importances_',
  experiment_label = 'grocery',
  responseVar = 'wk1_sales_all_stores',
  identifierVar = 'promo_identifier_latex',
  doVisualisation=True,
  doSaveExcel = True):

  # Create an identifier
  df_eng_all['promo_identifier'] = df_eng_all.base_product_number_std + \
    ' (' + df_eng_all.offer_description + ')'

  # for latex
  #all_bpns = df_eng_all.base_product_number_std.unique().tolist()
  all_bpns = [*set(df_eng_all.base_product_number_std)-set(base_product_number_std)]
  all_bpns_as_ids = ['product_' + str(idx) for idx in range(0, len(all_bpns))]
  dict_bpns = dict(zip(all_bpns, all_bpns_as_ids))
  dict_bpns.update({base_product_number_std: 'new_product'})

  df_eng_all['promo_identifier_latex'] = df_eng_all.base_product_number_std.map(dict_bpns)


  prefix = experiment_label + '_' + base_product_number_std + '_' + dt.datetime.today().strftime('%d_%m_%Y_%HH')

  idx_product = df_eng_all.base_product_number_std == base_product_number_std
  df_test = df_eng_all[idx_product].copy()
  df_test.reset_index(inplace=True)

  print(df_test.promo_identifier.iloc[0])

  # From the product, find the PSGC and exclude the product itself
  this_psgc = df_test.product_sub_group_code.iloc[0]
  idx_psgc = df_eng_all.product_sub_group_code.str.contains(this_psgc)
  df = df_eng_all[idx_psgc & ~ idx_product].copy()
  df.reset_index(inplace=True)

  # NUMERICAL:
  # Make sure the numericals are encoded as such.
  vTypes = pt.infer_variable_type(df)
  # Numerical vars
  #numerical_vars = [*set(inputVars) - set(categoricalVars)]
  # Make sure all of them are coded as numerical ones
  missing_vars = [*set(numerical_vars)-set(vTypes['numerical'])]
  if missing_vars:
    print(f'List of missing vars {missing_vars}')

  # CATEGORICAL: Save a copy of the categorical variables as 
  # this encoder overwrites them
  enc_postfix = '_encoded'
  enc_categoricalVars = []
  for varName in categoricalVars:
          currentVarName = varName + enc_postfix
          enc_categoricalVars.append(currentVarName)
          df[currentVarName] = df[varName]
          df_test[currentVarName] = df_test[varName]

  # For quick remapping
  catVarsMapping = dict(zip(enc_categoricalVars, categoricalVars))

  # get the index of the categorical variables for CatBoost
  inputVars = numerical_vars + enc_categoricalVars



  # JamesSteinEncoder
  inputVars_encoder = inputVars + categoricalVars
  encoder_js  = ce.JamesSteinEncoder(cols=enc_categoricalVars, verbose=1)

  # fit training
  df_A_enc = encoder_js.fit_transform(df[inputVars_encoder], df[responseVar])
  df_A_enc[responseVar] = df[responseVar]
  df_A_enc[identifierVar] = df[identifierVar]
  # fit test
  df_test_enc = encoder_js.transform(df_test[inputVars_encoder])
  df_test_enc[responseVar] = df_test[responseVar]
  df_test_enc[identifierVar] = df_test[identifierVar]


  '''
    Train, val and test
  '''
  num_inputVars = len(inputVars)

  X_train = df_A_enc[inputVars].values
  y_train = df_A_enc[responseVar].values
  id_train = df_A_enc[identifierVar].values

  X_test =  df_test_enc[inputVars].values
  y_test = df_test_enc[responseVar].values


  '''
    Model. Using CatBoost here
  '''
  # Create the forecaster
  contrastiveReg = contrastiveRegressor(num_neighbours = num_neighbours, 
    validation_test_size = validation_test_size)

  # CatBoost
  cb_model = CatBoostRegressor(iterations=num_iterations, learning_rate=learning_rate,
  depth=depth, loss_function='RMSE', cat_features=None, silent=True)
  # Set the regressor
  contrastiveReg.set_regressor(cb_model, feat_importance_keyword, inputVars)
  # fit the regressor
  contrastiveReg.fit(X_train, y_train)
  # eval results
  contrastiveReg.predict_eval_test()
  eval_results = contrastiveReg.get_results()

  # Predict
  contrastiveReg.predict(X_test, categorical_mapping = catVarsMapping)
  cold_start_results = contrastiveReg.get_results()


  # Sort by importance
  df_feature_importances = cold_start_results.get('df_feat_importances', None)
  print(df_feature_importances)


  # Arrange the results in a DF so we can easily plot them
  df_frc = df_test.copy()
  df_frc['y_hat'] = cold_start_results['y_hat_weighted']


  # review the cold-start forecast
  all_cold_forecast = []
  all_frc_latex = []

  model_vars = ['y_actual', 'y_forecast', \
      'y_train', 'delta_y_train', \
      'y_train_plus_delta', 'y_train_distances']

  vars_latex = ['y_train', 'delta_y_train', \
      'y_train_plus_delta', 'y_train_distances']


  # Annonymise
  dict_feature_importances = df_feature_importances.to_dict(orient='dict').get(0, None)
  vars_model = numerical_vars + categoricalVars
  inputObfuscated = []
  for idx, iVar in enumerate(vars_model,1):
    str_feat_weight = f' (vi:{dict_feature_importances.get(iVar, 0):3.2f})'
    inputObfuscated.append('v_' + str(idx) + str_feat_weight)

  mapObfuscatedVars = dict(zip(vars_model, inputObfuscated))
  mapObfuscatedVars[responseVar] = 'response'

  list_vars = [iVar for iVar in df_feature_importances.index.tolist() if 'ref_' not in iVar]

  for idx_review in range(df_test.shape[0]):
    print('Running {idx_review}...')
    df_forecast_ext = contrastiveReg.arrange_regressor_results(idx_review, df_A_enc, \
    y_train, id_train, list_vars, \
    identifierVar, df_test_enc, y_test, num_inputVars)
    df_forecast_ext.reset_index(inplace=True)
    
    df_forecast_ext['y_train'].iloc[-2] = \
      df_forecast_ext['y_weighted_forecast'].iloc[-2]
    
    print(df_forecast_ext)
    all_cold_forecast.append(df_forecast_ext)

    
    y_actual = df_forecast_ext['y_actual'].iloc[-2]
    y_forecast = df_forecast_ext['y_weighted_forecast'].iloc[-2]
    print(f'(actual: {y_actual:3.2f}, forecast: {y_forecast:3.2f})')
    all_frc_latex.append(df_forecast_ext[0:-1])

  # Append them all
  df_all_cold_forecast = pd.concat(all_cold_forecast)
  df_all_latex = pd.concat(all_frc_latex)


  top_n_features = 4 
  if top_n_features < num_inputVars:
    list_vars_LaTeX = list_vars[0:top_n_features]
  else:
    list_vars_LaTeX = list_vars

  if doSaveExcel:

      # Created an obfuscated LaTeX version
      df_latex = df_all_latex[[identifierVar] + list_vars_LaTeX + vars_latex].copy()
      # Obfuscate the TSR so we don't get into problems
      tsr_col = 'total_store_revenue' 
      if tsr_col in df_latex.columns.tolist(): 
        df_latex[tsr_col] = df_latex[tsr_col].apply(lambda x: np.log10(x))
        df_latex.rename(columns=mapObfuscatedVars, inplace=True)

      str_latex = fhelp.prepareTableLaTeX(df_latex)
      tex_file_name = _p.join('tex', prefix + '_table.tex')
      fhelp.writeTextFile(str_latex, tex_file_name)

      # Excel
      list_vars.extend(model_vars)
      vars_xls = list_vars + [identifierVar]
      xlsx_file_name = _p.join('results', prefix + '_table.xlsx')
      fhelp.to_excel_file(df_all_cold_forecast[vars_xls], xlsx_file_name)

  '''
    Visualise
  '''
  if doVisualisation:
    varX = list_vars[0]
    varY = list_vars[1]
    varZ = responseVar

    if varX in categoricalVars:
      df[varX] = df[varX].astype(int)
      df_frc[varX] = df_frc[varX].astype(int)
    if varY in categoricalVars:
      df[varY] = df[varY].astype(int)
      df_frc[varY] = df_frc[varY].astype(int)

    _alpha = 0.75
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # All the subgroup
    ax.scatter(df[varX], df[varY], df[varZ], alpha=0.15)
    # Also plot the test points
    ax.scatter(df_frc[varX], df_frc[varY], df_frc['y_hat'], alpha=1.0, label='cold-forecast', color='red', s=75)
    # plot the selected products

    for idx_forecast in range(0, len(cold_start_results['y_hat'])):
      idx_closest_promos = cold_start_results['y_idx_closest_promos'][idx_forecast]
      df_A = df.iloc[idx_closest_promos].copy()
      ax.scatter(df_A[varX], df_A[varY], df_A[varZ], alpha=_alpha, label='neighbours_' + str(idx_forecast), s=50)


    for idx, row in df_frc.iterrows():
        #point_name = f'F{(1+idx)} ({row[varX]}, {row[varY]:2.0f}, {row.y_hat:3.2f})'
        point_name = f'Frc_{idx}'
        ax.text(row[varX]+2.5,row[varY],row['y_hat'], point_name, color='black', fontsize=9)

    ax.set_xlabel(mapObfuscatedVars[varX])
    ax.set_ylabel(mapObfuscatedVars[varY])
    ax.set_zlabel(mapObfuscatedVars[varZ])

    ax.view_init(elev=32, azim=-50)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show(block = True)
    pfg_file_name = _p.join('figs', prefix + '_plot_3D.png')
    plt.savefig(pfg_file_name)


  d = {'eval_results': eval_results, \
    'cold_start_results': cold_start_results, \
    'contrastiveRegressor': contrastiveReg,
    'X_test': X_test, 'y_test': y_test, 
    'df_train': df, 'df_test': df_test}
  return d



def frc_runner_model_agnostic(df_eng_all, base_product_number_std,
  categoricalVars, numerical_vars,
  num_neighbours, validation_test_size,
  num_iterations, learning_rate, depth,
  base_regressors: 'list of regressors',
  feat_importance_keyword = 'feature_importances_',
  experiment_label = 'grocery',
  responseVar = 'wk1_sales_all_stores',
  identifierVar = 'promo_identifier_latex',
  doVisualisation=True,
  doSaveExcel = True, 
  df_test=None):

  '''
    If df_test is not set, then get it as the whole BPNS
  '''

  # Create an identifier
  df_eng_all['promo_identifier'] = df_eng_all.base_product_number_std + \
    ' (' + df_eng_all.offer_description + ')'

  # for latex
  #all_bpns = df_eng_all.base_product_number_std.unique().tolist()
  all_bpns = [*set(df_eng_all.base_product_number_std)-set(base_product_number_std)]
  all_bpns_as_ids = ['product_' + str(idx) for idx in range(0, len(all_bpns))]
  dict_bpns = dict(zip(all_bpns, all_bpns_as_ids))
  dict_bpns.update({base_product_number_std: 'new_product'})

  # Deactivate this one for one experiment
  print('Testing line 327 frc_runner')
  near_PROD = False
  # This is the one to go into PROD
  if near_PROD:
    df_eng_all['promo_identifier_latex'] = df_eng_all.base_product_number_std.map(dict_bpns)
  else:
    df_naming = df_eng_all[['base_product_number_std', 'offer_description']].copy()
    df_naming.drop_duplicates(keep='first', inplace=True)
    df_naming['descriptor'] = df_naming.base_product_number_std + ' (' + df_naming.offer_description + ')'
    dict_mapper = dict(zip(df_naming.base_product_number_std, df_naming.descriptor))
    df_eng_all['promo_identifier_latex'] = df_eng_all.base_product_number_std.map(dict_mapper)


  prefix = experiment_label + '_' + base_product_number_std + '_' + dt.datetime.today().strftime('%d_%m_%Y_%HH')

  # From the product, find the PSGC and exclude the product itself
  this_psgc = df_test.product_sub_group_code.iloc[0]
  idx_psgc = df_eng_all.product_sub_group_code.str.contains(this_psgc)

  # We either provide with a test set, or thew algo interprets that we want to forecast the whole cold-start product
  if isinstance(df_test, type(None)):
    idx_product = df_eng_all.base_product_number_std == base_product_number_std
    df_test = df_eng_all[idx_product].copy()
    df_test.reset_index(inplace=True)
    # Exclude the product itself
    df = df_eng_all[idx_psgc & ~ idx_product].copy()

  elif isinstance(df_test, pd.DataFrame):
    if 'promo_identifier_latex' not in df_test.columns.tolist():
      df_test['promo_identifier_latex'] = df_test.base_product_number_std.map(dict_bpns)
    if 'promo_identifier' not in df_test.columns.tolist():
      df_test['promo_identifier'] = df_test.base_product_number_std + \
    ' (' + df_test.offer_description + ')'
    # Allow any product within the category
    df = df_eng_all[idx_psgc].copy()

  print(df_test.promo_identifier.iloc[0])
  df.reset_index(inplace=True)

  # NUMERICAL:
  # Make sure the numericals are encoded as such.
  vTypes = pt.infer_variable_type(df)
  # Numerical vars
  #numerical_vars = [*set(inputVars) - set(categoricalVars)]
  # Make sure all of them are coded as numerical ones
  missing_vars = [*set(numerical_vars)-set(vTypes['numerical'])]
  if missing_vars:
    print(f'List of missing vars {missing_vars}')

  # CATEGORICAL: Save a copy of the categorical variables as 
  # this encoder overwrites them
  enc_postfix = '_encoded'
  enc_categoricalVars = []
  for varName in categoricalVars:
          currentVarName = varName + enc_postfix
          enc_categoricalVars.append(currentVarName)
          df[currentVarName] = df[varName]
          df_test[currentVarName] = df_test[varName]

  # For quick remapping
  catVarsMapping = dict(zip(enc_categoricalVars, categoricalVars))

  # get the index of the categorical variables for CatBoost
  inputVars = numerical_vars + enc_categoricalVars


  # JamesSteinEncoder
  inputVars_encoder = inputVars + categoricalVars
  encoder_js  = ce.JamesSteinEncoder(cols=enc_categoricalVars, verbose=1)

  # fit training
  df_A_enc = encoder_js.fit_transform(df[inputVars_encoder], df[responseVar])
  df_A_enc[responseVar] = df[responseVar]
  df_A_enc[identifierVar] = df[identifierVar]
  # fit test
  df_test_enc = encoder_js.transform(df_test[inputVars_encoder])
  df_test_enc[responseVar] = df_test[responseVar]
  df_test_enc[identifierVar] = df_test[identifierVar]


  '''
    Train, val and test
  '''
  num_inputVars = len(inputVars)

  X_train = df_A_enc[inputVars].values
  y_train = df_A_enc[responseVar].values
  id_train = df_A_enc[identifierVar].values

  X_test =  df_test_enc[inputVars].values
  y_test = df_test_enc[responseVar].values


  '''
    Model where each base learner comes from the list of base_regressors
  '''
  contrastiveRegressors = []
  for base_regressor in base_regressors:

    print(f'Training and Forecasting with {base_regressor}...')
    
    # Create the forecaster
    contrastiveReg = contrastiveRegressor(num_neighbours = num_neighbours, 
      validation_test_size = validation_test_size)

    # Set the regressor
    contrastiveReg.set_regressor(base_regressor, feat_importance_keyword, inputVars)
    # fit the regressor
    contrastiveReg.fit(X_train, y_train)
    # eval results
    contrastiveReg.predict_eval_test()
    eval_results = contrastiveReg.get_results()

    # Predict
    contrastiveReg.predict(X_test, categorical_mapping = catVarsMapping)
    cold_start_results = contrastiveReg.get_results()


    # Sort by importance
    df_feature_importances = cold_start_results.get('df_feat_importances', None)
    print(df_feature_importances)


    # Arrange the results in a DF so we can easily plot them
    df_frc = df_test.copy()
    df_frc['y_hat'] = cold_start_results['y_hat_weighted']


    # review the cold-start forecast
    all_cold_forecast = []
    all_frc_latex = []

    model_vars = ['y_actual', 'y_forecast', \
        'y_train', 'delta_y_train', \
        'y_train_plus_delta', 'y_train_distances']

    vars_latex = ['y_train', 'delta_y_train', \
        'y_train_plus_delta', 'y_train_distances']


    # Annonymise
    dict_feature_importances = df_feature_importances.to_dict(orient='dict').get(0, None)
    vars_model = numerical_vars + categoricalVars
    inputObfuscated = []
    for idx, iVar in enumerate(vars_model,1):
      str_feat_weight = f' (vi:{dict_feature_importances.get(iVar, 0):3.2f})'
      inputObfuscated.append('v_' + str(idx) + str_feat_weight)

    mapObfuscatedVars = dict(zip(vars_model, inputObfuscated))
    mapObfuscatedVars[responseVar] = 'response'

    list_vars = [iVar for iVar in df_feature_importances.index.tolist() if 'ref_' not in iVar]

    for idx_review in range(df_test.shape[0]):
      print('Running {idx_review}...')
      df_forecast_ext = contrastiveReg.arrange_regressor_results(idx_review, df_A_enc, \
      y_train, id_train, list_vars, \
      identifierVar, df_test_enc, y_test, num_inputVars)
      df_forecast_ext.reset_index(inplace=True)
      
      df_forecast_ext['y_train'].iloc[-2] = \
        df_forecast_ext['y_weighted_forecast'].iloc[-2]
      
      print(df_forecast_ext)
      all_cold_forecast.append(df_forecast_ext)

      
      y_actual = df_forecast_ext['y_actual'].iloc[-2]
      y_forecast = df_forecast_ext['y_weighted_forecast'].iloc[-2]
      print(f'(actual: {y_actual:3.2f}, forecast: {y_forecast:3.2f})')
      all_frc_latex.append(df_forecast_ext[0:-1])

    # Append them all
    df_all_cold_forecast = pd.concat(all_cold_forecast)
    df_all_latex = pd.concat(all_frc_latex)


    top_n_features = 4 
    if top_n_features < num_inputVars:
      list_vars_LaTeX = list_vars[0:top_n_features]
    else:
      list_vars_LaTeX = list_vars

    if doSaveExcel:

        # Created an obfuscated LaTeX version
        df_latex = df_all_latex[[identifierVar] + list_vars_LaTeX + vars_latex].copy()
        # Obfuscate the TSR so we don't get into problems
        tsr_col = 'total_store_revenue' 
        if tsr_col in df_latex.columns.tolist(): 
          df_latex[tsr_col] = df_latex[tsr_col].apply(lambda x: np.log10(x))
          df_latex.rename(columns=mapObfuscatedVars, inplace=True)

        str_latex = fhelp.prepareTableLaTeX(df_latex)
        tex_file_name = _p.join('tex', prefix + '_table.tex')
        fhelp.writeTextFile(str_latex, tex_file_name)

        # Excel    
        list_vars.extend(model_vars)
        vars_xls = list_vars + [identifierVar]

        print(vars_xls)
        print(df_all_cold_forecast.head())

        fileName = 'forecast_' + type(base_regressor).__name__ + \
        f'''neighbours {num_neighbours}-validation {validation_test_size}-''' + \
        f'''iterations {num_iterations}-learning_rate {learning_rate}-''' + \
        f'''depth {depth}-''' + dt.datetime.today().strftime('%d_%m_%Y_%HH%mm') + '.xlsx'
        xlsx_file_name = _p.join('forecast', fileName)
        fhelp.to_excel_file(df_all_cold_forecast[vars_xls], xlsx_file_name)

    '''
      Visualise
    '''
    if doVisualisation:
      varX = list_vars[0]
      varY = list_vars[1]
      varZ = responseVar

      if varX in categoricalVars:
        df[varX] = df[varX].astype(int)
        df_frc[varX] = df_frc[varX].astype(int)
      if varY in categoricalVars:
        df[varY] = df[varY].astype(int)
        df_frc[varY] = df_frc[varY].astype(int)

      _alpha = 0.75
      fig = plt.figure()
      ax = plt.axes(projection='3d')

      # All the subgroup
      ax.scatter(df[varX], df[varY], df[varZ], alpha=0.15)
      # Also plot the test points
      ax.scatter(df_frc[varX], df_frc[varY], df_frc['y_hat'], alpha=1.0, label='cold-forecast', color='red', s=75)
      # plot the selected products

      for idx_forecast in range(0, len(cold_start_results['y_hat'])):
        idx_closest_promos = cold_start_results['y_idx_closest_promos'][idx_forecast]
        df_A = df.iloc[idx_closest_promos].copy()
        ax.scatter(df_A[varX], df_A[varY], df_A[varZ], alpha=_alpha, label='neighbours_' + str(idx_forecast), s=50)


      for idx, row in df_frc.iterrows():
          #point_name = f'F{(1+idx)} ({row[varX]}, {row[varY]:2.0f}, {row.y_hat:3.2f})'
          point_name = f'Frc_{idx}'
          ax.text(row[varX]+2.5,row[varY],row['y_hat'], point_name, color='black', fontsize=9)

      ax.set_xlabel(mapObfuscatedVars[varX])
      ax.set_ylabel(mapObfuscatedVars[varY])
      ax.set_zlabel(mapObfuscatedVars[varZ])

      ax.view_init(elev=32, azim=-50)
      ax.legend()
      ax.grid(True)

      plt.tight_layout()
      plt.show(block = True)
      pfg_file_name = _p.join('figs', prefix + '_plot_3D.png')
      plt.savefig(pfg_file_name)

    used_vars = numerical_vars + categoricalVars + [responseVar]

    contrastiveRegressors.append({'eval_results': eval_results, \
      'model_type': type(base_regressor).__name__,
      'cold_start_results': cold_start_results, \
      'contrastiveRegressor': contrastiveReg,
      'X_test': X_test, 'y_test': y_test, 
      'df_train': df[used_vars], 'df_test': df_test[used_vars]})
  
  return contrastiveRegressors