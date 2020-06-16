import pandas as pd
import numpy as np
import random
from sklearn.base import BaseEstimator, RegressorMixin
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import pairwise_distances, r2_score, explained_variance_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from ngboost import NGBRegressor

class contrastiveRegressor(BaseEstimator, RegressorMixin):
  '''contrastiveRegressor class compatible with sklearn.

    Docstring description for contrastiveRegressor (TODO)

    Methods
    -------
    fit(self, X, y)
        Fits the regressor

    Updates
    -------
    16.03.2020 - Euclidean distance as per the paper
    17.03.2020 - Symmetrical Euclidean W = Q.T Q
    04.06.2020 - Add NGBoost as another base regressor


    Contact
    -------
    carlos.aguilar.palacios@gmail.com

  '''
  # TODO: All estimators should specify all the
  #  parameters that can be set at the class level in their __init__ as explicit keyword arguments 
  # (no *args or **kwargs).
  numRecords = 0
  numFeatures = 0
  trainingSize = 0
  M = []
  e = []

  regressor = []
  inputVars = []

  
  # Default class parameters - explicitly initialised in __init__
  num_neighbours = None
  validation_test_size = None
  feat_importance_keyword = None
  is_fitted = False

  feat_importances = None
  x_weights = None
  x_ref_weights = None
  # The weights vector does not play by the book :(
  x_ref_weights = None
  int_vars = None

  X_val = None
  y_val = None
  eval_set = None

  normalisation_type = 'z-score'
  normalisation_type = 'z-mungus'

  # Default regressor is CatBoost
  regressor = CatBoostRegressor(iterations=100, 
    learning_rate=0.05, depth=12, 
    loss_function='RMSE', 
    cat_features=None, 
    silent=True)
  
  results = {}

  # Constructor
  def __init__(self,
    num_neighbours = 5, 
    validation_test_size = 0.20, 
    feat_importance_keyword = 'feature_importances_'):
    
    self.num_neighbours = num_neighbours
    self.validation_test_size = validation_test_size
    self.feat_importance_keyword = feat_importance_keyword


  def set_regressor(self, _regressor, _feat_importance_keyword, _input_vars):
    '''Set the regressor'''
    self.regressor = _regressor
    self.feat_importance_keyword = _feat_importance_keyword
    self.inputVars = _input_vars
  
  # The fit method also ALWAYS has to return self.
  def fit(self, X, y, training_split=None, num_neighbours = None, 
    validation_test_size=None, verbosity=False, sortByResponseVar=False):
    '''
    Arrange the contrastive explainer using a given model
    (explicit differences model) and fit

    Parameters
    ----------
    X : np.array
      Real valued dependent data.
        
    y : np.array
        The response variable.
    '''
 
    # Save the X and y sets
    self.X_train = X.copy()
    self.y_train = y.copy()

    # Number of promotions used to compare to
    self.numRecords  = self.X_train.shape[0]
    self.numFeatures = self.X_train.shape[1]
    
    if num_neighbours:
      self.num_neighbours = num_neighbours

    if validation_test_size:
      self.validation_test_size = validation_test_size

    # Split into training and evaluation
    X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = \
      train_test_split(self.X_train, self.y_train, test_size=self.validation_test_size)

    self.X_val = X_val_xgb
    self.y_val = y_val_xgb

    trainingSize = X_train_xgb.shape[0]

    print('Preparing Training set...')

    # Create the training set
    M = []
    e = []
    currentPromo    = np.zeros(trainingSize, dtype=bool)
    remainingPromos = np.ones(trainingSize, dtype=bool)

    for idx in range(0, trainingSize):

      currentPromo[idx]    = True
      remainingPromos[idx] = False

      if sortByResponseVar:
        response_distances = np.abs(y_train_xgb[currentPromo]-y_train_xgb[remainingPromos])
        idx_neighbours = np.argsort(response_distances)[0:self.num_neighbours]
      else:  
        idx_neighbours = np.random.choice(trainingSize-1, self.num_neighbours, replace=False)


      x_A = X_train_xgb[remainingPromos][idx_neighbours]
      x_B = np.tile(X_train_xgb[currentPromo], (self.num_neighbours, 1))
      X_AB = np.concatenate([x_A, x_B], axis=1)
      M.append(X_AB)

      y_BA = y_train_xgb[currentPromo] - y_train_xgb[remainingPromos][idx_neighbours]
      e.append(y_BA)

      currentPromo[idx]    = False
      remainingPromos[idx] = True

    M = np.concatenate(M, axis=0).copy()
    e = np.concatenate(e, axis=0).copy()
      

    # Create the evaluation set where
    # the neighbours are withdrawn from the training set
    M_eval = []
    e_eval = []
    
    evaluationSize  = X_val_xgb.shape[0]
    for idx_eval in range(0, evaluationSize):

      if sortByResponseVar:
        response_distances = np.abs(y_train_xgb-y_val_xgb[idx_eval])
        idx_neighbours = np.argsort(response_distances)[0:self.num_neighbours]
      else:  
        idx_neighbours = np.random.choice(trainingSize, self.num_neighbours, replace=False)

      x_A = X_train_xgb[idx_neighbours]
      x_B = np.tile(X_val_xgb[idx_eval], (self.num_neighbours, 1))
      X_AB = np.concatenate([x_A, x_B], axis=1)
      M_eval.append(X_AB)

      y_BA = y_val_xgb[idx_eval] - y_train_xgb[idx_neighbours]
      e_eval.append(y_BA)

    M_eval = np.concatenate(M_eval, axis=0).copy()
    e_eval = np.concatenate(e_eval, axis=0).copy()

    eval_set = [(M_eval, e_eval)]
    self.eval_set = Pool(M_eval, e_eval)

    print(f'Training set {M.shape}. Evaluation {M_eval.shape}...', end='')

    # The fit() method varies a bit depending on the regressor
    if isinstance(self.regressor, ExtraTreesRegressor):
      self.regressor.fit(M, e)
    elif isinstance(self.regressor, NGBRegressor):
      self.regressor.fit(M, e, X_val=M_eval, Y_val=e_eval)
    else:
      self.regressor.fit(M, e, eval_set=self.eval_set, verbose=verbosity)
    print('done.')
    self.is_fitted = True

    # get the feature importances
    self.get_feature_importance()

    return self


  def get_feature_importance(self):
    # If ngboost, take the feature importance for loc (mu) trees
    
    if self.is_fitted:

      refVars = ['ref_' + iVar for iVar in self.inputVars]
      int_vars = self.inputVars.copy()
      int_vars.extend(refVars)
      self.int_vars = int_vars
      
      if isinstance(self.regressor, CatBoostRegressor):
        self.feat_importances = self.regressor.get_feature_importance()
      elif isinstance(self.regressor, NGBRegressor):
        self.feat_importances = getattr(self.regressor, self.feat_importance_keyword, None)[0]
      else:
        self.feat_importances = getattr(self.regressor, self.feat_importance_keyword, None) 

  
      self.x_weights = self.feat_importances[0:self.numFeatures]
      self.x_ref_weights = self.feat_importances[self.numFeatures::]
      self.x_combined_weights = self.x_weights + self.x_ref_weights

    return self.feat_importances


  def predict(self, X, categorical_mapping = None):
    
    # Scale the test set. Select wether to use the reference or 
    # just the train set weights
    doRefWeightsScaling = False
    doSymmetricalWeights = True

    #if self.is_fitted:
    X_test = X.copy()
    
    # Prepare for the Euclidean distance
    r_test   = X_test.shape[0]
    t_test   = self.X_train.shape[0]
    X_temp   = np.concatenate([X_test, self.X_train], axis=0)
    
    # Normalisation. If not z-score, normalise from 0 to 1
    if self.normalisation_type == 'z-score':
      x_normed = StandardScaler().fit_transform(X_temp)
    else:
      x_normed = (X_temp - X_temp.min(0)) / np.maximum(X_temp.ptp(0), 1)


    if doSymmetricalWeights:
      print('...Symmetrical Weights')
      # scale using Cholesky's so W = Q.T Q
      V = np.diag(self.x_combined_weights)
      try:
          Q = np.linalg.cholesky(V)
          weights_Q = np.diag(Q)
      except np.linalg.LinAlgError as err:
        print(err)
        doSymmetricalWeights = False

    if doRefWeightsScaling:
      X_test_scaled = np.multiply(x_normed[0:r_test, :], np.sqrt(self.x_ref_weights))
    elif doSymmetricalWeights:
      X_test_scaled = np.multiply(x_normed[0:r_test, :], weights_Q)
    else:
      X_test_scaled = np.multiply(x_normed[0:r_test, :], np.sqrt(self.x_weights))
    
    # Scale the training set
    if doSymmetricalWeights:
      X_train_scaled = np.multiply(x_normed[r_test:(r_test+t_test), :], weights_Q)
    else:
      X_train_scaled = np.multiply(x_normed[r_test:(r_test+t_test), :], np.sqrt(self.x_weights))

    y_k_all_list = []
    y_k_list = []
    y_k_weighted_list = []
    y_delta_list = []
    y_idx_closest_promos = []
    y_distances_closest_promos = []
    testSize = X_test.shape[0]

    for idx_test in range(0, testSize):

      # Select the closest promotions. Try scaling...
      current_promo_scaled = X_test_scaled[idx_test].reshape(1, -1)

      # >> Euclidean distances
      euclidean = np.squeeze(pairwise_distances(X_train_scaled, current_promo_scaled))
      idxSorted = np.argsort(euclidean)[0:self.num_neighbours]

      x_A = self.X_train[idxSorted]
      x_B = np.tile(X_test[idx_test], (self.num_neighbours, 1))
      X_AB_test = np.concatenate([x_A, x_B], axis=1)

      # differences regarding the reference promotions
      xgb_frc = self.regressor.predict(X_AB_test)

      # Get the average
      y_delta_list.append(xgb_frc)
      y_k_hat_all = xgb_frc + self.y_train[idxSorted]
      y_k_hat = np.mean(y_k_hat_all)

      # Weighted by the Euclidean distances
      w_distance = 1.0/np.maximum(euclidean[idxSorted], 1e-3)
      y_k_hat_distances = \
        w_distance.dot(y_k_hat_all.T)/np.sum(w_distance)
      y_k_weighted_list.append(y_k_hat_distances)
      
      # Append to the list
      y_k_all_list.append(y_k_hat_all)
      y_k_list.append(y_k_hat)
      y_idx_closest_promos.append(idxSorted)
      y_distances_closest_promos.append(euclidean[idxSorted])

    # Arrange the forecast as np-arrays
    y_hat = np.array(y_k_list)
    y_hat_weighted = np.array(y_k_weighted_list)
          

    # Arrange the outputs
    self.results = {'y_idx_closest_promos': y_idx_closest_promos, 
    'y_hat': y_hat,
    'y_hat_weighted': y_hat_weighted,
    'y_delta_list': y_delta_list,
    'y_k_all_list': y_k_all_list,
    'y_distances_closest_promos': y_distances_closest_promos,
    'feat_importances': self.feat_importances,
    'internal_var_names': self.int_vars}

    # Sort out feature importances
    if not doRefWeightsScaling:
      idx_importances = np.argsort(self.x_combined_weights)[::-1]
      int_var_names = self.int_vars[0:self.numFeatures]
    else:
      idx_importances = np.argsort(self.feat_importances)[::-1]
      int_var_names = self.int_vars

    # If provided
    if categorical_mapping:
      '''
        Get the feature importances as a DF
      '''
      inputVars_plain = [categorical_mapping.get(iVar, iVar) for iVar in 
          [int_var_names[this_idx] for this_idx in idx_importances]]
      df_feat_importances = pd.DataFrame(self.feat_importances[idx_importances], index=inputVars_plain)  
    else:
        inputVars_plain =[int_var_names[this_idx] for this_idx in idx_importances]
        df_feat_importances = pd.DataFrame(self.x_combined_weights[idx_importances], index=inputVars_plain)
    
    self.results['df_feat_importances'] = df_feat_importances
    return y_hat_weighted

  def predict_eval_test(self):
    if self.is_fitted:
      y_hat_weighted = self.predict(self.X_val)
      self.get_frc_errors(self.y_val, y_hat_weighted)

  def get_results(self):
    return self.results

  def arrange_regressor_results(self, idx_review, df_A_enc, y_train, id_train, inputVars_plain, \
    identifierVar, df_test_enc, y_test, num_inputVars):

    dm_frc = self.results

    idx_closest_promos = dm_frc['y_idx_closest_promos'][idx_review]

    df_forecast = df_A_enc.loc[idx_closest_promos, inputVars_plain]
    df_forecast['y_train'] = y_train[idx_closest_promos]
    df_forecast['delta_y_train'] = dm_frc['y_delta_list'][idx_review]
    df_forecast['y_train_plus_delta'] = dm_frc['y_k_all_list'][idx_review]
    df_forecast['y_train_distances'] = dm_frc['y_distances_closest_promos'][idx_review]

    if identifierVar:# and (id_train != None):
      df_forecast[identifierVar] = id_train[idx_closest_promos]

    #df_target = pd.DataFrame(X_test[idx_review], index = inputVars).T
    
    #df_target = df_test_enc.loc[idx_review, inputVars_plain]
    df_target = df_test_enc.iloc[idx_review][inputVars_plain]
    
    # Also pass the descriptor
    if identifierVar:
      df_target[identifierVar] = df_test_enc.iloc[idx_review][identifierVar]

    df_target['y_actual'] = y_test[idx_review]
    df_target['y_forecast'] = dm_frc['y_hat'][idx_review]
    df_target['y_weighted_forecast'] = dm_frc['y_hat_weighted'][idx_review]

    df_forecast_ext = df_forecast.append(df_target, sort=False)

    if 'df_feat_importances' in dm_frc.keys():
        df_feature_importances = dm_frc['df_feat_importances'].loc[inputVars_plain]
    else:
        df_feature_importances = pd.DataFrame(dm_frc['feat_importances'][0:num_inputVars], index=inputVars_plain)
    # Do not normalise
    # df_feature_importances = 100*df_feature_importances/df_feature_importances.sum()
    df_forecast_ext = df_forecast_ext.append(df_feature_importances.T, sort=False)

    return df_forecast_ext

  def get_frc_errors(self, y, y_hat, verbose=True):
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
      'MSE': MSE, 
      'RMSE': np.sqrt(MSE), 
      'meanError': meanError,
      'MAPE': MAPE,
      'R2': r2,
      'frc_error': frc_error,
      'frc_bias': frc_bias,
      'frc_acc': frc_acc,
      'Var explained': var_explained}
      #,'residuals': e_t}
      if verbose:
        for k,v in d.items():
          print(f'{k}: {v:3.2f}')
      return d