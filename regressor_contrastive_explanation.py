'''

  Use a XGBoost model to produce what the NextDoorNeighboor does
  
  Arrange the training promotions as |X|X_ref|

'''

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances


def dm_regressor(regressor, inputVars, X_train, y_train, X_test, 
  training_split=0.25, num_neighbours = 5, validation_test_size=0.2, 
  feat_importance_keyword = 'feature_importances_'):
  '''

    Use a regressor (XGBoost model) to produce what the NextDoorNeighboor does
    
    Arrange the training promotions as |X_train|X_ref| and the target as |y_train-y_ref|

    Updates:
    01.11.2019

  '''


  '''
    Part 1

    Arrange the contrastive explainer using a given model
    (explicit differences model)
  '''
  # Number of promotions used to compare to
  numRecords  = X_train.shape[0]
  numFeatures = X_train.shape[1]
  trainingSize = round(numRecords*training_split)
  idx_test = np.random.choice(numRecords, trainingSize, replace=False)

  M = []
  e = []
  currentPromo    = np.zeros(numRecords, dtype=bool)
  remainingPromos = np.ones(numRecords, dtype=bool)

  for k in range(0, trainingSize):
    idx = idx_test[k]
    currentPromo[idx]    = True
    remainingPromos[idx] = False
    idx_neighbours = np.random.choice(numRecords-1, num_neighbours, replace=False)

    x_A = X_train[remainingPromos][[idx_neighbours]]
    x_B = np.tile(X_train[currentPromo], (num_neighbours, 1))
    X_AB = np.concatenate([x_A, x_B], axis=1)
    M.append(X_AB)

    y_BA = y_train[currentPromo] - y_train[remainingPromos][[idx_neighbours]]
    e.append(y_BA)

    currentPromo[idx]    = False
    remainingPromos[idx] = True

  M = np.concatenate(M, axis=0).copy()
  e = np.concatenate(e, axis=0).copy()
    

  # This is tailored to XGBoost tbh
  X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(M, e, test_size=validation_test_size)
  eval_set = [(X_val_xgb, y_val_xgb)]
  regressor.fit(X_train_xgb, y_train_xgb, eval_set=eval_set, verbose=False)



  '''
    Part 2
    Neighbourhood selection: 
    Set the weights of the variables to the feature importance calculated during training
  '''
  refVars = ['ref_' + iVar for iVar in inputVars]
  xgb_vars = inputVars.copy()
  xgb_vars.extend(refVars)

  feat_importances = getattr(regressor, feat_importance_keyword, None)


  # Take the weights of the train matrix ignoring the reference matrix
  x_weights = feat_importances[0:numFeatures]

  # Scale the test set
  X_test_scaled = np.multiply(X_test, x_weights)
  # Scale the training set
  X_train_scaled = np.multiply(X_train, x_weights)


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
    idxSorted = np.argsort(euclidean)[0:num_neighbours]

    x_A = X_train[idxSorted]
    x_B = np.tile(X_test[idx_test], (num_neighbours, 1))
    X_AB_test = np.concatenate([x_A, x_B], axis=1)

    # differences regarding the reference promotions
    xgb_frc = regressor.predict(X_AB_test)

    # Get the average
    y_delta_list.append(xgb_frc)
    y_k_hat_all = xgb_frc + y_train[idxSorted]
    y_k_hat = np.mean(y_k_hat_all)

    # Weighted by the Euclidean distances
    y_k_hat_distances = \
      euclidean[idxSorted].dot(y_train[idxSorted].T)/np.sum(euclidean[idxSorted])
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
  results = {'y_idx_closest_promos': y_idx_closest_promos, 
  'y_hat': y_hat,
  'y_hat_weighted': y_hat_weighted,
  'y_delta_list': y_delta_list,
  'y_k_all_list': y_k_all_list,
  'y_distances_closest_promos': y_distances_closest_promos,
  'feat_importances': feat_importances}

  return results