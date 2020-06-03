import fcn_helpers as fhelp
from frc_runner import frc_runner
import pandas as pd


def run_comparison(experiment_label, baseFolder, baseFile, \
  base_product_number_std, \
  categoricalVars, numericalVars, \
  num_neighbours, validation_test_size, \
  num_iterations, learning_rate, depth):
  ''' 
    
    Helper for the paper to run the same tests with different products

  '''


  # Hyper-parameters
  feat_importance_keyword = 'feature_importances_'


  # Read and filter
  df_promos = fhelp.BPNS_reader(baseFolder, baseFile, base_product_number_std)

  sales_threshold = 20
  stores_threshold = 10
  idx_A = df_promos.wk1_sales_all_stores > sales_threshold 
  idx_B = df_promos.total_stores > stores_threshold
  df_promos = df_promos[idx_A & idx_B].copy()


  frc_results = frc_runner(df_promos, base_product_number_std,
    categoricalVars, numericalVars,
    num_neighbours, validation_test_size,
    num_iterations, learning_rate, depth,
    feat_importance_keyword = feat_importance_keyword,
    experiment_label = experiment_label,
    doVisualisation=False,
    doSaveExcel = False)

  # Add the random neighbours and the plain xgboost
  contrastiveReg = frc_results['contrastiveRegressor']
  cold_start_results = frc_results['cold_start_results']

  # Predict using random neighbours
  y_hat_random = fhelp.frc_with_random_neighbours(contrastiveReg.X_train, frc_results['X_test'], \
    contrastiveReg.num_neighbours, contrastiveReg)

  y_hat_catboost = fhelp.frc_plain_CatBoost(contrastiveReg.num_neighbours, validation_test_size,
      num_iterations, learning_rate, depth, \
      contrastiveReg.X_train, contrastiveReg.y_train, frc_results['X_test'])

  autoGluon = fhelp.frc_AutoGluon(frc_results['df_train'], frc_results['df_test'], \
      categoricalVars, experiment_label=experiment_label)

  # Arrange the results in a DF so we can easily plot them
  df_results = pd.DataFrame({'experiment_label': experiment_label,
  'num_neighbours': contrastiveReg.num_neighbours,
  'y_actual': frc_results['y_test'],
  'y_hat': cold_start_results['y_hat'], 
  'y_hat_weighted': cold_start_results['y_hat_weighted'], 
  'y_hat_random': y_hat_random,
  'y_hat_catboost': y_hat_catboost,
  'y_hat_autoGluon': autoGluon['autoGluon_frc']})

  df_all_autogluon = pd.DataFrame(autoGluon['individual_frc'])
  df_results = pd.concat([df_results, df_all_autogluon], axis=1)

  return df_results