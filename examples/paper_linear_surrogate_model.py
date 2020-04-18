'''

  Experiment #1 of the paper

  Linear surrogate model >> Analysis of the variable importance
  
  Generate a linear model where:
  - all the input variables are within the same range
  - the response variable is a linear combination where the weights are known.

  Updates:
    14.03.2020
    16.03.2020 - Add Excel exporter
    10.04.2020 - Write down the experiment in the code.

  Runner:
  python3 paper_linear_surrogate_model.py

'''


'''

  Analysis of the variable importance. 
  To generate contrastive explanations we rearrange the data into neighbours and reference. Therefore, the variable importance 
  calculated by the GBDT regressor is divided into parts. Asumming that data do not have date or time variables, the importance
  vector can be written as $\mathbf{v} = [\mathbf{v^{neig}}, \mathbf{v^{ref}}]$.

  In this experiment we demonstrate that the method is able to accurately calculate the variable importance and moreover we
  can rearrange the importances as $\mathbf{v'}= \mathbf{v^{neig}} + \mathbf{v^{ref}}$ to facilitate interpretability.

  To demonstrate this, let us generate a linear model with 5 independent variables drawn from drawn from a uniform distribution $\textit{U}(0,1)$ and 500 samples.

  500 samples with

    Linear model (no noise).
    500 samples with 5 independent variables.

  $\mathbf{X} \in \mathbb{R}^{500 \times 5}$

    Dependant variable is a linear combination
    $\mathbf{w} = [13,9,6,1,0]$

    The variables of the model are the following: variable $\mathbf{x_1}$ is drawn from the standard uniform distribution $\textit{U}(0,1)$

    \begin{equation}
    \mathbf{w}
    \label{eqn:eq8}
    \end{equation}

'''


import contrastiveRegressor.fcn_helpers as fhelp
import pandas as pd
import numpy as np
from contrastiveRegressor.contrastiveRegressor import contrastiveRegressor
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool
import contrastiveRegressor.preprocessing_utils as pt
from os import path as _p
import datetime as dt





# Fake sales
experiment_label = 'linear_model'
num_samples = 500
num_features = 5
input_vars = [f'x_{idx}' for idx in range(1,num_features+1)]
input_data = np.random.rand(num_samples, num_features)

weights = np.array([42,34,16,0,8])
y_train = np.dot(input_data, weights.T)

df = pd.DataFrame(input_data, columns=input_vars)
#df['response_var'] = response_var
#noisy_sales = 10 * np.random.random(size=(num_samples)) + x_3_values


# Ad-hoc test set to see the influence of the variables
df_test = pd.DataFrame([{'x_1': 0.1, 'x_2': 0.5, 'x_3': 0.5, 'x_4': 0.5, 'x_5': 0.5},
{'x_1': 0.9, 'x_2': 0.5, 'x_3': 0.5, 'x_4': 0.5, 'x_5':0.5 },
{'x_1': 0.5, 'x_2': 0.9, 'x_3': 0.5, 'x_4': 0.5, 'x_5':0.5 },
{'x_1': 0.5, 'x_2': 0.1, 'x_3': 0.5, 'x_4': 0.5, 'x_5':0.5 }])
y_actual = np.dot(df_test.values, weights.T)


numericalVars = input_vars
categoricalVars = []

num_inputVars = len(input_vars)

# Hyper-parameters
num_neighbours = 5
validation_test_size = 0.20
feat_importance_keyword = 'feature_importances_'

# Regressor
num_iterations = 300
learning_rate  = 0.08 
depth = 12





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
contrastiveReg.set_regressor(cb_model, feat_importance_keyword, input_vars)
# fit the regressor
contrastiveReg.fit(df.values, y_train)
# eval results
contrastiveReg.predict_eval_test()
eval_results = contrastiveReg.get_results()

# Predict
contrastiveReg.predict(df_test.values)

cold_start_results = contrastiveReg.get_results()
print(cold_start_results['feat_importances'])

y_forecast = cold_start_results['y_hat_weighted']
contrastiveReg.get_frc_errors(y_actual, y_forecast)


# Feat importance
refVars = ['ref_' + iVar for iVar in input_vars]
int_vars = input_vars.copy()
int_vars.extend(refVars)
#
feat_importances = cold_start_results['feat_importances']
df_feature_importances = pd.DataFrame(feat_importances, index=int_vars)
print(df_feature_importances)
str_latex = df_feature_importances.to_latex(index=False, float_format='{:3.2f}'.format)
print(str_latex)


# Optimal weights
ideal_W = 50*weights/weights.sum()
df_weights = pd.DataFrame(np.concatenate([ideal_W, ideal_W], axis=0), index=int_vars)
print(df_weights)

# Actual weights
actual_W = 100*weights/weights.sum()
df_actual_weights = pd.DataFrame(actual_W, index=int_vars[0:len(actual_W)])
print(df_actual_weights)

# Operate on the added weights
list(contrastiveReg.x_combined_weights)


# scale using Cholesky's
V = np.diag(contrastiveReg.x_combined_weights)
Q = np.linalg.cholesky(V)
weights_to_use = np.diag(Q)
#np.dot(Q, Q.T)



# Predict using random neighbours
y_hat_random = fhelp.frc_with_random_neighbours(contrastiveReg.X_train, df_test.values, \
  contrastiveReg.num_neighbours, contrastiveReg)

y_hat_catboost = fhelp.frc_plain_CatBoost(contrastiveReg.num_neighbours, validation_test_size,
    num_iterations, learning_rate, depth, \
    contrastiveReg.X_train, contrastiveReg.y_train, df_test.values)


all_cold_forecast = []

for idx_review in range(y_actual.shape[0]):
  print('Running {idx_review}...')
  df_forecast_ext = contrastiveReg.arrange_regressor_results(idx_review, df, \
  y_train, None, input_vars, \
  None, df_test, y_actual, num_inputVars)

  df_forecast_ext['y_hat_random'] = ''
  df_forecast_ext['y_hat_catboost'] = ''

  df_forecast_ext.reset_index(inplace=True)
  
  '''
  df_forecast_ext['y_train'].iloc[-2] = \
    df_forecast_ext['y_weighted_forecast'].iloc[-2]
  '''
  df_forecast_ext['y_hat_random'].iloc[-2] = y_hat_random[idx_review]
  df_forecast_ext['y_hat_catboost'].iloc[-2] = y_hat_catboost[idx_review]

  print(df_forecast_ext)
  all_cold_forecast.append(df_forecast_ext)

  
  y_actual_A = df_forecast_ext['y_actual'].iloc[-2]
  y_forecast = df_forecast_ext['y_weighted_forecast'].iloc[-2]
  print(f'(actual: {y_actual_A:3.2f}, forecast: {y_forecast:3.2f})')

# Append them all
df_all_cold_forecast = pd.concat(all_cold_forecast)

print(df_all_cold_forecast)

prefix = experiment_label + '_' + dt.datetime.today().strftime('%d_%m_%Y_%HH')
xlsx_file_name = _p.join('results', prefix + '_table.xlsx')
fhelp.to_excel_file(df_all_cold_forecast, xlsx_file_name)