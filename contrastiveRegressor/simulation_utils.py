'''
  Snippets to simulate students
'''

import pandas as pd
import random
import numpy as np

def generate_students_data(num_samples = 100, rho = 0.65):
  '''
    Generate correlated study_time/score data + noise
  '''
  # this is in seconds
  study_time_mu = 300
  study_time_sigma = 80

  student_score_mu = 50
  student_score_sigma = 20

  # 1 - Correlation matrix
  corr_mat = np.array([[1.0, rho],
                      [rho, 1.0]])
  # 2 - Decompose
  # Compute the (upper) Cholesky decomposition matrix
  L = np.linalg.cholesky(corr_mat)

  # 2.b - For a pair of signals, L is quite simple
  # if rho is the desired correlation
  # L = np.array([[1,rho], [0,np.sqrt(1-rho**2)]])
  L = np.array([[1,rho], [0,np.sqrt(1-rho**2)]])

  # 3 - Define random gaussian signals
  noise = np.random.normal(0.0, 1.0, size=(num_samples, 2))
  correlated_signals = np.matmul(noise, L)

  # 4 - Scale the signals
  timeSpent = correlated_signals[:,0]*study_time_sigma + study_time_mu
  score = correlated_signals[:,1]*student_score_sigma + student_score_mu


  df = pd.DataFrame({
      "noise_1" : np.random.randint(0, 100, size=(num_samples)),
      "noise_2" : np.random.randint(75, 900, size=(num_samples)),
      "timeSpent" : timeSpent,
      "score": score
  })  
  return df