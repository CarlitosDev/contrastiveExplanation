'''
  Generate sales for toy example (Experiment #2)
  
  carlos.aguilar.palacios@gmail.com
'''
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from sklearn.preprocessing import MinMaxScaler
import pandas as pd



def Gompertz_distribution(t=np.linspace(0, 5, 50), b_scale=1, eta_shape=0.2):
  # https://www.wikiwand.com/en/Gompertz_distribution
  return b_scale*eta_shape*np.exp(eta_shape)*np.exp(b_scale*t)*np.exp(-eta_shape*np.exp(b_scale*t))


def shifted_Gompertz_distribution(t=np.linspace(0, 5, 50), b_scale=0.4, eta_shape=10):
  # https://www.wikiwand.com/en/Shifted_Gompertz_distribution
  e_bt = np.exp(-b_scale*t)
  return b_scale*e_bt*np.exp(-eta_shape*e_bt)*(1+eta_shape*(1-e_bt))

def generate_Gompertz_sales(num_samples,
  mu_sales, sigma_sales,
  price_mu,
  discount_sigma, 
  shelf_capacity, shelf_impact,
  b_scale, eta_shape):
  '''Generate sales for toy example.
  The discount effect on the sales is modelled as a shifted 
  Gompertz distribution.
  The baseline sales are modelled as a Gaussian curve.


  Parameters
  ----------
  num_samples: integer
  mu_sales: float
  sigma_sales: float
  discount_sigma: float
  shelf_capacity: integer
  shelf_impact: float
  
  '''

  # 1 - Baseline sales
  baseline_sales = np.random.normal(mu_sales, sigma_sales, num_samples)


  # 3 - Discount
  # Let's assume that we are working with price cut promos, so the discount
  # is always GE 0
  # Half-normal distribution
  discount_mu = 0.0
  discount = np.abs(np.random.normal(discount_mu, discount_sigma, num_samples))
  price    = price_mu - discount

  # 4 - Sales-gain due to the discount
  # The impact of the price/discount on sales is modelled as a shifted Gompertz response.
  # Tweak the Gompertz dist and scale the baseline sales
  sales_response = lambda t: 1.0+(2.2*shifted_Gompertz_distribution(t, b_scale, eta_shape))
  sales_driven_by_discount = np.multiply(sales_response(discount), baseline_sales)


  # 4- Shelves
  # Easy-peasy: A curve to map the increment per shelf
  
  # This is fixed up to a 12% gain. This value is then multiplied by the shelf_impact
  max_gain = 0.12
  min_shelf_size = 0.0
  max_shelf_size = 25

  # Randomly pick the capacities from the list
  shelves = np.random.choice(shelf_capacity, num_samples, replace=True)
  
  
  shelves_gain = shelf_impact*max_gain*(shelves-min_shelf_size) \
    / (max_shelf_size-min_shelf_size)
  shelf_sales = np.multiply(shelves_gain, sales_driven_by_discount)

  product_sales = sales_driven_by_discount + shelf_sales

  # Stick the data into a DF
  df = pd.DataFrame({'price': price,
  'discount': discount,
  'baseline_sales': baseline_sales,
  'shelves': shelves,
  'shelf_sales': shelf_sales,
  'product_sales': product_sales})

  return df
