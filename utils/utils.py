import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit


# function to convert from amb to fraction of on beads
def amb_to_on(x):
  return 1-np.exp(-x)

def on_to_amb(f):
  return -np.log(1-f)

# geometric mean function that ignores NaNs by default
def geometric_mean_estimator(x, axis=0, nan_policy='omit'):
    return stats.gmean(x, axis=0, nan_policy='omit')

# 5PL model and its inverse
def log5pl(x, A, B, C, D, G):
  # 5PL model: x is expected to be in log space, and y is returned in log space
  # G is the asymmetry parameter (when G=1, this reduces to 4PL)
  return np.log(D + (A - D) / ((1.0 + (np.exp(x) / C) ** B) ** G))


def linear_5pl(x, A, B, C, D, G):
  # 5PL model: x is expected to be in linear space, and y is returned in linear space
  # G is the asymmetry parameter (when G=1, this reduces to 4PL)
  return D + (A - D) / ((1.0 + (x / C) ** B) ** G)

# Inverse of the 5PL model
def invlog5pl(y, superparams):
  # y is in linear space and x is returned in linear space
  params, conclimit, amblimit = superparams
  A, B, C, D, G = params[0], params[1], params[2], params[3], params[4]
  
  # Inverse 5PL formula
  logx = np.log(C) + (1/B) * np.log(((A - D) / (y - D)) ** (1/G) - 1)
  x = np.exp(logx)
  # Apply limits
  x = np.where(y < A, conclimit[0], x)
  #x = np.where(y > D, conclimit[1], x)
  return x

def inv5pl_basic(y, superparams):
  # y is in linear space and x is returned in linear space
  params, conclimit, amblimit = superparams
  A, B, C, D, G = params[0], params[1], params[2], params[3], params[4]
  
  # Inverse 5PL formula
  x = C * (((A - D) / (y - D)) ** (1/G) - 1) ** (1/B)
  
  # Apply limits
  x = np.where(y < A, conclimit[0], x)
  x = np.where(y > D, conclimit[1], x)
  return x


def invlog5pl_after_lod(y, superparams):
  # y is in linear space and x is returned in linear space
  params, conclimit, amblimit = superparams
  A, B, C, D, G = params[0], params[1], params[2], params[3], params[4]
  
  # Inverse 5PL formula
  logx = np.log(C) + (1/B) * np.log(((A - D) / (y - D)) ** (1/G) - 1)
  x = np.exp(logx)
  
  # Apply limits based on amblimit
  x = np.where(y < amblimit[0], conclimit[0], x)
  x = np.where(y > amblimit[1], conclimit[1], x)
  
  
  return x

def inv5pl_after_lod(y, superparams):
  # y is in linear space and x is returned in linear space
  params, conclimit, amblimit = superparams
  A, B, C, D, G = params[0], params[1], params[2], params[3], params[4]
  
  # Inverse 5PL formula
  x = C * (((A - D) / (y - D)) ** (1/G) - 1) ** (1/B)
  
  # Apply limits based on amblimit
  x = np.where(y < amblimit[0], conclimit[0], x)
  x = np.where(y > amblimit[1], conclimit[1], x)
  
  return x



def cal_curve_fit_basic(df, x_var='conc', y_var='amb'):
  df = df[~(df[x_var].isna())]
  df = df[~(df[y_var].isna())]
  # Filter out rows where 'conc' or 'amb' are zero
  df = df[~(df[x_var]==0.0)]
  df = df[~(df[y_var]==0.0)]
  x = df[x_var]
  y = df[y_var]
  #x = np.log(x)
  #y = np.log(y)
  
  # Adapt initial parameter estimation based on data range
  y_min, y_max = y.min(), y.max()
  x_mid = np.exp((np.log(x.min()) + np.log(x.max())) / 2)
  
  # Estimate initial parameters from data for 5PL
  A = y_min  # Bottom asymptote
  D = y_max  # Top asymptote  
  C = x_mid  # Inflection point (middle concentration)
  B = 1.0           # Hill slope
  G = 1.0           # Asymmetry parameter (start with symmetric case)
  
  p0 = [A, B, C, D, G]

  sigma_data = np.abs(y)  # Example: Poisson-like noise model
  
  params, _ = curve_fit(linear_5pl, x, y, p0=p0, maxfev=100000, sigma=sigma_data, absolute_sigma=False)
  A, B, C, D, G = params[0], params[1], params[2], params[3], params[4]
  
  x_min, x_max = x.min(), x.max()
  x_range = np.logspace(np.log(x_min), np.log(x_max),base=np.e)
  yfit = [linear_5pl(x, A, B, C, D, G) for x in x_range]
  return x_range, yfit, params

def cal_curve_fit(df, x_var='conc', y_var='amb'):
  df = df[~(df[x_var].isna())]
  df = df[~(df[y_var].isna())]
  # Filter out rows where 'conc' or 'amb' are zero
  df = df[~(df[x_var]==0.0)]
  df = df[~(df[y_var]==0.0)]
  x = df[x_var]
  y = df[y_var]

  _x=x.copy()
  _y=y.copy()
  
  x = np.log(x)
  y = np.log(y)
  
  # Adapt initial parameter estimation based on data range
  y_min, y_max = y.min(), y.max()
  x_mid = (x.min() + x.max()) / 2
  
  # Estimate initial parameters from data for 5PL
  A = _y.min()  # Bottom asymptote
  D = _y.max()  # Top asymptote
  C = np.exp(x_mid)  # Inflection point (middle concentration)
  B = 1.0           # Hill slope
  G = 1.0           # Asymmetry parameter (start with symmetric case)
  
  p0 = [A, B, C, D, G]
  
  params, _ = curve_fit(log5pl, x, y, p0=p0, maxfev=100000, xtol=1e-03)#, ftol=1e-03, xtol=1e-08, gtol=0)
  A, B, C, D, G = params[0], params[1], params[2], params[3], params[4]
  
  x_min, x_max = x.min(), x.max()
  x_range = np.linspace(x_min, x_max)
  yfit = [log5pl(x, A, B, C, D, G) for x in x_range]
  return np.exp(x_range), np.exp(yfit), params