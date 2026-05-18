import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_scaler(feature_range=(0, 1)):
    return MinMaxScaler(feature_range=feature_range)

def scale_features(scaler, data, fit=True):