import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_scaler(feature_range=(0, 1)):
    return MinMaxScaler(feature_range=feature_range)

def scale_features(scaler, data, fit=True):
    if fit:
        return scaler.fit_transform(data)
    return scaler.transform(data)

def inverse_scale_price(scaler, scaled_price):
    dummy = np.array([[scaled_price, 0.0]])  
    return float(scaler.inverse_transform(dummy)[0, 0])