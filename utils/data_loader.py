import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_psm_data(data_dir="C:/Dataset/PSM"):
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    test_df = pd.read_csv(f"{data_dir}/test.csv")
    label_df = pd.read_csv(f"{data_dir}/test_label.csv", header=None)
    
    X_train = train_df.drop(columns=["timestamp_(min)"]).values
    X_test = test_df.drop(columns=["timestamp_(min)"]).values
    y_test = label_df.values.squeeze()
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_test
