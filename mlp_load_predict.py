import pandas as pd
import os, sys
import numpy as np
import openbabel
import math
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import pybel  # type: ignore
except ImportError:
    try:
        from openbabel import pybel
    except ImportError:
        pybel = None

curr_dir = os.getcwd()
xdata = pd.read_csv(curr_dir + '/csv_files/lig_prop_data.csv')
ydata1 = pd.read_csv(curr_dir + '/csv_files/IrCNNN_smiles_wts.csv')
ydata = pd.read_csv(curr_dir + '/csv_files/Spectra_46_intensities.csv')

# merging common x-data and y-data
ydata = ydata.merge(ydata1[['Complex_i', 'TANH_550_100']], how='inner', on='Complex_i')
ydata = ydata.merge(xdata, how='inner', on='Complex_i')

# extracting x-data
xdata = ydata[[ 'cn_IE', 'cn_EA', 'cn_ST', 'cn_mu', 'cn_alpha', 'cn_HOMO', 'cn_LUMO', 'cn_gap', 'cn_R2', 'cn_ZPVE', 'cn_U0', 'cn_U', 'cn_H', 'cn_G', 'cn_Cv', 'cn_w1', 'nn_IE', 'nn_EA', 'nn_ST', 'nn_mu', 'nn_alpha', 'nn_HOMO', 'nn_LUMO', 'nn_gap', 'nn_R2', 'nn_ZPVE', 'nn_U0', 'nn_U', 'nn_H', 'nn_G', 'nn_Cv', 'nn_w1']]
xdata = xdata.values
clf_x = StandardScaler()
xdata = clf_x.fit_transform(xdata)

# define the csv file containing the ligand features for the test complex and scale data to the training set
xnew_data = pd.read_csv(curr_dir + '/csv_files/CN111_NN6.csv')
xnew_data = xnew_data.drop(['Complex_i'], axis=1)
xnew1 = xnew_data.values
X1T = clf_x.transform(xnew1)

from tensorflow.keras.models import load_model

# load model
model = load_model(curr_dir + '/new_mlp_model.hdf5')

YT_pred = []

# generate 100 predictions
for j in range(100):
    YT_pred = (np.round(model.predict(X1T), 2).tolist())

Ypred_array = np.array(YT_pred)
Ypred_avg = np.mean(Ypred_array, axis=0)
Ypred_std = np.std(Ypred_array, axis=0)

# Perform min-max scaling
min_val = np.min(Ypred_avg)
min_val = min(0.0, min_val)
max_val = np.max(Ypred_avg)

Ypred = np.round((Ypred_avg - min_val) / (max_val - min_val), 2)


print("Mean prediction: ", Ypred)
print("Prediction std dev: ", Ypred_std)
