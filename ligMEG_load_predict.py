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
# define the cn and the nn xyz files for the test complex
cn_xyz = [111]
nn_xyz = [6]
cn_structure = [list(pybel.readfile(format='xyz', filename=curr_dir + '/ligs_xyz/cn_'+str(x)+'.xyz'))[0] for x in cn_xyz]
nn_structure = [list(pybel.readfile(format='xyz', filename=curr_dir + '/ligs_xyz/nn_'+str(x)+'.xyz'))[0] for x in nn_xyz]

from megnet.data.molecule import MolecularGraph
from megnet.data.molecule import SimpleMolGraph
from megnet.models import MEGNetModel

mg = MolecularGraph()

from tensorflow.keras.models import load_model
from megnet.layers import _CUSTOM_OBJECTS

# load model
model = load_model(curr_dir + '/ligsMEG_model.hdf5', custom_objects=_CUSTOM_OBJECTS)

YT_pred = []

# generate 100 predictions
for j in range(100):
    cn_graph = mg.convert(cn_structure[0], full_pair_matrix=False)
    nn_graph = mg.convert(nn_structure[0], full_pair_matrix=False)
    cn_inp = mg.graph_to_input(cn_graph)
    nn_inp = mg.graph_to_input(nn_graph)
    YT_pred.append(np.round(model.predict([(cn_inp,nn_inp)])[0,0].ravel()[0:46], 2).tolist())

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
