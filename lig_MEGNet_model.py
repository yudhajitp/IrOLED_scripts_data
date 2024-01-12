import pandas as pd
import os, sys
import numpy as np
import openbabel
import math
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def weight_calc_metrics(y_true, y_pred, yweights, nfeatures = None):
    """
    calculates metrics to evaluate regression models

    Parameters
    ----------
    y_true: actual values (type = list, or 1D array)
    y_pred: predicted values (type = list, or 1D array)
    nfeatures: number of features required to calculate adjusted R squared (type = integer, default=None)

    Returns
    -------
    metrics_dict: dictionary with all metrics
    """
    metrics_dict = {}

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_weights = np.asarray(yweights)
    y_true_mean = np.average(y_true, weights=y_weights, axis=0)

    e = y_true - y_pred
    ae = np.absolute(e)
    se = np.square(e)
    var = np.average(np.square(y_true - y_true_mean),weights=y_weights, axis=0)

    # weighted mean absolute error: absolute difference between the predicted value and observed value
    metrics_dict['WMAE'] = np.mean(np.average(ae, weights=y_weights, axis=0))

    # weighted mean squared error: squared difference between the predicted value and observed value
    wmse = np.average(se, weights=y_weights, axis=0)
    metrics_dict['WMSE'] = np.mean(wmse)

    # weighted root mean squared error: square error of the mse
    # used when it is desired to penalize the higher differences more, sensitive to outliers, minimizing squared error over a set of numbers results in finding its mean
    metrics_dict['WRMSE'] = np.mean(np.sqrt(np.average(se, weights=y_weights, axis = 0)))

    #  weighted R squared and adjusted R squared: explain how well features capture the variation in target properties
    r2 = 1 - (np.mean(wmse))/(np.mean(var))
    metrics_dict['r_squared'] = r2

    return metrics_dict

# import pybel to convert molecular data between formats
try:
    import pybel  # type: ignore
except ImportError:
    try:
        from openbabel import pybel
    except ImportError:
        pybel = None

# import molecular and graph data structures and basic model architecture from MEGNet
from megnet.data.molecule import MolecularGraph
#from megnet.data.graph import StructureGraph
from megnet.models import MEGNetModel

# input data
curr_dir = os.getcwd()
xdata = pd.read_csv(curr_dir + '/csv_files/lig_prop_data.csv')
ydata1 = pd.read_csv(curr_dir + '/csv_files/IrCNNN_smiles_wts.csv')
ydata = pd.read_csv(curr_dir + '/csv_files/Spectra_46_intensities.csv')
ydata = ydata.merge(ydata1, how='inner', on='Complex_i')

# merging common x-data and y-data. We will not be using the x-data to train, but just to use consistent training and testing sets as the mlp_model script
ydata = ydata.merge(xdata, how='inner', on='Complex_i')

# input ligand structure information
cn_structures = [list(pybel.readfile(format='xyz', filename=curr_dir + '/ligs_xyz/cn_'+str(x)+'.xyz'))[0] for x in ydata['CN']]
nn_structures = [list(pybel.readfile(format='xyz', filename=curr_dir + '/ligs_xyz/nn_'+str(x)+'.xyz'))[0] for x in ydata['NN']]

# using SMILES strings as input
#cn_structures = [pybel.readstring("smi", x) for x in ydata['CN_smiles']]
#nn_structures = [pybel.readstring("smi", x) for x in ydata['NN_smiles']]

print(len(cn_structures))
print(len(nn_structures))
#sys.exit("Done")

data_weights = ydata['TANH_550_100'].tolist()

# extracting the y-data
ydata = ydata.drop(['Complex_i','CN','NN','CN_smiles','NN_smiles','TANH_550_100','cn_IE', 'cn_EA', 'cn_ST', 'cn_mu', 'cn_alpha', 'cn_HOMO', 'cn_LUMO', 'cn_gap', 'cn_R2', 'cn_ZPVE', 'cn_U0', 'cn_U', 'cn_H', 'cn_G', 'cn_Cv', 'cn_w1', 'nn_IE', 'nn_EA', 'nn_ST', 'nn_mu', 'nn_alpha', 'nn_HOMO', 'nn_LUMO', 'nn_gap', 'nn_R2', 'nn_ZPVE', 'nn_U0', 'nn_U', 'nn_H', 'nn_G', 'nn_Cv', 'nn_w1'], axis=1)
targets = ydata.values.tolist()

# random splitting of the dataset into 80:20 train-test splits, splitting the indices
train_indices, test_indices = train_test_split(list(range(len(targets))), test_size=len(targets)-(math.floor(0.8*len(targets)/5)*5), random_state=17)

print("Train set: ", len(train_indices))
print("Test set: ", len(test_indices))
#print("Train set: ", len(train_indices))
#print("Test set: ", len(test_indices))
#sys.exit()

cn_train_graphs = []
nn_train_graphs = []
cn_test_graphs = []
nn_test_graphs = []
Y = []
Y_wt = []
YT = []
YT_wt = [] 

# defining a base MEGNet model which will be used as the basis to generate input data structures for our new model
model = MEGNetModel(20,2,1, nblocks=3, lr=0.01, n1=16, n2=16, n3=16, batch_size = len(train_indices), npass=1, ntarget=46, graph_converter=MolecularGraph())

# extracting cn and nn graphs for training set
for i in train_indices:
    cn_graph = model.graph_converter.convert(cn_structures[i], full_pair_matrix=False)
    nn_graph = model.graph_converter.convert(nn_structures[i], full_pair_matrix=False)
    #list_desc = (xdata[i].tolist())
    #graph["state"] = [(list_desc)]
    cn_train_graphs.append(cn_graph)
    nn_train_graphs.append(nn_graph)
    Y.append(targets[i])
    Y_wt.append(data_weights[i])

# extracting cn and nn graphs for testing set
for i in test_indices:
    cn_graph = model.graph_converter.convert(cn_structures[i], full_pair_matrix=False)
    nn_graph = model.graph_converter.convert(nn_structures[i], full_pair_matrix=False)
    #list_desc = (xdata[i].tolist())
    #graph["state"] = [(list_desc)]
    cn_test_graphs.append(cn_graph)
    nn_test_graphs.append(nn_graph)
    YT.append(targets[i])
    YT_wt.append(data_weights[i])

train_nb_atoms = [len(i["atom"]) for i in cn_train_graphs]
train_targets = [model.target_scaler.transform(i, j) for i, j in zip(Y, train_nb_atoms)]

# creating input data from the graphs extracted earlier
cn_train_inputs = model.graph_converter.get_flat_data(cn_train_graphs, train_targets)
cn_train_generator = model._create_generator(*cn_train_inputs, sample_weights=Y_wt, batch_size=len(Y), is_shuffle=False)
nn_train_inputs = model.graph_converter.get_flat_data(nn_train_graphs, train_targets)
nn_train_generator = model._create_generator(*nn_train_inputs, sample_weights=Y_wt, batch_size=len(Y), is_shuffle=False)

# importing tensorflow/keras and MEGNet functions for building our new model
from tensorflow.keras.layers import Add, Concatenate, Dense, Dropout, Embedding, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from typing import Callable

from megnet.activations import softplus2
from megnet.config import DataType
from megnet.layers import GaussianExpansion, MEGNetLayer, Set2Set

# function to define the new MEGNet-based model 
def make_merged_model(
    nfeat_edge: int = 20,
    nfeat_global: int = 32,
    nfeat_node: int = 1,
    nblocks: int = 3,
    n1: int = 64,
    n2: int = 32,
    n3: int = 16,
    nvocal: int = 95,
    embedding_dim: int = 16,
    #nbvocal: int = None,
    #bond_embedding_dim: int = None,
    #ngvocal: int = None,
    #global_embedding_dim: int = None,
    npass: int = 3,
    ntarget: int = 1,
    act: Callable = softplus2,
    is_classification: bool = False,
    #l2_coef: float = None,
    dropout: float = 0.25,
    dropout_on_predict: bool = False,
    **kwargs,
) -> Model:
    """Make a MEGNet Model
    Args:
        nfeat_edge: (int) number of bond features
        nfeat_global: (int) number of state features
        nfeat_node: (int) number of atom features
        nblocks: (int) number of MEGNetLayer blocks
        n1: (int) number of hidden units in layer 1 in MEGNetLayer
        n2: (int) number of hidden units in layer 2 in MEGNetLayer
        n3: (int) number of hidden units in layer 3 in MEGNetLayer
        nvocal: (int) number of total element
        embedding_dim: (int) number of embedding dimension
        nbvocal: (int) number of bond types if bond attributes are types
        bond_embedding_dim: (int) number of bond embedding dimension
        ngvocal: (int) number of global types if global attributes are types
        global_embedding_dim: (int) number of global embedding dimension
        npass: (int) number of recurrent steps in Set2Set layer
        ntarget: (int) number of output targets
        act: (object) activation function
        l2_coef: (float or None) l2 regularization parameter
        is_classification: (bool) whether it is a classification task
        dropout: (float) dropout rate
        dropout_on_predict (bool): Whether to use dropout during prediction and training
        kwargs (dict): in the case where bond inputs are pure distances (not the expanded
                distances nor integers for embedding, i.e., nfeat_edge=None and bond_embedding_dim=None),
                kwargs can take additional inputs for expand the distance using Gaussian basis.

                centers (np.ndarray): array for defining the Gaussian expansion centers
                width (float): width for the Gaussian basis
    Returns:
        (Model) Keras model, ready to run
    """

    # Get the setting for the training kwarg of Dropout
    dropout_training = True if dropout_on_predict else None

    # atom inputs

    cn_x1 = Input(shape=(None, nfeat_node), name="cn_atom_feature_input")
    cn_x1_ = cn_x1
    
    nn_x1 = Input(shape=(None, nfeat_node), name="nn_atom_feature_input")
    nn_x1_ = nn_x1

    # bond inputs
    cn_x2 = Input(shape=(None, nfeat_edge), name="cn_bond_feature_input")
    cn_x2_ = cn_x2

    nn_x2 = Input(shape=(None, nfeat_edge), name="nn_bond_feature_input")
    nn_x2_ = nn_x2

    # state inputs
    if nfeat_global is None:
        # take default vector of two zeros
        cn_x3 = Input(shape=(None, 2), dtype=DataType.tf_float, name="cn_state_default_input")
        cn_x3_ = cn_x3
        nn_x3 = Input(shape=(None, 2), dtype=DataType.tf_float, name="nn_state_default_input")
        nn_x3_ = nn_x3
    else:
        cn_x3 = Input(shape=(None, nfeat_global), name="cn_state_feature_input")
        cn_x3_ = cn_x3
        nn_x3 = Input(shape=(None, nfeat_global), name="nn_state_feature_input")
        nn_x3_ = nn_x3
    cn_x4 = Input(shape=(None,), dtype=DataType.tf_int, name="cn_bond_index_1_input")
    cn_x5 = Input(shape=(None,), dtype=DataType.tf_int, name="cn_bond_index_2_input")
    cn_x6 = Input(shape=(None,), dtype=DataType.tf_int, name="cn_atom_graph_index_input")
    cn_x7 = Input(shape=(None,), dtype=DataType.tf_int, name="cn_bond_graph_index_input")
    nn_x4 = Input(shape=(None,), dtype=DataType.tf_int, name="nn_bond_index_1_input")
    nn_x5 = Input(shape=(None,), dtype=DataType.tf_int, name="nn_bond_index_2_input")
    nn_x6 = Input(shape=(None,), dtype=DataType.tf_int, name="nn_atom_graph_index_input")
    nn_x7 = Input(shape=(None,), dtype=DataType.tf_int, name="nn_bond_graph_index_input")

    # two feedforward layers
    def ff(x, n_hiddens=[n1, n2], name_prefix=None):
        if name_prefix is None:
            name_prefix = "FF"
        out = x
        for k, i in enumerate(n_hiddens):
            out = Dense(i, activation=act, name=f"{name_prefix}_{k}")(out)
        return out

    # a block corresponds to two feedforward layers + one MEGNetLayer layer
    # Note the first block does not contain the feedforward layer since
    # it will be explicitly added before the block
    # define the blocks for cn-graph part of the network
    def cn_one_block(a, b, c, has_ff=True, block_index=0):
        if has_ff:
            x1_ = ff(a, name_prefix=f"cn_block_{block_index}_atom_ff")
            x2_ = ff(b, name_prefix=f"cn_block_{block_index}_bond_ff")
            x3_ = ff(c, name_prefix=f"cn_block_{block_index}_state_ff")
        else:
            x1_ = a
            x2_ = b
            x3_ = c
        out = MEGNetLayer(
            [n1, n1, n2],
            [n1, n1, n2],
            [n1, n1, n2],
            pool_method="mean",
            activation=act,
            name=f"cn_megnet_{block_index}",
        )([x1_, x2_, x3_, cn_x4, cn_x5, cn_x6, cn_x7])

        x1_temp = out[0]
        x2_temp = out[1]
        x3_temp = out[2]
        if dropout:
            x1_temp = Dropout(dropout, name=f"cn_dropout_atom_{block_index}")(x1_temp, training=dropout_training)
            x2_temp = Dropout(dropout, name=f"cn_dropout_bond_{block_index}")(x2_temp, training=dropout_training)
            x3_temp = Dropout(dropout, name=f"cn_dropout_state_{block_index}")(x3_temp, training=dropout_training)
        return x1_temp, x2_temp, x3_temp

    # define the blocks for nn-graph part of the network
    def nn_one_block(a, b, c, has_ff=True, block_index=0):
        if has_ff:
            x1_ = ff(a, name_prefix=f"nn_block_{block_index}_atom_ff")
            x2_ = ff(b, name_prefix=f"nn_block_{block_index}_bond_ff")
            x3_ = ff(c, name_prefix=f"nn_block_{block_index}_state_ff")
        else:
            x1_ = a
            x2_ = b
            x3_ = c
        out = MEGNetLayer(
            [n1, n1, n2],
            [n1, n1, n2],
            [n1, n1, n2],
            pool_method="mean",
            activation=act,
            name=f"nn_megnet_{block_index}",
        )([x1_, x2_, x3_, nn_x4, nn_x5, nn_x6, nn_x7])

        x1_temp = out[0]
        x2_temp = out[1]
        x3_temp = out[2]
        if dropout:
            x1_temp = Dropout(dropout, name=f"nn_dropout_atom_{block_index}")(x1_temp, training=dropout_training)
            x2_temp = Dropout(dropout, name=f"nn_dropout_bond_{block_index}")(x2_temp, training=dropout_training)
            x3_temp = Dropout(dropout, name=f"nn_dropout_state_{block_index}")(x3_temp, training=dropout_training)
        return x1_temp, x2_temp, x3_temp

    cn_x1_ = ff(cn_x1_, name_prefix="cn_preblock_atom")
    cn_x2_ = ff(cn_x2_, name_prefix="cn_preblock_bond")
    cn_x3_ = ff(cn_x3_, name_prefix="cn_preblock_state")
    nn_x1_ = ff(nn_x1_, name_prefix="nn_preblock_atom")
    nn_x2_ = ff(nn_x2_, name_prefix="nn_preblock_bond")
    nn_x3_ = ff(nn_x3_, name_prefix="nn_preblock_state")
    for i in range(nblocks):
        if i == 0:
            has_ff = False
        else:
            has_ff = True
        cn_x1_1 = cn_x1_
        cn_x2_1 = cn_x2_
        cn_x3_1 = cn_x3_
        cn_x1_1, cn_x2_1, cn_x3_1 = cn_one_block(cn_x1_1, cn_x2_1, cn_x3_1, has_ff, block_index=i)
        nn_x1_1 = nn_x1_
        nn_x2_1 = nn_x2_
        nn_x3_1 = nn_x3_
        nn_x1_1, nn_x2_1, nn_x3_1 = nn_one_block(nn_x1_1, nn_x2_1, nn_x3_1, has_ff, block_index=i)
        # skip connection
        cn_x1_ = Add(name=f"cn_block_{i}_add_atom")([cn_x1_, cn_x1_1])
        cn_x2_ = Add(name=f"cn_block_{i}_add_bond")([cn_x2_, cn_x2_1])
        cn_x3_ = Add(name=f"cn_block_{i}_add_state")([cn_x3_, cn_x3_1])
        nn_x1_ = Add(name=f"nn_block_{i}_add_atom")([nn_x1_, nn_x1_1])
        nn_x2_ = Add(name=f"nn_block_{i}_add_bond")([nn_x2_, nn_x2_1])
        nn_x3_ = Add(name=f"nn_block_{i}_add_state")([nn_x3_, nn_x3_1])

    # print(Set2Set(T=npass, n_hidden=n3, kernel_regularizer=reg, name='set2set_atom'
    #             ).compute_output_shape([i.shape for i in [x1_, x6]]))
    # set2set for both the atom and bond
    cn_node_vec = Set2Set(T=npass, n_hidden=n3, name="cn_set2set_atom")([cn_x1_, cn_x6])
    nn_node_vec = Set2Set(T=npass, n_hidden=n3, name="nn_set2set_atom")([nn_x1_, nn_x6])
    # print('Node vec', node_vec)
    cn_edge_vec = Set2Set(T=npass, n_hidden=n3, name="cn_set2set_bond")([cn_x2_, cn_x7])
    nn_edge_vec = Set2Set(T=npass, n_hidden=n3, name="nn_set2set_bond")([nn_x2_, nn_x7])
    # concatenate atom, bond, and global
    cn_final_vec = Concatenate(axis=-1,name="cn_concatenated_layer")([cn_node_vec, cn_edge_vec, cn_x3_])
    nn_final_vec = Concatenate(axis=-1,name="nn_concatenated_layer")([nn_node_vec, nn_edge_vec, nn_x3_])
    # concatenate the cn and nn parts of the network
    final_vec = Concatenate(axis=-1,name="concatenated_layer")([cn_final_vec, nn_final_vec])
    
    if dropout:
        final_vec = Dropout(dropout, name="dropout_final")(final_vec, training=dropout_training)
    # final dense layers
    final_vec = Dense(128, activation=act, name="readout_0")(final_vec)
    final_vec = Dense(64, activation=act, name="readout_1")(final_vec)
    if is_classification:
        final_act = "sigmoid"
    else:
        final_act = None  # type: ignore
    out = Dense(ntarget, activation=final_act, name="readout_2")(final_vec)
    model = Model(inputs=[cn_x1, cn_x2, cn_x3, cn_x4, cn_x5, cn_x6, cn_x7, nn_x1, nn_x2, nn_x3, nn_x4, nn_x5, nn_x6, nn_x7], outputs=out)
    return model

# checks for data structure consistency
print("No. of CN bonds",len(cn_train_generator.__getitem__(0)[0][1][-1]))
print("No. of NN bonds",len(nn_train_generator.__getitem__(0)[0][1][-1]))
#print("Targets: ",len(cn_train_generator.__getitem__(0)[1]))
print("No. of targets in training set: ",len(cn_train_generator.__getitem__(0)[1][-1]))


def step_decay(epoch):
    """
    implementing learning-rate decay using preset decay steps

    Parameters
    ----------
    epoch: (int) epoch number during callback

    Returns
    -------
    lrate: current learning rate for the ML algorithm
    """
    #print(loss_history.losses)
    base_lrate = 0.01
    if epoch > 500:
        base_lrate = 0.001
    drop = 0.1
    epochs_drop = 11500.0
    lrate = base_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    #lrate = base_lrate
    return lrate

lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)
callbacks_list = [lrate]

# define new model
new_model = make_merged_model( 20, 2, 1, lr=0.01, n1=16, n2=16, n3=16, npass=1, ntarget=46, dropout=0.25, dropout_on_predict=True, graph_converter=MolecularGraph())
#new_model = make_merged_model( 20, 2, 1, lr=0.01, n1=16, n2=16, n3=16, npass=1, ntarget=46, graph_converter=MolecularGraph())

# compile new model
new_model.compile('adam', 'mse', metrics=['RootMeanSquaredError'])

# train new model
new_model.fit([(cn_train_generator.__getitem__(0)[0], nn_train_generator.__getitem__(0)[0])], cn_train_generator.__getitem__(0)[1], callbacks=callbacks_list, verbose=2, epochs=12000)
#print(new_model.summary())

Y_pred = []
YT_pred = []

# collect model predictions for training set
for i in train_indices:
    cn_graph = model.graph_converter.convert(cn_structures[i], full_pair_matrix=False)
    nn_graph = model.graph_converter.convert(nn_structures[i], full_pair_matrix=False)
    cn_inp = model.graph_converter.graph_to_input(cn_graph)
    nn_inp = model.graph_converter.graph_to_input(nn_graph)
    #list_desc = (xdata[i].tolist())
    #graph["state"] = [(list_desc)]
    Y_pred.append(np.round(new_model.predict([(cn_inp,nn_inp)])[0,0], 2).tolist())

# collect model predictions for testing set
for i in test_indices:
    cn_graph = model.graph_converter.convert(cn_structures[i], full_pair_matrix=False)
    nn_graph = model.graph_converter.convert(nn_structures[i], full_pair_matrix=False)
    cn_inp = model.graph_converter.graph_to_input(cn_graph)
    nn_inp = model.graph_converter.graph_to_input(nn_graph)
    #list_desc = (xdata[i].tolist())
    #graph["state"] = [(list_desc)]
    YT_pred.append(np.round(new_model.predict([(cn_inp,nn_inp)])[0,0], 2).tolist())


# save model
#print('Model saved in ligsMEG_model.hdf5 file')
print('16, 16, 16, 3-block Megnet for reg with 0.25 dropout, trained for 12000 epochs in one batch')
#new_model.save(curr_dir + '/ligsMEG_model.hdf5')

# calculating performance metrics
train_metrics = weight_calc_metrics(y_true=Y, y_pred=Y_pred, yweights=Y_wt)
print("\n\ntrain set weighted metrics: ", train_metrics)
test_metrics = weight_calc_metrics(y_true=YT, y_pred=YT_pred, yweights=YT_wt)
print("\n\ntest set weighted metrics: ", test_metrics)

# gather true intensities and predicted intensities to calculate E50/50
YT_df = pd.DataFrame(YT, columns = ['475','480','485','490','495','500','505','510','515',
                                   '520','525','530','535','540','545','550','555','560',
                                   '565','570','575','580','585','590','595','600','605',
                                   '610','615','620','625','630','635','640','645','650',
                                   '655','660','665','670','675','680','685','690','695','700'])

YT_pred_df = pd.DataFrame(YT_pred, columns = ['475','480','485','490','495','500','505','510','515',
                                   '520','525','530','535','540','545','550','555','560',
                                   '565','570','575','580','585','590','595','600','605',
                                   '610','615','620','625','630','635','640','645','650',
                                   '655','660','665','670','675','680','685','690','695','700'])

wavelengths = np.asarray([float(x) for x in YT_df.columns.values.tolist()])
print(wavelengths.shape)

def find_half_sum_index(row):
    """
    find wavelength at which integral of spectrum is half

    Parameters
    ----------
    row: (pd.Series): A Pandas Series representing a row in the DataFrame.

    Returns
    -------
    output: the wavelength at which integral of spectrum is half 
    """
    # calculate cumulative sum and its halfway point
    cumulative_sum = row.cumsum()
    half_sum = row.sum() / 2
    # find the wavelength at which the halfway point is cleared
    high = (cumulative_sum >= half_sum).idxmax()
    int_hi = int(high)
    int_lo = int_hi - 5
    low = str(int_lo)
    # interpolate between the wavelenths closest to the halfway point to find the actual value 
    output = int_lo + (5)*(half_sum - cumulative_sum[low])/(cumulative_sum[high] - cumulative_sum[low])
    #print(output)
    return output

# calculating E50/50 and returning metrics for its prediction
e50_true = 28590.4/YT_df.apply(find_half_sum_index, axis=1)
e50_pred = 28590.4/YT_pred_df.apply(find_half_sum_index, axis=1)
e50_metrics = weight_calc_metrics(y_true=e50_true, y_pred=e50_pred, yweights=YT_wt)
print("\n\nE50 weighted metrics: ", e50_metrics)

