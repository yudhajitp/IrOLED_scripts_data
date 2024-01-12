import pandas as pd
import os, sys
import numpy as np
import math
import tensorflow as tf
import tensorflow.keras.layers as kl
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

# input data
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

# extracting y-data and its weights
ydata1 = ydata.TANH_550_100.values.reshape(-1,1) 
ydata_sl = ydata.Complex_i.values
ydata = ydata.drop(['Complex_i','TANH_550_100','cn_IE', 'cn_EA', 'cn_ST', 'cn_mu', 'cn_alpha', 'cn_HOMO', 'cn_LUMO', 'cn_gap', 'cn_R2', 'cn_ZPVE', 'cn_U0', 'cn_U', 'cn_H', 'cn_G', 'cn_Cv', 'cn_w1', 'nn_IE', 'nn_EA', 'nn_ST', 'nn_mu', 'nn_alpha', 'nn_HOMO', 'nn_LUMO', 'nn_gap', 'nn_R2', 'nn_ZPVE', 'nn_U0', 'nn_U', 'nn_H', 'nn_G', 'nn_Cv', 'nn_w1'], axis=1)
ydata3 = ydata.values.reshape(-1,46)
ydata = ydata3

# random splitting of the dataset into 80:20 train-test splits, splitting the indices
print("after reading: ", xdata.shape, ydata.shape)
train_indices, test_indices = train_test_split(list(range(len(ydata))), test_size=len(ydata)-(math.floor(0.8*len(ydata)/5)*5), random_state=17)

# scaling the x-data and extracting train and test x-data
clf_x = StandardScaler()
X = xdata[train_indices]
X = clf_x.fit_transform(X)
XT = xdata[test_indices]
XT = clf_x.transform(XT)
feats = X.shape[1]

# extracting train and test y-data
Y = ydata[train_indices]
Y_wt = ydata1[train_indices]
YT = ydata[test_indices]
YT_wt = ydata1[test_indices]

print("after processing: ", X.shape, Y.shape, XT.shape, YT.shape)

# defining Keras MLP model 
def makeModel():
    """
    DENSE -> DENSE -> DENSE 
    
    Arguments:
    None

    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """
    model = tf.keras.Sequential(layers=[
    kl.Dense(128, input_dim=feats, activation='relu'),
    kl.Dropout(0.25),
    kl.BatchNormalization(),
    kl.Dense(64, activation='relu'),
    kl.Dense(46, activation='linear')
    ])
    return model

# compiling model
mlp_lig_model = makeModel()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
mlp_lig_model.compile(optimizer=optimizer,
                   loss='MeanSquaredError',
                   metrics=['RootMeanSquaredError'],
                   weighted_metrics=['RootMeanSquaredError'])

# implementing learning-rate decay
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_delta = 0.01, min_lr=0.00001)
callbacks_list = [reduce_lr]

# training model and generating predictions post-training for both train and test sets
mlp_lig_model.fit(X, Y, sample_weight=Y_wt, callbacks=callbacks_list, verbose=1, epochs=2000, batch_size=32)
mlp_lig_model.evaluate(XT, YT, sample_weight=YT_wt)
Y_pred = mlp_lig_model.predict(X)
YT_pred = mlp_lig_model.predict(XT)

# calculating performance metrics
train_metrics = weight_calc_metrics(y_true=Y, y_pred=Y_pred, yweights=Y_wt.reshape(-1))
print("\n\ntrain set weighted metrics: ", train_metrics)
test_metrics = weight_calc_metrics(y_true=YT, y_pred=YT_pred, yweights=YT_wt.reshape(-1))
print("\n\ntest set weighted metrics: ", test_metrics)


# gather true intensities and predicted intensities to calculate E50/50
YT_df = pd.DataFrame(YT, columns = ['475','480','485','490','495','500','505','510','515','520','525','530',
                                    '535','540','545','550','555','560','565','570','575','580','585','590',
                                    '595','600','605','610','615','620','625','630','635','640','645','650',
                                    '655','660','665','670','675','680','685','690','695','700'])

YT_pred_df = pd.DataFrame(YT_pred, columns = ['475','480','485','490','495','500','505','510','515','520','525','530',
                                    '535','540','545','550','555','560','565','570','575','580','585','590',
                                    '595','600','605','610','615','620','625','630','635','640','645','650',
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
e50_metrics = weight_calc_metrics(y_true=e50_true, y_pred=e50_pred, yweights=YT_wt.reshape(-1))
print("\n\nE50 weighted metrics: ", e50_metrics)


# save model
#mlp_lig_model.save(curr_dir + '/new_mlp_model.hdf5')


