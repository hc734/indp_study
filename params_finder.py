import pandas as pd 
import os
import numpy as np
import itertools as it
import logging


import numpy as np
import sys
import time
from tqdm._tqdm_notebook import tqdm_notebook
import pickle
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras import optimizers
# from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

search_params_old = {
    "batch_size": [20, 30, 40],
    "time_steps": [30, 60, 90], 
    "lr": [0.01, 0.001, 0.0001],
    "epochs": [30, 50, 70]
}

search_params = {
    "batch_size": [25, 40, 50],
    "time_steps": [20, 15, 30], 
    "lr": [0.0001],
    "epochs": [60, 100, 120]
}





def trim_dataset(mat,batch_size):
    """
    trims dataset to a size that's divisible by BATCH_SIZE
    """
    no_of_rows_drop = mat.shape[0]%batch_size
    if no_of_rows_drop > 0:
        return mat[:-no_of_rows_drop]
    else:
        return mat


def eval_model(mat, combination, i):
    """
    implement your logic to build a model, train it and then calculate validation loss.
    Save this validation loss using CSVLogger of Keras or in a text file. Later you can
    query to get the best combination.
    """
    params = {
    "batch_size": combination[0],  # 20<16<10, 25 was a bust
    "time_steps": combination[1],
    "lr": combination[2],
    "epochs": combination[3]
    }

    TIME_STEPS = params["time_steps"]
    BATCH_SIZE = params["batch_size"]


    def create_model():
        lstm_model = Sequential()
        # (batch_size, timesteps, data_dim)
        lstm_model.add(LSTM(100, batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]),
                            dropout=0.0, recurrent_dropout=0.0, stateful=True, return_sequences=True,
                            kernel_initializer='random_uniform'))
        lstm_model.add(Dropout(0.4))
        lstm_model.add(LSTM(60, dropout=0.0))
        lstm_model.add(Dropout(0.4))
        lstm_model.add(Dense(20,activation='relu'))
        lstm_model.add(Dense(1,activation='sigmoid'))
        optimizer = optimizers.RMSprop(lr=params["lr"])
        # optimizer = optimizers.SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True)
        lstm_model.compile(loss='mean_squared_error', optimizer=optimizer)
        return lstm_model

    def build_timeseries(mat, y_col_index):
        """
    Converts ndarray into timeseries format and supervised data format. Takes first TIME_STEPS
    number of rows as input and sets the TIME_STEPS+1th data as corresponding output and so on.
    :param mat: ndarray which holds the dataset
    :param y_col_index: index of column which acts as output
    :return: returns two ndarrays-- input and output in format suitable to feed
    to LSTM.
        """
    # total number of time-series samples would be len(mat) - TIME_STEPS
        dim_0 = mat.shape[0] - TIME_STEPS
        dim_1 = mat.shape[1]
        x = np.zeros((dim_0, TIME_STEPS, dim_1))
        y = np.zeros((dim_0,))
        print("dim_0",dim_0)
        for i in tqdm_notebook(range(dim_0)):
            x[i] = mat[i:TIME_STEPS+i]
            y[i] = mat[TIME_STEPS+i, y_col_index]
#         if i < 10:
#           print(i,"-->", x[i,-1,:], y[i])
        print("length of time-series i/o",x.shape,y.shape)
        return x, y


    train_cols = ["JPM"]
    df_train, df_test = train_test_split(mat, train_size=0.8, test_size=0.2, shuffle=False)
    print("Train--Test size", len(df_train), len(df_test))
    x = df_train.loc[:,train_cols].values
    min_max_scaler = MinMaxScaler()
    x_train = min_max_scaler.fit_transform(x)
    x_test = min_max_scaler.transform(df_test.loc[:,train_cols])

    print("Deleting unused dataframes of total size(KB)",(sys.getsizeof(mat)+sys.getsizeof(df_train)+sys.getsizeof(df_test))//1024)

    del mat
    del df_test
    del df_train
    del x

    print("Are any NaNs present in train/test matrices?",np.isnan(x_train).any(), np.isnan(x_train).any())
    x_t, y_t = build_timeseries(x_train, 0)
    x_t = trim_dataset(x_t, BATCH_SIZE)
    y_t = trim_dataset(y_t, BATCH_SIZE)
    print("Batch trimmed size",x_t.shape, y_t.shape)

    x_temp, y_temp = build_timeseries(x_test, 0)
    x_val, x_test_t = np.split(trim_dataset(x_temp, BATCH_SIZE),2)
    y_val, y_test_t = np.split(trim_dataset(y_temp, BATCH_SIZE),2)


    print("Test size", x_test_t.shape, y_test_t.shape, x_val.shape, y_val.shape)


    model = None
    results = []

    is_update_model = False 
    if model is None or is_update_model:
        from keras import backend as K
        print("Building model...")
        print("checking if GPU available", K.tensorflow_backend._get_available_gpus())
        model = create_model()
        
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                           patience=40, min_delta=0.0001)
        
        mcp = ModelCheckpoint(os.path.join(OUTPUT_PATH,
                              "best_model.h5"), monitor='val_loss', verbose=1,
                              save_best_only=True, save_weights_only=False, mode='min', period=1)

        # Not used here. But leaving it here as a reminder for future
        r_lr_plat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30, 
                                      verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
        
        csv_logger = CSVLogger(os.path.join(OUTPUT_PATH, 'training_log_' + time.ctime().replace(" ","_") + '.log'), append=True)
        
        history = model.fit(x_t, y_t, epochs=params["epochs"], verbose=2, batch_size=BATCH_SIZE,
                            shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),
                            trim_dataset(y_val, BATCH_SIZE)))
        
        y_pred = model.predict(trim_dataset(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)
        y_pred = y_pred.flatten()
        y_test_t = trim_dataset(y_test_t, BATCH_SIZE)
        error = mean_squared_error(y_test_t, y_pred)
        print("Error is", error, y_pred.shape, y_test_t.shape)

        results = [results, error]

    return(print(results, combination))




def get_all_combinations(params):
    all_names = params.keys()
    combinations = it.product(*(params[name] for name in all_names))
    return list(combinations)

def run_search(mat, params):
    param_combs = get_all_combinations(params) # list of tuples
    logging.info("Total combinations to try = {}".format(len(param_combs)))
    for i, combination in enumerate(param_combs):
        logging.info("Trying combo no. {} {}".format(i, combination))
        eval_model(mat, combination, i)


iter_changes = "dropout_layers_0.1"

INPUT_PATH = "C:/Users/piano/Desktop/Spring 2019/Research/Stock-Price-Prediction-master"+"/inputs"
OUTPUT_PATH = "C:/Users/piano/Desktop/Spring 2019/Research/Stock-Price-Prediction-master"+"/outputs/lstm_best_4-26-19_12AM/"+iter_changes

df_ge = pd.read_csv(os.path.join(INPUT_PATH, "jpm.csv"), engine='python')

run_search(df_ge, search_params)