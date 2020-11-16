

'''
Create a class model
'''
# univariate lstm example
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split

# print(tf.version)

class LSTM_Model():

    def __init__(self, n_steps, n_features, n_folds):
        
        self.n_steps = n_steps
        self.n_features = n_features
        self.n_folds = n_folds

    def set_data(self, data_loc, data_type):
        '''
        data_loc = location of the csv file
        data_type = 'new_case_count' or 'case_count' etc.
        '''
        time_data = pd.read_csv(data_loc)
        self.case_count_data = time_data[data_type]
        self.case_count_data = list(self.case_count_data)
        for i in range(len(self.case_count_data)-1):
            if i!=0:
                self.case_count_data[i] = float(self.case_count_data[i])
            else:
                self.case_count_data[i] = float(self.case_count_data[i])

        self.case_count_data.remove('.')
        self.case_count_data = np.array(self.case_count_data)
        self.max_count = max(self.case_count_data)
        self.min_count = min(self.case_count_data)
    
        # Normalise
        self.case_count_data = (self.case_count_data - self.min_count)/(self.max_count - self.min_count)
        return self.case_count_data, self.max_count, self.min_count

    '''split a univariate sequence into samples'''
    def split_sequence(self, sequence, n_steps):
        X, y = list(), list()
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the sequence
            if end_ix > len(sequence)-1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    ''' Make data '''
    def make_data(self, stateID):
        self.stateID = stateID
        self.X, self.y = self.split_sequence(self.case_count_data, self.n_steps)
        self.X = self.X.reshape((self.X.shape[0], self.X.shape[1], self.n_features))
        self.X_polyfit = self.X[-self.n_folds:]
        self.y_polyfit = self.y[-self.n_folds:]
        train_days = len(self.X)
        self.X = self.X[ : train_days-self.n_folds]
        self.y = self.y[ : train_days-self.n_folds]
        self.n_fold_test_train = {'train' : [], 'test': []}
        
        for i in range(self.n_folds):
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=i)
            self.n_fold_test_train['train'].append([self.X_train, self.y_train])
            self.n_fold_test_train['test'].append([self.X_test, self.y_test])
        

    ''' Model fit '''
    def model_compile(self):
        self.model = Sequential()
        self.model.add(LSTM(100, return_sequences=True, activation='relu', input_shape=(self.n_steps, self.n_features)))
        self.model.add(LSTM(100, activation='relu'))
        self.model.add(Dense(256, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), 
                                  bias_regularizer=regularizers.l2(1e-4), 
                                  activity_regularizer=regularizers.l2(1e-5)))
        # self.model.add(Dropout(0.25))
        self.model.add(Dense(1))

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.Huber(), metrics = 'mse')
        return self.model

    '''Train Model'''
    def train_model(self, n_iter, input_data, output):
        parent_dir = 'models/'
        print("Training Model .............")
        self.model = self.model_compile()  
        self.model.fit(input_data, output, epochs=50, verbose=1)
        print("Trained.............")
        # model.fit(time_series_data, Output_timeSeries, epochs=50, verbose=2)
        print("Saving Model .............")
        self.model.save(parent_dir+"CTST_2LSTM_100_{}_{}.h5".format(n_iter, self.stateID))
        # fold of iteration, id of the state
        print("Saved")

    '''n-Fold Training'''
    def n_fold_train(self):
        for i in range(len(self.n_fold_test_train['train'])):
            self.train_model(i, self.n_fold_test_train['train'][i][0], self.n_fold_test_train['train'][i][1]) 

    '''n-Fold Predict'''    
    def n_fold_predict(self):
        output = []
        for i in range(self.n_folds):
            parent_dir = "models/"
            window = self.n_steps
            X_window = self.X_polyfit[i]
            X_window = X_window.reshape((1, self.n_steps, self.n_features))
            temp_output = []
            for j in range(self.n_folds):
                model = load_model(parent_dir+"CTST_2LSTM_100_{}_{}.h5".format(j, self.stateID))
                case_stat = model.predict(X_window)
                temp_output.append(case_stat[0][0])
            output.append(temp_output)

        return output



    '''Model Predict'''
    def predict(self):
        parent_dir = "models/"
        output = self.n_fold_predict()
        test_model = self.case_count_data[-self.n_steps:]
        test_model = test_model.reshape((1, self.n_steps, self.n_features))
        test_output = np.zeros((5,))
        for i in range(self.n_folds):
            self.model = load_model(parent_dir+"CTST_2LSTM_100_{}_{}.h5".format(i, self.stateID))
            y = self.model.predict(test_model)
            test_output[i] = y

        y_polyfit_new = self.y_polyfit.reshape(len(self.y_polyfit), 1)
        weights = np.linalg.inv(output).dot(y_polyfit_new)

        final_output = np.dot(test_output, weights)
        
        self.case_count_prediction = final_output*(self.max_count-self.min_count) + self.min_count
        
        return self.case_count_prediction