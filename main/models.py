'''
Create a class model
'''
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

import subprocess 
import geopandas as gpd
import math
# print(tf.version)

class LSTM_Model():

    '''
    Class for the Time series prediction model
    '''

    def __init__(self, n_steps, n_features):
        
        self.n_steps = n_steps
        self.n_features = n_features
        # self.n_folds = n_folds

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


    
    def split_sequence(self, sequence, n_steps):
        '''split sequence into samples'''

        X, y = list(), list()
        for i in range(len(sequence)):
            end_ix = i + n_steps
            if end_ix > len(sequence)-1:
                break
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)


    def make_data(self, stateID):
        ''' Make data '''

        self.stateID = stateID
        self.X, self.y = self.split_sequence(self.case_count_data, self.n_steps)
        self.X = self.X.reshape((self.X.shape[0], self.X.shape[1], self.n_features))
        train_days = len(self.X)
        self.X = self.X[ : train_days-self.n_folds]
        self.y = self.y[ : train_days-self.n_folds]
        

    
    def model_compile(self):
        ''' Model fit '''


        self.model = Sequential()
        self.model.add(LSTM(128,activation='relu', input_shape=(self.n_steps, self.n_features)))
        # self.model.add(LSTM(100, activation='relu'))
        self.model.add(Dense(2048, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-6), 
                                  bias_regularizer=regularizers.l2(1e-4), 
                                  activity_regularizer=regularizers.l2(1e-5)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1))

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.Huber(), metrics = 'mse')
        return self.model

    
    def train_model(self):
        '''Train Model'''

        parent_dir = '../trained_models/'
        print("Training Model .............")
        self.model = self.model_compile()  
        self.model.fit(self.X, self.y, epochs=50, verbose=1)
        print("Trained.............")
        # model.fit(time_series_data, Output_timeSeries, epochs=50, verbose=2)
        print("Saving Model .............")
        self.model.save(parent_dir+"CTST_2LSTM_100_{}.h5".format(self.stateID))
        # fold of iteration, id of the state
        print("Saved")


    '''Model Predict for the next 3 days'''
    def predict_3_days(self):
        parent_dir = "../trained_models/"
        test_model = self.case_count_data[-self.n_steps:]
        print(test_model)
        # test_model = test_model.reshape((1, self.n_steps, self.n_features))
        final_predict = []
        window = test_model
        for j in range(3):
            x_window = window.reshape((1,self.n_steps,self.n_features))
            # for i in range(self.n_folds):
            model = load_model(parent_dir+"CTST_2LSTM_100_{}.h5".format(self.stateID))
            print(model.summary())
            y = model.predict(x_window)
            test_output = y

            case_count_prediction = test_output*(self.max_count-self.min_count) + self.min_count
            final_predict.append(case_count_prediction[0][0])
            # window.append(case_count_prediction)
            # print(window)
            window = np.append(window, test_output[0])
            window = window[1:]
            

        return final_predict

    '''
    Prediction for the next day and update CSV
    '''
    def update_csv(self):
        parent_dir = "../trained_models/"
        predictions = []
        # For 56 states
        for i in range(1,57,1):
            try:
                self.set_data("../databases/{}.csv".format(i), "case_count") # Case count data set
                x_input = self.case_count_data[-self.n_steps:]
                x_input = x_input.reshape((1,self.n_steps,self.n_features))
                
                model = load_model(parent_dir+"CTST_2LSTM_100_{}.h5".format(i))
                y = model.predict(x_input)
                case_count_prediction = y[0][0]*(self.max_count-self.min_count) + self.min_count
                predictions.append(int(case_count_prediction))
            except Exception as inst:
                print(inst, i)
                

        print(predictions[:4])
        path_to_csv = "../R_code/Data_input/health_input.csv"
        df = pd.read_csv(path_to_csv)
        print(len(predictions))
        df['Total_Case'] = predictions

        normalised_predictions = [(case-min(predictions))/(max(predictions)-min(predictions)) for case in predictions] # Normalised Predictions
        df['TC_S'] = normalised_predictions # Update the TC_S column
        df.to_csv(path_to_csv) # Save in .csv file
        # print(df['TC_S'].head(10))



class Vulnerability_Calculator():
    '''
    Class for Vulnerability calculation
    '''
    def calculate_weights(self):
        
        path_to_dea = "../R_code/Data_Envelopment_Analysis.R"
        path_to_wt_assgn = "../R_code/weight_assignment.R"

        print("Running R files...")
        subprocess.call(["/usr/bin/Rscript", path_to_dea])
        subprocess.call(["/usr/bin/Rscript", path_to_wt_assgn])


    def update_json(self):
        '''
        Updating json files
        ''' 
        # Updating State Polygon json file
        print("Updating State Polygon JSON file....")
        states_wt_df = pd.read_csv("../R_code/Data_input/dea_efficiencies_health_USA.csv")
        state_poly_df = gpd.read_file("../json/State_polygon_vulnerability_.json")

        state_poly_df['health_vul'] = states_wt_df['ccr.eff']

        Total_vul = []
        for i in range(len(state_poly_df['health_vul'])):
            Total_vul.append(math.sqrt(state_poly_df['health_vul'][i] * state_poly_df['Social_vul'][i]))

        state_poly_df['Total_vul'] = Total_vul
        print("Done!")

        # Updating State centroid json file
        print("Updating State centroid JSON file....")
        states_wt_df = pd.read_csv("../R_code/Data_input/dea_efficiencies_health_USA.csv")
        state_cent_df = gpd.read_file("../json/state_centroid_vulnerability.json")

        state_cent_df['health_vul'] = states_wt_df['ccr.eff']

        Total_vul = []
        for i in range(len(state_cent_df['health_vul'])):
            Total_vul.append(math.sqrt(state_cent_df['health_vul'][i] * state_cent_df['Social_vul'][i]))

        state_cent_df['Total_vul'] = Total_vul
        print("Done!")

        # Updating Road Networks json file
        print("Updating Road Networks json file...")
        road_wt_df = pd.read_csv("../R_code/R_output/USA_weights_distance.csv")
        road_vul_df = gpd.read_file("../json/State_road_vulnerability_upd.json")

        for idx,rows in road_wt_df.iterrows():
            i_name = rows['i_name']
            j_name = rows['j_name']
            wt = rows['weight']
            # print(i_name, j_name, wt)
            for idx1, rows1 in road_vul_df.iterrows():
                if i_name == rows1['i_name'] and j_name == rows1['j_name']:
                    rows1['wt_ij'] == wt 
                    break
        print("Done!")

        
