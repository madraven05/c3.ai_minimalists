import numpy as np
import tensorflow as tf
from models import LSTM_Model, Vulnerability_Calculator
from plot import path_state_vul, path_road_vul, path_centroid_vul, path_air_transport_vul, path_air_state_vul
from plot import plot_state_health_vul, plot_state_social_vul, plot_state_road_vul
import pandas as pd


# Predict and update csv
# lstm_model_1 = LSTM_Model(n_steps = 7, n_features = 1)
# lstm_model_1.update_csv()


cal = Vulnerability_Calculator()

# # Run R files
# cal.calculate_weights()
# # Update json files
# cal.update_json()

# Plot
# plot_state_road_vul(road_path=path_road_vul, state_path=path_state_vul, centroid_path=path_centroid_vul)

# print(output)


# Run R files
# cal.calculate_weights()
# Update json files
# cal.update_json()
plot_state_road_vul(road_path=path_road_vul, state_path=path_state_vul, centroid_path=path_centroid_vul)
# plot_state_health_vul(path_state_vul)