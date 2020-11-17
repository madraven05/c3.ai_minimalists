# ML Guided Network Science Based Vulnerability Model
The Covid-19 Pandemic has affected the whole world. As we slowly move towards recovering from this pandemic, we are bound to explore options that can help us effectively restore normalcy. 
<br />We use a LSTM based Time-Series prediction model which accurately predicts the future case counts. This in turn is used to calculate the Health, Social and Total Vulnerability of all the states of the USA. 
<br /> We also calculate travel bubbles or corridors (using the Total vulnerability calculated) that give an insight on which ones are the safest.  

## Using the model
```python
from models import LSTM_Model, Vulnerability_Calculator
from plot import path_state_vul, path_road_vul, path_centroid_vul
from plot import plot_state_health_vul, plot_state_social_vul, plot_state_road_vul

lstm_model = LSTM_Model(n_steps=7, n_features=1)
# Predict next day value and update the required csv
lstm_model.update_csv()

cal = Vulnerability_Calculator()

# Using R codes calculate files and update the required json files
cal.calculate_weights()
cal.update_json()

# Plotting Road Vulnerabilties
plot_state_road_vul(road_path=path_road_vul, state_path=path_state_vul, centroid_path=path_centroid_vul)
```
### Output
The output of the above code is, 
![Road Vulnerabilities](media/road_vul.png)

The social vulnerabilities of the state can also be calculated. Shown below is one of the results that we get,
![Social Vulnerabilities](media/social_vul.png)

## Video Presentation
<figure class="video_container">
  <video controls="true" allowfullscreen="true">
    <source src="media/C3.Ai-1.mp4" type="video/mp4">
  </video>
</figure>