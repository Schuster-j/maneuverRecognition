```python
# Maneuver Recognition Package
```

## Contents

## About


```python

```


```python

```

...

## Example application: Real world vehicle maneuver recognition using smartphone sensors and LSTM models

### Project background

<center>
<img src="images/sensor_axes.png" width="800"/>
</center>

Positioning ...

![Sensor axes with positioning in vehicle](images/sensor_positioning.png)




```python

```


```python
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import torch
import torch.nn as nn
```

### Loading the dataset
The dataset consists of 19 variables and the recording of three different people with different vehicles. Each driving maneuver was assigned a maneuver type, the route section, and road type as well as an unique maneuver ID. The smartphone sensor data includes datetime, acceleration on and rotation around three orthogonal axes, as well as GPS information such as longitude, latitude, altitude, accuracy and speed.


```python
df = pd.read_csv("data/SensorRec_data_eng.csv")
df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>accX</th>
      <th>accY</th>
      <th>accZ</th>
      <th>gyroX</th>
      <th>gyroY</th>
      <th>gyroZ</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>altitude</th>
      <th>accuracy</th>
      <th>speed</th>
      <th>maneuverID</th>
      <th>maneuverType</th>
      <th>maneuverElement</th>
      <th>section</th>
      <th>roadType</th>
      <th>vehicle</th>
      <th>person</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-01-06 10:44:29.466</td>
      <td>0.021387</td>
      <td>-0.006456</td>
      <td>-0.084595</td>
      <td>-0.072036</td>
      <td>-0.009720</td>
      <td>-0.006924</td>
      <td>52.366269</td>
      <td>9.761421</td>
      <td>98.700005</td>
      <td>11.716</td>
      <td>0.060892</td>
      <td>000000-P1M</td>
      <td>stationary</td>
      <td>stationary</td>
      <td>1</td>
      <td>city_road</td>
      <td>Skoda Fabia</td>
      <td>P01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-01-06 10:44:29.958</td>
      <td>0.023271</td>
      <td>0.100068</td>
      <td>0.143770</td>
      <td>0.013182</td>
      <td>0.004794</td>
      <td>0.000399</td>
      <td>52.366269</td>
      <td>9.761421</td>
      <td>98.700005</td>
      <td>11.716</td>
      <td>0.060892</td>
      <td>000000-P1M</td>
      <td>stationary</td>
      <td>stationary</td>
      <td>1</td>
      <td>city_road</td>
      <td>Skoda Fabia</td>
      <td>P01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-01-06 10:44:30.456</td>
      <td>0.030368</td>
      <td>-0.006946</td>
      <td>0.019102</td>
      <td>0.000000</td>
      <td>-0.001997</td>
      <td>0.001198</td>
      <td>52.366270</td>
      <td>9.761421</td>
      <td>98.700005</td>
      <td>9.591</td>
      <td>0.035770</td>
      <td>000000-P1M</td>
      <td>stationary</td>
      <td>stationary</td>
      <td>1</td>
      <td>city_road</td>
      <td>Skoda Fabia</td>
      <td>P01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-01-06 10:44:30.957</td>
      <td>0.012233</td>
      <td>-0.005673</td>
      <td>0.002899</td>
      <td>0.000266</td>
      <td>0.001198</td>
      <td>0.001065</td>
      <td>52.366270</td>
      <td>9.761421</td>
      <td>98.700005</td>
      <td>9.591</td>
      <td>0.035770</td>
      <td>000000-P1M</td>
      <td>stationary</td>
      <td>stationary</td>
      <td>1</td>
      <td>city_road</td>
      <td>Skoda Fabia</td>
      <td>P01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-01-06 10:44:31.456</td>
      <td>0.017861</td>
      <td>0.001417</td>
      <td>0.017249</td>
      <td>-0.001332</td>
      <td>-0.002397</td>
      <td>0.000399</td>
      <td>52.366270</td>
      <td>9.761421</td>
      <td>98.700005</td>
      <td>9.591</td>
      <td>0.035770</td>
      <td>000000-P1M</td>
      <td>stationary</td>
      <td>stationary</td>
      <td>1</td>
      <td>city_road</td>
      <td>Skoda Fabia</td>
      <td>P01</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_p1 = df.loc[df['person'] == 'P01', :]
df_p2 = df.loc[df['person'] == 'P02', :]
df_p3 = df.loc[df['person'] == 'P03', :]

dataframe_list = [df_p1, df_p2, df_p3]
```

## Preprocessing

#### Windowing and Train-/Test-Splitting with preprocessing.timeseries_train_test_split

Our goal is to create a model which can correctly identify driving maneuvers in unknown data of future driving recordings. Therefore, to train the model, we cannot simply use the individual maneuvers already assigned to their IDs, but have to use a sliding window to create training and testing cases.

In order to do that, we are in need of two specific functions:
1. A windowing function that gets our data and returns windows with a specific amount of time steps and a specific step length between those windows in the shape of (cases, timesteps, variables).
2. A function to split the data into training and testing data before the windowing is applied. If we do not split training and testing data before using the windowing function, there will be data leakage because overlapping windows may end up in both partitions.

For this purpose we can use the timeseries_train_test_split function in the preprocessing module. It first splits the given dataframe in an arbitrary number of partitions. Then it randomly selects some of these partitions in regard to our test size and removes them from a copy of the full dataframe. The remaining data will be used for training. We can also use this function to apply a robust scaling which will use the training data for fitting the scaler. The windowing function will then be applied on the training and testing partitions separately.

With this approach, it can still happen occasionally that we create unnatural transitions between maneuvers that occurred before and after the partitions we cut out for the test data or that we break some maneuvers. However, the amount of it which depends on the number of splits we use will be negligible, as it is clearly more important that we have absolutely no data leakage. 

The following parameters can be configured:
- splits: Number of random partitions in which data will be separated before windowing will be applied.
- test_size: Proportion of data to use for testing.
- time_steps: Length of windows in number of rows.
- step_size: Steps between windows in number of rows.
- scale: Bool whether we apply robust scaling or not.

Since we have data of three different persons and to ensure that different route segments of the trips can appear in the training and test data we will apply the timeseries_train_test_split function on each individual dataset and combine the windowed data subsequently. Our windows will have a length of 20 timesteps which is 10 seconds in our dataset and we will use a stepsize of 4 timesteps which equals 2 seconds. We then merge the data, with every window seperated and already in the right shape.


```python
from maneuver_recognition import preprocessing

x_vars = ['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ', 'speed']
y_var = 'maneuverType'
splits = 40
test_size = 0.2
time_steps = 20
step_size = 4

X_train, y_train, X_test, y_test = [], [], [], []
for df in dataframe_list:
    X_train_df, y_train_df, X_test_df, y_test_df = preprocessing.timeseries_train_test_split(df, x_variables=x_vars,
                y_variable=y_var, splits=splits, test_size=test_size,time_steps=time_steps, step_size=step_size, scale=True)

    
    X_train.append(X_train_df)
    y_train.append(y_train_df)
    X_test.append(X_test_df)
    y_test.append(y_test_df)
    
X_train, y_train, X_test, y_test = np.vstack(X_train), np.vstack(y_train), np.vstack(X_test), np.vstack(y_test)
```

Using this function will result in having our train and test data partitions in a shape of (cases, timesteps, variables) for x and (cases, label) for y.


```python
X_train.shape, X_test.shape, y_train.shape, y_test.shape
```




    ((5745, 20, 7), (1325, 20, 7), (5745, 1), (1325, 1))





#### Dropping rare maneuvers and balancing classes with preprocessing.remove_maneuvers

To examine how many individual driving maneuvers were recorded per maneuver type, the following plot can be considered. It turns out that some maneuvers, such as overtaking or crossing an intersection, occurred in too few cases overall. Therefore these maneuvers should be excluded. By using preprocessing.remove_maneuvers we can drop maneuvers completely or apply undersampling to balance the class distributions within both our test and training data.


```python
fig = go.Figure(data=[
    go.Bar(name='Training data', x=np.unique(y_train), y=np.unique(y_train, return_counts=True)[1]),
    go.Bar(name='Test data', x=np.unique(y_test), y=np.unique(y_test, return_counts=True)[1])
])

fig.update_layout(barmode='stack', title="Amount of windows per maneuver type in the train and test data")
fig.update_xaxes(title_text='maneuver')
fig.update_yaxes(title_text='Amount')
fig.show()
```


<div>                            <div id="ec262c2a-ad91-4cce-afca-cb158eb43aab" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("ec262c2a-ad91-4cce-afca-cb158eb43aab")) {                    Plotly.newPlot(                        "ec262c2a-ad91-4cce-afca-cb158eb43aab",                        [{"name":"Training data","x":["acceleration_from_standing","acceleration_lane","continuous_driving","crossing_intersection","crossing_roundabout","curve_left","curve_right","deceleration_lane","overtaking","stationary","targeted_braking","turn_left","turn_right"],"y":[233,3,3512,49,65,204,218,54,27,670,391,191,128],"type":"bar"},{"name":"Test data","x":["acceleration_from_standing","acceleration_lane","continuous_driving","crossing_intersection","crossing_roundabout","curve_left","curve_right","deceleration_lane","overtaking","stationary","targeted_braking","turn_left","turn_right"],"y":[69,9,512,23,11,64,66,22,5,317,121,50,56],"type":"bar"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"barmode":"stack","title":{"text":"Amount of windows per maneuver type in the train and test data"},"xaxis":{"title":{"text":"maneuver"}},"yaxis":{"title":{"text":"Amount"}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('ec262c2a-ad91-4cce-afca-cb158eb43aab');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
drop_maneuvers = ['acceleration_lane', 'overtaking', 'deceleration_lane', 'crossing_roundabout', 'crossing_intersection']


X_train, y_train, X_test, y_test = preprocessing.remove_maneuvers(X_train, y_train, X_test, y_test, drop_maneuvers)
X_train, y_train, X_test, y_test = preprocessing.remove_maneuvers(X_train, y_train, X_test, y_test, 
                                                                  'continuous_driving', 0.9)
X_train, y_train, X_test, y_test = preprocessing.remove_maneuvers(X_train, y_train, X_test, y_test, 
                                                                  'stationary', 0.8)
```

Let's have another look at the amount of windows in our new training and testing partitions. There will still be slight imbalance between our maneuver classes, but now the data set is much better suited for training and testing our model and since we are dealing with real world data it is fine not to synthetically bring the data to an absolute balanced ratio.



```python
fig = go.Figure(data=[
    go.Bar(name='Training data', x=np.unique(y_train), y=np.unique(y_train, return_counts=True)[1],
           text=np.unique(y_train, return_counts=True)[1], textposition='auto'),
    go.Bar(name='Test data', x=np.unique(y_test), y=np.unique(y_test, return_counts=True)[1],
           text=np.unique(y_test, return_counts=True)[1], textposition='auto')
])

fig.update_layout(barmode='stack', title="Amount of windows per maneuver type in the train and test data")
fig.update_xaxes(title_text='maneuver')
fig.update_yaxes(title_text='Amount')
fig.show()
```


<div>                            <div id="278de4eb-59e0-42c6-bfd5-4053297ab6e9" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("278de4eb-59e0-42c6-bfd5-4053297ab6e9")) {                    Plotly.newPlot(                        "278de4eb-59e0-42c6-bfd5-4053297ab6e9",                        [{"name":"Training data","text":[233.0,351.0,204.0,218.0,134.0,391.0,191.0,128.0],"textposition":"auto","x":["acceleration_from_standing","continuous_driving","curve_left","curve_right","stationary","targeted_braking","turn_left","turn_right"],"y":[233,351,204,218,134,391,191,128],"type":"bar"},{"name":"Test data","text":[69.0,51.0,64.0,66.0,63.0,121.0,50.0,56.0],"textposition":"auto","x":["acceleration_from_standing","continuous_driving","curve_left","curve_right","stationary","targeted_braking","turn_left","turn_right"],"y":[69,51,64,66,63,121,50,56],"type":"bar"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"barmode":"stack","title":{"text":"Amount of windows per maneuver type in the train and test data"},"xaxis":{"title":{"text":"maneuver"}},"yaxis":{"title":{"text":"Amount"}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('278de4eb-59e0-42c6-bfd5-4053297ab6e9');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


#### Label encoding with preprocessing.LabelEncoding and variable transformation with preprocessing.transform_variables

Now we have to create a label encoder so that we can extract the correct labels of the predictions later. The last step is to turn our training and testing data into the format of a PyTorch variable.


```python
encoding = preprocessing.LabelEncoding(y_train, y_test)
y_train, y_test = encoding.transform()
```


```python
X_train, y_train, X_test, y_test = preprocessing.transform_variables(X_train, y_train, X_test, y_test)
```

## Modelling
Before using the modelling module of the maneuver recognition package, we can set the device to use for training and testing our model.


```python
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
```

    Using cpu device


#### Defining model architecture and initializing the model with modelling.ManeuverModel
The modelling module can then be used to create and use a pytorch LSTM model. The base structure of the model is already defined and consists of a LSTM layer, two linear fully connected layers with a dropout of 0.3 and a final linear layer for multi class classification. Different configurations have been tested for this use case, but with the following parameters the architecture of the model can be configured individually:
- hidden_size: The number of features in the hidden state.
- lstm_layers: Number of stacked LSTM layers.
- lstm_dropout: Rate of applied dropout in LSTM layers.



```python
from maneuver_recognition import modelling


number_of_features = X_train.shape[2]
number_of_classes = len(np.unique(y_train))
hidden_size = 24
lstm_layers = 2
lstm_dropout = 0.7

model     = modelling.ManeuverModel(number_of_features, number_of_classes, hidden_size, lstm_layers, lstm_dropout).to(device)

print(model)
```

    ManeuverModel(
      (lstm): LSTM(7, 24, num_layers=2, batch_first=True, dropout=0.7)
      (full_layer1): Linear(in_features=24, out_features=64, bias=True)
      (dropout): Dropout(p=0.3, inplace=False)
      (full_layer2): Linear(in_features=64, out_features=32, bias=True)
      (classifier): Linear(in_features=32, out_features=8, bias=True)
    )


#### Training the model with modelling.train_maneuver_model

Now we can use the function train_maneuver_model to fit the model. The function uses PyTorch's DataLoader wrapper, so we can directly input our training and testing data and define a specific batch_size. We also have to set the number of epochs and define an optimizer as well as the type of loss function to use for the training process.


```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn   = nn.CrossEntropyLoss()

loss_list, accuracy_list = modelling.train_maneuver_model(model, X_train, y_train, X_test, y_test, epochs=100,
                                         batch_size=128, loss_function=loss_fn, optimizer=optimizer,
                                         device=device)
```

    Epoch 1
    -------------------------------
    Test Error: Accuracy: 12.8%, Avg loss: 2.072988 
    
    Epoch 2
    -------------------------------
    Test Error: Accuracy: 27.8%, Avg loss: 2.047849 
    
    Epoch 3
    -------------------------------
    Test Error: Accuracy: 22.6%, Avg loss: 1.986272 
    
    Epoch 4
    -------------------------------
    Test Error: Accuracy: 34.6%, Avg loss: 1.878559 
    
    Epoch 5
    -------------------------------
    Test Error: Accuracy: 37.8%, Avg loss: 1.730095 
    
    Epoch 6
    -------------------------------
    Test Error: Accuracy: 39.8%, Avg loss: 1.587386 
    
    Epoch 7
    -------------------------------
    Test Error: Accuracy: 42.4%, Avg loss: 1.457997 
    
    Epoch 8
    -------------------------------
    Test Error: Accuracy: 43.1%, Avg loss: 1.362437 
    
    Epoch 9
    -------------------------------
    Test Error: Accuracy: 44.4%, Avg loss: 1.313968 
    
    Epoch 10
    -------------------------------
    Test Error: Accuracy: 45.9%, Avg loss: 1.281142 
    
    Epoch 11
    -------------------------------
    Test Error: Accuracy: 47.6%, Avg loss: 1.245323 
    
    Epoch 12
    -------------------------------
    Test Error: Accuracy: 48.5%, Avg loss: 1.198241 
    
    Epoch 13
    -------------------------------
    Test Error: Accuracy: 50.2%, Avg loss: 1.167081 
    
    Epoch 14
    -------------------------------
    Test Error: Accuracy: 52.8%, Avg loss: 1.159455 
    
    Epoch 15
    -------------------------------
    Test Error: Accuracy: 58.5%, Avg loss: 1.087496 
    
    Epoch 16
    -------------------------------
    Test Error: Accuracy: 60.0%, Avg loss: 1.110333 
    
    Epoch 17
    -------------------------------
    Test Error: Accuracy: 60.7%, Avg loss: 1.074538 
    
    Epoch 18
    -------------------------------
    Test Error: Accuracy: 64.4%, Avg loss: 0.999354 
    
    Epoch 19
    -------------------------------
    Test Error: Accuracy: 63.5%, Avg loss: 0.983838 
    
    Epoch 20
    -------------------------------
    Test Error: Accuracy: 65.2%, Avg loss: 0.959804 
    
    Epoch 21
    -------------------------------
    Test Error: Accuracy: 64.3%, Avg loss: 0.952485 
    
    Epoch 22
    -------------------------------
    Test Error: Accuracy: 67.2%, Avg loss: 0.887889 
    
    Epoch 23
    -------------------------------
    Test Error: Accuracy: 68.7%, Avg loss: 0.858646 
    
    Epoch 24
    -------------------------------
    Test Error: Accuracy: 65.7%, Avg loss: 0.903415 
    
    Epoch 25
    -------------------------------
    Test Error: Accuracy: 69.6%, Avg loss: 0.850130 
    
    Epoch 26
    -------------------------------
    Test Error: Accuracy: 70.9%, Avg loss: 0.820661 
    
    Epoch 27
    -------------------------------
    Test Error: Accuracy: 67.2%, Avg loss: 0.857598 
    
    Epoch 28
    -------------------------------
    Test Error: Accuracy: 70.7%, Avg loss: 0.803806 
    
    Epoch 29
    -------------------------------
    Test Error: Accuracy: 70.6%, Avg loss: 0.787560 
    
    Epoch 30
    -------------------------------
    Test Error: Accuracy: 70.9%, Avg loss: 0.798270 
    
    Epoch 31
    -------------------------------
    Test Error: Accuracy: 72.4%, Avg loss: 0.786665 
    
    Epoch 32
    -------------------------------
    Test Error: Accuracy: 72.2%, Avg loss: 0.762232 
    
    Epoch 33
    -------------------------------
    Test Error: Accuracy: 71.1%, Avg loss: 0.786611 
    
    Epoch 34
    -------------------------------
    Test Error: Accuracy: 72.6%, Avg loss: 0.729715 
    
    Epoch 35
    -------------------------------
    Test Error: Accuracy: 73.1%, Avg loss: 0.748185 
    
    Epoch 36
    -------------------------------
    Test Error: Accuracy: 75.0%, Avg loss: 0.718007 
    
    Epoch 37
    -------------------------------
    Test Error: Accuracy: 75.6%, Avg loss: 0.737695 
    
    Epoch 38
    -------------------------------
    Test Error: Accuracy: 75.2%, Avg loss: 0.754207 
    
    Epoch 39
    -------------------------------
    Test Error: Accuracy: 75.7%, Avg loss: 0.705044 
    
    Epoch 40
    -------------------------------
    Test Error: Accuracy: 76.7%, Avg loss: 0.696122 
    
    Epoch 41
    -------------------------------
    Test Error: Accuracy: 75.6%, Avg loss: 0.731860 
    
    Epoch 42
    -------------------------------
    Test Error: Accuracy: 76.5%, Avg loss: 0.726888 
    
    Epoch 43
    -------------------------------
    Test Error: Accuracy: 77.6%, Avg loss: 0.677331 
    
    Epoch 44
    -------------------------------
    Test Error: Accuracy: 78.1%, Avg loss: 0.700297 
    
    Epoch 45
    -------------------------------
    Test Error: Accuracy: 77.4%, Avg loss: 0.686909 
    
    Epoch 46
    -------------------------------
    Test Error: Accuracy: 76.5%, Avg loss: 0.678183 
    
    Epoch 47
    -------------------------------
    Test Error: Accuracy: 78.3%, Avg loss: 0.690441 
    
    Epoch 48
    -------------------------------
    Test Error: Accuracy: 78.0%, Avg loss: 0.705492 
    
    Epoch 49
    -------------------------------
    Test Error: Accuracy: 78.7%, Avg loss: 0.667457 
    
    Epoch 50
    -------------------------------
    Test Error: Accuracy: 77.8%, Avg loss: 0.695554 
    
    Epoch 51
    -------------------------------
    Test Error: Accuracy: 78.7%, Avg loss: 0.667083 
    
    Epoch 52
    -------------------------------
    Test Error: Accuracy: 78.7%, Avg loss: 0.670404 
    
    Epoch 53
    -------------------------------
    Test Error: Accuracy: 78.9%, Avg loss: 0.712084 
    
    Epoch 54
    -------------------------------
    Test Error: Accuracy: 78.7%, Avg loss: 0.684064 
    
    Epoch 55
    -------------------------------
    Test Error: Accuracy: 78.9%, Avg loss: 0.667200 
    
    Epoch 56
    -------------------------------
    Test Error: Accuracy: 79.1%, Avg loss: 0.685108 
    
    Epoch 57
    -------------------------------
    Test Error: Accuracy: 79.1%, Avg loss: 0.685150 
    
    Epoch 58
    -------------------------------
    Test Error: Accuracy: 80.0%, Avg loss: 0.672102 
    
    Epoch 59
    -------------------------------
    Test Error: Accuracy: 78.9%, Avg loss: 0.660734 
    
    Epoch 60
    -------------------------------
    Test Error: Accuracy: 79.6%, Avg loss: 0.688697 
    
    Epoch 61
    -------------------------------
    Test Error: Accuracy: 80.0%, Avg loss: 0.689271 
    
    Epoch 62
    -------------------------------
    Test Error: Accuracy: 78.7%, Avg loss: 0.665909 
    
    Epoch 63
    -------------------------------
    Test Error: Accuracy: 79.6%, Avg loss: 0.661878 
    
    Epoch 64
    -------------------------------
    Test Error: Accuracy: 80.2%, Avg loss: 0.662054 
    
    Epoch 65
    -------------------------------
    Test Error: Accuracy: 80.0%, Avg loss: 0.677838 
    
    Epoch 66
    -------------------------------
    Test Error: Accuracy: 80.6%, Avg loss: 0.658495 
    
    Epoch 67
    -------------------------------
    Test Error: Accuracy: 79.6%, Avg loss: 0.693472 
    
    Epoch 68
    -------------------------------
    Test Error: Accuracy: 79.1%, Avg loss: 0.715555 
    
    Epoch 69
    -------------------------------
    Test Error: Accuracy: 81.5%, Avg loss: 0.649442 
    
    Epoch 70
    -------------------------------
    Test Error: Accuracy: 80.2%, Avg loss: 0.704396 
    
    Epoch 71
    -------------------------------
    Test Error: Accuracy: 80.9%, Avg loss: 0.676354 
    
    Epoch 72
    -------------------------------
    Test Error: Accuracy: 80.0%, Avg loss: 0.663990 
    
    Epoch 73
    -------------------------------
    Test Error: Accuracy: 79.4%, Avg loss: 0.703316 
    
    Epoch 74
    -------------------------------
    Test Error: Accuracy: 81.3%, Avg loss: 0.686300 
    
    Epoch 75
    -------------------------------
    Test Error: Accuracy: 80.6%, Avg loss: 0.690457 
    
    Epoch 76
    -------------------------------
    Test Error: Accuracy: 80.6%, Avg loss: 0.716997 
    
    Epoch 77
    -------------------------------
    Test Error: Accuracy: 79.4%, Avg loss: 0.718149 
    
    Epoch 78
    -------------------------------
    Test Error: Accuracy: 79.8%, Avg loss: 0.708140 
    
    Epoch 79
    -------------------------------
    Test Error: Accuracy: 80.4%, Avg loss: 0.709422 
    
    Epoch 80
    -------------------------------
    Test Error: Accuracy: 80.9%, Avg loss: 0.686119 
    
    Epoch 81
    -------------------------------
    Test Error: Accuracy: 80.6%, Avg loss: 0.717335 
    
    Epoch 82
    -------------------------------
    Test Error: Accuracy: 80.2%, Avg loss: 0.721800 
    
    Epoch 83
    -------------------------------
    Test Error: Accuracy: 79.8%, Avg loss: 0.707526 
    
    Epoch 84
    -------------------------------
    Test Error: Accuracy: 80.0%, Avg loss: 0.743724 
    
    Epoch 85
    -------------------------------
    Test Error: Accuracy: 81.5%, Avg loss: 0.691140 
    
    Epoch 86
    -------------------------------
    Test Error: Accuracy: 81.9%, Avg loss: 0.722117 
    
    Epoch 87
    -------------------------------
    Test Error: Accuracy: 80.4%, Avg loss: 0.715088 
    
    Epoch 88
    -------------------------------
    Test Error: Accuracy: 80.9%, Avg loss: 0.711530 
    
    Epoch 89
    -------------------------------
    Test Error: Accuracy: 80.4%, Avg loss: 0.737247 
    
    Epoch 90
    -------------------------------
    Test Error: Accuracy: 79.8%, Avg loss: 0.763219 
    
    Epoch 91
    -------------------------------
    Test Error: Accuracy: 80.9%, Avg loss: 0.772730 
    
    Epoch 92
    -------------------------------
    Test Error: Accuracy: 79.6%, Avg loss: 0.759872 
    
    Epoch 93
    -------------------------------
    Test Error: Accuracy: 81.3%, Avg loss: 0.751118 
    
    Epoch 94
    -------------------------------
    Test Error: Accuracy: 80.2%, Avg loss: 0.833243 
    
    Epoch 95
    -------------------------------
    Test Error: Accuracy: 80.7%, Avg loss: 0.730047 
    
    Epoch 96
    -------------------------------
    Test Error: Accuracy: 79.6%, Avg loss: 0.775176 
    
    Epoch 97
    -------------------------------
    Test Error: Accuracy: 79.6%, Avg loss: 0.758460 
    
    Epoch 98
    -------------------------------
    Test Error: Accuracy: 80.9%, Avg loss: 0.779982 
    
    Epoch 99
    -------------------------------
    Test Error: Accuracy: 79.8%, Avg loss: 0.822958 
    
    Epoch 100
    -------------------------------
    Test Error: Accuracy: 79.1%, Avg loss: 0.885200 
    
    Done!


## Evaluation

#### Plot validation accuracy and loss in training process with plot_training_process

By examining the evolution of validation accuracy and validation loss, we can see how good our training process works and whether we tend to some kind of over- or underfitting.


```python
from maneuver_recognition import evaluation
evaluation.plot_training_process(loss_list, accuracy_list)
```


    
![png](output_35_0.png)
    


#### Evaluate model performance using a multi class correlation matrix with evaluation.confusion_heatmap
In order to evaluate the performance of our model, we can use the function predict() from the modelling module with our test data and then compare the predicted with the actual values. This can be done by first using our already created encoding object which inherits our label_encoder. The label_encoder enables the access to the previously encoded classes and features the function inverse_transform().

With these class labels and the inverse transformed y_test and y_pred data we can use the confusion_heatmap() function of the evaluation module to plot the comparison of actual and predicted values.
 
Since we have an unbalanced multi-class use case, and the color intensity of a regular heatmap takes all fields into account, the distribution should instead be inspected row by row which can be done by setting the parameter relative to True. In this way we can inspect each class separately for the amount of correctly classified values and get a True Positive rate for every single class in the diagonal.



```python
y_pred = modelling.predict(X_test, model)

# Inverse transform the encoded y
y_test_inverse = encoding.label_encoder.inverse_transform(y_test)
y_pred_inverse = encoding.label_encoder.inverse_transform(y_pred)
classes = encoding.label_encoder.classes_

fig = evaluation.confusion_heatmap(y_test, y_pred, classes, relative=True)
fig.show()
```


<div>                            <div id="21e41119-5f9b-4738-a072-82c3675fc48f" class="plotly-graph-div" style="height:900px; width:900px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("21e41119-5f9b-4738-a072-82c3675fc48f")) {                    Plotly.newPlot(                        "21e41119-5f9b-4738-a072-82c3675fc48f",                        [{"colorscale":[[0.0,"rgb(247,251,255)"],[0.125,"rgb(222,235,247)"],[0.25,"rgb(198,219,239)"],[0.375,"rgb(158,202,225)"],[0.5,"rgb(107,174,214)"],[0.625,"rgb(66,146,198)"],[0.75,"rgb(33,113,181)"],[0.875,"rgb(8,81,156)"],[1.0,"rgb(8,48,107)"]],"x":["acceleration_from_standing","continuous_driving","curve_left","curve_right","stationary","targeted_braking","turn_left","turn_right"],"y":["acceleration_from_standing","continuous_driving","curve_left","curve_right","stationary","targeted_braking","turn_left","turn_right"],"z":[[0.855072463768116,0.0,0.0,0.0,0.043478260869565216,0.0,0.043478260869565216,0.057971014492753624],[0.058823529411764705,0.6078431372549019,0.058823529411764705,0.0392156862745098,0.0,0.23529411764705882,0.0,0.0],[0.0,0.078125,0.828125,0.046875,0.0,0.015625,0.03125,0.0],[0.030303030303030304,0.0,0.0,0.7272727272727273,0.0,0.07575757575757576,0.0,0.16666666666666666],[0.015873015873015872,0.0,0.0,0.0,0.9206349206349206,0.06349206349206349,0.0,0.0],[0.01652892561983471,0.06611570247933884,0.0,0.008264462809917356,0.01652892561983471,0.8842975206611571,0.0,0.008264462809917356],[0.14,0.0,0.0,0.0,0.0,0.08,0.78,0.0],[0.17857142857142858,0.03571428571428571,0.0,0.125,0.017857142857142856,0.07142857142857142,0.0,0.5714285714285714]],"type":"heatmap"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"yaxis":{"categoryorder":"category descending","title":{"text":"Actual"}},"title":{"text":"Confusion Heatmap (relative values)"},"xaxis":{"title":{"text":"Predicted"}},"height":900,"width":900},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('21e41119-5f9b-4738-a072-82c3675fc48f');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python

```
