---
title: EDA
notebook: EDA.ipynb
nav_include: 4
---

## Contents
{:.no_toc}
*  
{: toc}





### 1. Importing libraries



```python
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.neighbors import KNeighborsRegressor

%matplotlib inline
```


### 2. Loading the song dataset



```python
songs = pd.read_csv('user_spotify_v3.json.tracks1.csv')
print(songs.shape)
songs.head(10)
```


    (109233, 15)





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>acousticness</th>
      <th>danceability</th>
      <th>duration_ms</th>
      <th>energy</th>
      <th>instrumentalness</th>
      <th>key</th>
      <th>liveness</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>valence</th>
      <th>genres</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.77900</td>
      <td>0.587</td>
      <td>259550</td>
      <td>0.299</td>
      <td>0.000000</td>
      <td>8</td>
      <td>0.1230</td>
      <td>-7.365</td>
      <td>1</td>
      <td>0.0263</td>
      <td>94.992</td>
      <td>3</td>
      <td>0.356</td>
      <td>pop dance pop pop pop rap post-teen pop r&amp;b</td>
      <td>1bhUWB0zJMIKr9yVPrkEuI</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.24500</td>
      <td>0.638</td>
      <td>205748</td>
      <td>0.658</td>
      <td>0.000004</td>
      <td>3</td>
      <td>0.0919</td>
      <td>-6.318</td>
      <td>1</td>
      <td>0.0456</td>
      <td>105.076</td>
      <td>4</td>
      <td>0.330</td>
      <td>dance pop edm pop tropical house uk funky danc...</td>
      <td>2xmrfQpmS2iJExTlklLoAL</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.63300</td>
      <td>0.765</td>
      <td>229573</td>
      <td>0.688</td>
      <td>0.000000</td>
      <td>4</td>
      <td>0.0734</td>
      <td>-5.566</td>
      <td>1</td>
      <td>0.0841</td>
      <td>90.013</td>
      <td>4</td>
      <td>0.434</td>
      <td>pop rap rap</td>
      <td>42CeaId2XNlxugDvyqHfDf</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.12900</td>
      <td>0.720</td>
      <td>197993</td>
      <td>0.807</td>
      <td>0.000000</td>
      <td>11</td>
      <td>0.1830</td>
      <td>-4.590</td>
      <td>0</td>
      <td>0.0432</td>
      <td>124.946</td>
      <td>4</td>
      <td>0.305</td>
      <td>dance pop pop post-teen pop brostep edm progre...</td>
      <td>0tBbt8CrmxbjRP0pueQkyU</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.00413</td>
      <td>0.653</td>
      <td>202805</td>
      <td>0.718</td>
      <td>0.000000</td>
      <td>3</td>
      <td>0.0537</td>
      <td>-5.232</td>
      <td>0</td>
      <td>0.2130</td>
      <td>82.034</td>
      <td>4</td>
      <td>0.216</td>
      <td>hip hop pop rap rap southern hip hop trap musi...</td>
      <td>0OI7AFifLSoGzpb8bdBLLV</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.08350</td>
      <td>0.648</td>
      <td>190643</td>
      <td>0.608</td>
      <td>0.000000</td>
      <td>8</td>
      <td>0.1050</td>
      <td>-5.160</td>
      <td>1</td>
      <td>0.0587</td>
      <td>126.120</td>
      <td>4</td>
      <td>0.488</td>
      <td>dance pop pop pop christmas</td>
      <td>7eFmN6wnsb7WowRKAqRFfs</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.03200</td>
      <td>0.667</td>
      <td>174800</td>
      <td>0.726</td>
      <td>0.000000</td>
      <td>8</td>
      <td>0.0745</td>
      <td>-4.172</td>
      <td>1</td>
      <td>0.0540</td>
      <td>103.001</td>
      <td>4</td>
      <td>0.770</td>
      <td>dance pop pop post-teen pop big room dance pop...</td>
      <td>5Gu0PDLN4YJeW75PpBSg9p</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.31600</td>
      <td>0.661</td>
      <td>212120</td>
      <td>0.715</td>
      <td>0.000000</td>
      <td>5</td>
      <td>0.1780</td>
      <td>-5.651</td>
      <td>0</td>
      <td>0.1190</td>
      <td>148.027</td>
      <td>4</td>
      <td>0.411</td>
      <td>NaN</td>
      <td>2amzBJRBPOGszBem4FedfE</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.02330</td>
      <td>0.845</td>
      <td>187521</td>
      <td>0.709</td>
      <td>0.000000</td>
      <td>10</td>
      <td>0.0940</td>
      <td>-4.547</td>
      <td>0</td>
      <td>0.0714</td>
      <td>98.062</td>
      <td>4</td>
      <td>0.620</td>
      <td>dance pop pop pop rap post-teen pop pop rap ra...</td>
      <td>2z4pcBLQXF2BXKFvd0BuB6</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.39800</td>
      <td>0.751</td>
      <td>199095</td>
      <td>0.579</td>
      <td>0.000023</td>
      <td>2</td>
      <td>0.1330</td>
      <td>-4.036</td>
      <td>1</td>
      <td>0.0321</td>
      <td>105.031</td>
      <td>4</td>
      <td>0.349</td>
      <td>dance pop pop post-teen pop latin latin hip ho...</td>
      <td>3whrwq4DtvucphBPUogRuJ</td>
    </tr>
  </tbody>
</table>
</div>





```python
songs.describe()
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>acousticness</th>
      <th>danceability</th>
      <th>duration_ms</th>
      <th>energy</th>
      <th>instrumentalness</th>
      <th>key</th>
      <th>liveness</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>valence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>109233.000000</td>
      <td>109233.000000</td>
      <td>1.092330e+05</td>
      <td>109233.000000</td>
      <td>109233.000000</td>
      <td>109233.000000</td>
      <td>109233.000000</td>
      <td>109233.000000</td>
      <td>109233.000000</td>
      <td>109233.000000</td>
      <td>109233.000000</td>
      <td>109233.000000</td>
      <td>109233.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.380904</td>
      <td>0.552116</td>
      <td>2.691760e+05</td>
      <td>0.543602</td>
      <td>0.148816</td>
      <td>5.229436</td>
      <td>0.194541</td>
      <td>-10.618803</td>
      <td>0.664369</td>
      <td>0.151028</td>
      <td>117.322210</td>
      <td>3.850732</td>
      <td>0.440351</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.363885</td>
      <td>0.174904</td>
      <td>2.462208e+05</td>
      <td>0.286918</td>
      <td>0.301031</td>
      <td>3.561608</td>
      <td>0.167634</td>
      <td>7.077973</td>
      <td>0.472213</td>
      <td>0.245035</td>
      <td>30.634027</td>
      <td>0.568357</td>
      <td>0.252394</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.155000e+03</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-58.555000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.029300</td>
      <td>0.443000</td>
      <td>1.894930e+05</td>
      <td>0.303000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.096800</td>
      <td>-13.548000</td>
      <td>0.000000</td>
      <td>0.035400</td>
      <td>94.995000</td>
      <td>4.000000</td>
      <td>0.236000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.248000</td>
      <td>0.574000</td>
      <td>2.239620e+05</td>
      <td>0.576000</td>
      <td>0.000109</td>
      <td>5.000000</td>
      <td>0.125000</td>
      <td>-8.279000</td>
      <td>1.000000</td>
      <td>0.048800</td>
      <td>116.538000</td>
      <td>4.000000</td>
      <td>0.417000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.753000</td>
      <td>0.678000</td>
      <td>2.720000e+05</td>
      <td>0.792000</td>
      <td>0.049600</td>
      <td>8.000000</td>
      <td>0.237000</td>
      <td>-5.577000</td>
      <td>1.000000</td>
      <td>0.103000</td>
      <td>135.522000</td>
      <td>4.000000</td>
      <td>0.632000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.996000</td>
      <td>0.985000</td>
      <td>5.925082e+06</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>11.000000</td>
      <td>0.997000</td>
      <td>1.974000</td>
      <td>1.000000</td>
      <td>0.969000</td>
      <td>232.690000</td>
      <td>5.000000</td>
      <td>0.999000</td>
    </tr>
  </tbody>
</table>
</div>





```python
# Removing duplicate rows and rows with null values
print("Original shape: {}".format(songs.shape))
songs.drop_duplicates(inplace=True)
songs.dropna(how='any', inplace=True)
print("Shape of dataset after modifications: {}".format(songs.shape))
```


    Original shape: (109233, 15)
    Shape of dataset after modifications: (56452, 15)




```python
# Getting genres (taking the first genre of the list)
genre = []

for s in songs['genres']:
    g = s[:s.find(" ")]
    genre.append(g)
#     print(s)
    
songs['genre'] = genre
```




```python
songs.head(10)
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>acousticness</th>
      <th>danceability</th>
      <th>duration_ms</th>
      <th>energy</th>
      <th>instrumentalness</th>
      <th>key</th>
      <th>liveness</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>valence</th>
      <th>genres</th>
      <th>id</th>
      <th>genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.77900</td>
      <td>0.587</td>
      <td>259550</td>
      <td>0.299</td>
      <td>0.000000</td>
      <td>8</td>
      <td>0.1230</td>
      <td>-7.365</td>
      <td>1</td>
      <td>0.0263</td>
      <td>94.992</td>
      <td>3</td>
      <td>0.356</td>
      <td>pop dance pop pop pop rap post-teen pop r&amp;b</td>
      <td>1bhUWB0zJMIKr9yVPrkEuI</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.24500</td>
      <td>0.638</td>
      <td>205748</td>
      <td>0.658</td>
      <td>0.000004</td>
      <td>3</td>
      <td>0.0919</td>
      <td>-6.318</td>
      <td>1</td>
      <td>0.0456</td>
      <td>105.076</td>
      <td>4</td>
      <td>0.330</td>
      <td>dance pop edm pop tropical house uk funky danc...</td>
      <td>2xmrfQpmS2iJExTlklLoAL</td>
      <td>dance</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.63300</td>
      <td>0.765</td>
      <td>229573</td>
      <td>0.688</td>
      <td>0.000000</td>
      <td>4</td>
      <td>0.0734</td>
      <td>-5.566</td>
      <td>1</td>
      <td>0.0841</td>
      <td>90.013</td>
      <td>4</td>
      <td>0.434</td>
      <td>pop rap rap</td>
      <td>42CeaId2XNlxugDvyqHfDf</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.12900</td>
      <td>0.720</td>
      <td>197993</td>
      <td>0.807</td>
      <td>0.000000</td>
      <td>11</td>
      <td>0.1830</td>
      <td>-4.590</td>
      <td>0</td>
      <td>0.0432</td>
      <td>124.946</td>
      <td>4</td>
      <td>0.305</td>
      <td>dance pop pop post-teen pop brostep edm progre...</td>
      <td>0tBbt8CrmxbjRP0pueQkyU</td>
      <td>dance</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.00413</td>
      <td>0.653</td>
      <td>202805</td>
      <td>0.718</td>
      <td>0.000000</td>
      <td>3</td>
      <td>0.0537</td>
      <td>-5.232</td>
      <td>0</td>
      <td>0.2130</td>
      <td>82.034</td>
      <td>4</td>
      <td>0.216</td>
      <td>hip hop pop rap rap southern hip hop trap musi...</td>
      <td>0OI7AFifLSoGzpb8bdBLLV</td>
      <td>hip</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.08350</td>
      <td>0.648</td>
      <td>190643</td>
      <td>0.608</td>
      <td>0.000000</td>
      <td>8</td>
      <td>0.1050</td>
      <td>-5.160</td>
      <td>1</td>
      <td>0.0587</td>
      <td>126.120</td>
      <td>4</td>
      <td>0.488</td>
      <td>dance pop pop pop christmas</td>
      <td>7eFmN6wnsb7WowRKAqRFfs</td>
      <td>dance</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.03200</td>
      <td>0.667</td>
      <td>174800</td>
      <td>0.726</td>
      <td>0.000000</td>
      <td>8</td>
      <td>0.0745</td>
      <td>-4.172</td>
      <td>1</td>
      <td>0.0540</td>
      <td>103.001</td>
      <td>4</td>
      <td>0.770</td>
      <td>dance pop pop post-teen pop big room dance pop...</td>
      <td>5Gu0PDLN4YJeW75PpBSg9p</td>
      <td>dance</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.02330</td>
      <td>0.845</td>
      <td>187521</td>
      <td>0.709</td>
      <td>0.000000</td>
      <td>10</td>
      <td>0.0940</td>
      <td>-4.547</td>
      <td>0</td>
      <td>0.0714</td>
      <td>98.062</td>
      <td>4</td>
      <td>0.620</td>
      <td>dance pop pop pop rap post-teen pop pop rap ra...</td>
      <td>2z4pcBLQXF2BXKFvd0BuB6</td>
      <td>dance</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.39800</td>
      <td>0.751</td>
      <td>199095</td>
      <td>0.579</td>
      <td>0.000023</td>
      <td>2</td>
      <td>0.1330</td>
      <td>-4.036</td>
      <td>1</td>
      <td>0.0321</td>
      <td>105.031</td>
      <td>4</td>
      <td>0.349</td>
      <td>dance pop pop post-teen pop latin latin hip ho...</td>
      <td>3whrwq4DtvucphBPUogRuJ</td>
      <td>dance</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.41300</td>
      <td>0.827</td>
      <td>187250</td>
      <td>0.419</td>
      <td>0.000000</td>
      <td>10</td>
      <td>0.1150</td>
      <td>-10.329</td>
      <td>0</td>
      <td>0.1120</td>
      <td>119.974</td>
      <td>4</td>
      <td>0.227</td>
      <td>underground hip hop</td>
      <td>3al2hpm92xE0pBalqWQHdD</td>
      <td>underground</td>
    </tr>
  </tbody>
</table>
</div>





```python
songs = songs.reset_index(drop=True)
```


### Selecting four features to define similarity: acousticness, danceability, energy and liveness

We need to take a couple of steps:
1. Scale all the data
2. Select a random song for a particular genre 
3. Get the closest X songs on those features (by euclidean distance)




#### 1. Scaling the data



```python
# Getting features
features = songs.iloc[:,:(songs.shape[1]-3)]


# Scaling featues
scaler = MinMaxScaler().fit(features)
data = scaler.transform(features)
data = pd.DataFrame(data, columns= features.columns)
data['genre'] = songs['genre']
data['id'] = songs['id']
```




```python
data.head(100)
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>acousticness</th>
      <th>danceability</th>
      <th>duration_ms</th>
      <th>energy</th>
      <th>instrumentalness</th>
      <th>key</th>
      <th>liveness</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>valence</th>
      <th>genre</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.782129</td>
      <td>0.595939</td>
      <td>0.047149</td>
      <td>0.299</td>
      <td>0.000000</td>
      <td>0.727273</td>
      <td>0.123370</td>
      <td>0.827932</td>
      <td>1.0</td>
      <td>0.027198</td>
      <td>0.408234</td>
      <td>0.6</td>
      <td>0.356356</td>
      <td>pop</td>
      <td>1bhUWB0zJMIKr9yVPrkEuI</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.245984</td>
      <td>0.647716</td>
      <td>0.037332</td>
      <td>0.658</td>
      <td>0.000004</td>
      <td>0.272727</td>
      <td>0.092177</td>
      <td>0.847222</td>
      <td>1.0</td>
      <td>0.047156</td>
      <td>0.451571</td>
      <td>0.8</td>
      <td>0.330330</td>
      <td>dance</td>
      <td>2xmrfQpmS2iJExTlklLoAL</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.635542</td>
      <td>0.776650</td>
      <td>0.041680</td>
      <td>0.688</td>
      <td>0.000000</td>
      <td>0.363636</td>
      <td>0.073621</td>
      <td>0.861078</td>
      <td>1.0</td>
      <td>0.086970</td>
      <td>0.386837</td>
      <td>0.8</td>
      <td>0.434434</td>
      <td>pop</td>
      <td>42CeaId2XNlxugDvyqHfDf</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.129518</td>
      <td>0.730964</td>
      <td>0.035917</td>
      <td>0.807</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.183551</td>
      <td>0.879060</td>
      <td>0.0</td>
      <td>0.044674</td>
      <td>0.536963</td>
      <td>0.8</td>
      <td>0.305305</td>
      <td>dance</td>
      <td>0tBbt8CrmxbjRP0pueQkyU</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.004147</td>
      <td>0.662944</td>
      <td>0.036795</td>
      <td>0.718</td>
      <td>0.000000</td>
      <td>0.272727</td>
      <td>0.053862</td>
      <td>0.867232</td>
      <td>0.0</td>
      <td>0.220269</td>
      <td>0.352546</td>
      <td>0.8</td>
      <td>0.216216</td>
      <td>hip</td>
      <td>0OI7AFifLSoGzpb8bdBLLV</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.083835</td>
      <td>0.657868</td>
      <td>0.034576</td>
      <td>0.608</td>
      <td>0.000000</td>
      <td>0.727273</td>
      <td>0.105316</td>
      <td>0.868558</td>
      <td>1.0</td>
      <td>0.060703</td>
      <td>0.542009</td>
      <td>0.8</td>
      <td>0.488488</td>
      <td>dance</td>
      <td>7eFmN6wnsb7WowRKAqRFfs</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.032129</td>
      <td>0.677157</td>
      <td>0.031685</td>
      <td>0.726</td>
      <td>0.000000</td>
      <td>0.727273</td>
      <td>0.074724</td>
      <td>0.886762</td>
      <td>1.0</td>
      <td>0.055843</td>
      <td>0.442653</td>
      <td>0.8</td>
      <td>0.770771</td>
      <td>dance</td>
      <td>5Gu0PDLN4YJeW75PpBSg9p</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.023394</td>
      <td>0.857868</td>
      <td>0.034006</td>
      <td>0.709</td>
      <td>0.000000</td>
      <td>0.909091</td>
      <td>0.094283</td>
      <td>0.879853</td>
      <td>0.0</td>
      <td>0.073837</td>
      <td>0.421428</td>
      <td>0.8</td>
      <td>0.620621</td>
      <td>dance</td>
      <td>2z4pcBLQXF2BXKFvd0BuB6</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.399598</td>
      <td>0.762437</td>
      <td>0.036118</td>
      <td>0.579</td>
      <td>0.000023</td>
      <td>0.181818</td>
      <td>0.133400</td>
      <td>0.889268</td>
      <td>1.0</td>
      <td>0.033195</td>
      <td>0.451377</td>
      <td>0.8</td>
      <td>0.349349</td>
      <td>dance</td>
      <td>3whrwq4DtvucphBPUogRuJ</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.414659</td>
      <td>0.839594</td>
      <td>0.033957</td>
      <td>0.419</td>
      <td>0.000000</td>
      <td>0.909091</td>
      <td>0.115346</td>
      <td>0.773321</td>
      <td>0.0</td>
      <td>0.115822</td>
      <td>0.515596</td>
      <td>0.8</td>
      <td>0.227227</td>
      <td>underground</td>
      <td>3al2hpm92xE0pBalqWQHdD</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.047189</td>
      <td>0.652792</td>
      <td>0.029691</td>
      <td>0.783</td>
      <td>0.000000</td>
      <td>0.909091</td>
      <td>0.083250</td>
      <td>0.844643</td>
      <td>1.0</td>
      <td>0.088521</td>
      <td>0.662186</td>
      <td>0.8</td>
      <td>0.579580</td>
      <td>dance</td>
      <td>7iDa6hUg2VgEL1o1HjmfBn</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.053614</td>
      <td>0.597970</td>
      <td>0.048850</td>
      <td>0.731</td>
      <td>0.000000</td>
      <td>0.181818</td>
      <td>0.308927</td>
      <td>0.846762</td>
      <td>1.0</td>
      <td>0.089762</td>
      <td>0.377790</td>
      <td>0.8</td>
      <td>0.191191</td>
      <td>pop</td>
      <td>3YU6vJbjYUG0tiJyXf9x5V</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.077008</td>
      <td>0.808122</td>
      <td>0.036256</td>
      <td>0.606</td>
      <td>0.000003</td>
      <td>0.454545</td>
      <td>0.086560</td>
      <td>0.866826</td>
      <td>0.0</td>
      <td>0.067942</td>
      <td>0.472762</td>
      <td>0.8</td>
      <td>0.420420</td>
      <td>dance</td>
      <td>04JL2liXXV9B9coeGuUsPw</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.078012</td>
      <td>0.581726</td>
      <td>0.040543</td>
      <td>0.543</td>
      <td>0.000000</td>
      <td>0.727273</td>
      <td>0.199599</td>
      <td>0.864929</td>
      <td>0.0</td>
      <td>0.041158</td>
      <td>0.618488</td>
      <td>0.8</td>
      <td>0.308308</td>
      <td>pop</td>
      <td>75ZvA4QfFiZvzhj2xkaWAh</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.108434</td>
      <td>0.462944</td>
      <td>0.031265</td>
      <td>0.833</td>
      <td>0.000017</td>
      <td>0.000000</td>
      <td>0.059378</td>
      <td>0.885472</td>
      <td>1.0</td>
      <td>0.099690</td>
      <td>0.686785</td>
      <td>0.8</td>
      <td>0.261261</td>
      <td>bmore</td>
      <td>4utgozKpeqZA18lFCcg70Q</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.011747</td>
      <td>0.850761</td>
      <td>0.044565</td>
      <td>0.771</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.085557</td>
      <td>0.893782</td>
      <td>1.0</td>
      <td>0.252327</td>
      <td>0.756186</td>
      <td>0.8</td>
      <td>0.405405</td>
      <td>pop</td>
      <td>2Xqd0wUttjueBfdcltADOv</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.239960</td>
      <td>0.950254</td>
      <td>0.022426</td>
      <td>0.523</td>
      <td>0.000000</td>
      <td>0.454545</td>
      <td>0.117352</td>
      <td>0.840000</td>
      <td>1.0</td>
      <td>0.061737</td>
      <td>0.515231</td>
      <td>0.8</td>
      <td>0.699700</td>
      <td>dwn</td>
      <td>43ZyHQITOjhciSUUNPVRHc</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.002751</td>
      <td>0.767513</td>
      <td>0.037986</td>
      <td>0.682</td>
      <td>0.000009</td>
      <td>0.818182</td>
      <td>0.147442</td>
      <td>0.842450</td>
      <td>0.0</td>
      <td>0.077870</td>
      <td>0.498552</td>
      <td>0.8</td>
      <td>0.589590</td>
      <td>dance</td>
      <td>6tF92PMv01Ug9Dh8Rmy6nH</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.060241</td>
      <td>0.685279</td>
      <td>0.032867</td>
      <td>0.736</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.088164</td>
      <td>0.897227</td>
      <td>1.0</td>
      <td>0.031231</td>
      <td>0.508698</td>
      <td>0.8</td>
      <td>0.607608</td>
      <td>dance</td>
      <td>7y9iMe8SOB6z3NoHE2OfXl</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.189759</td>
      <td>0.861929</td>
      <td>0.035962</td>
      <td>0.581</td>
      <td>0.000004</td>
      <td>0.090909</td>
      <td>0.080742</td>
      <td>0.867029</td>
      <td>0.0</td>
      <td>0.079317</td>
      <td>0.472766</td>
      <td>0.8</td>
      <td>0.779780</td>
      <td>dance</td>
      <td>32DGGj6KlNuBr6WaqRxpxi</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.180723</td>
      <td>0.786802</td>
      <td>0.035089</td>
      <td>0.679</td>
      <td>0.000073</td>
      <td>0.363636</td>
      <td>0.068205</td>
      <td>0.871783</td>
      <td>0.0</td>
      <td>0.139607</td>
      <td>0.713430</td>
      <td>0.8</td>
      <td>0.619620</td>
      <td>latin</td>
      <td>3Ga6eKrUFf12ouh9Yw3v2D</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.128514</td>
      <td>0.588832</td>
      <td>0.039626</td>
      <td>0.531</td>
      <td>0.000127</td>
      <td>0.454545</td>
      <td>0.143430</td>
      <td>0.841456</td>
      <td>0.0</td>
      <td>0.080248</td>
      <td>0.686690</td>
      <td>0.8</td>
      <td>0.141141</td>
      <td>pop</td>
      <td>1OmcAT5Y8eg5bUPv9qJT4R</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.654618</td>
      <td>0.708629</td>
      <td>0.036466</td>
      <td>0.375</td>
      <td>0.000000</td>
      <td>0.454545</td>
      <td>0.173521</td>
      <td>0.811092</td>
      <td>1.0</td>
      <td>0.050776</td>
      <td>0.395032</td>
      <td>0.8</td>
      <td>0.534535</td>
      <td>dance</td>
      <td>1mXVgsBdtIVeCLJnSnmtdV</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.036546</td>
      <td>0.637563</td>
      <td>0.039032</td>
      <td>0.797</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.104313</td>
      <td>0.890797</td>
      <td>0.0</td>
      <td>0.061634</td>
      <td>0.459538</td>
      <td>0.8</td>
      <td>0.321321</td>
      <td>dance</td>
      <td>7EI6Iki24tBHAMxtb4xQN2</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.149598</td>
      <td>0.893401</td>
      <td>0.031320</td>
      <td>0.428</td>
      <td>0.000051</td>
      <td>0.818182</td>
      <td>0.114343</td>
      <td>0.811073</td>
      <td>1.0</td>
      <td>0.213030</td>
      <td>0.429786</td>
      <td>0.8</td>
      <td>0.333333</td>
      <td>rap</td>
      <td>7sO5G9EABYOXQKNPNiE9NR</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.072691</td>
      <td>0.553299</td>
      <td>0.037256</td>
      <td>0.865</td>
      <td>0.000005</td>
      <td>0.545455</td>
      <td>0.921765</td>
      <td>0.915707</td>
      <td>0.0</td>
      <td>0.053981</td>
      <td>0.386746</td>
      <td>0.8</td>
      <td>0.332332</td>
      <td>big</td>
      <td>5mAxA6Q1SIym6dPNiFLUyd</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.257028</td>
      <td>0.527919</td>
      <td>0.032784</td>
      <td>0.761</td>
      <td>0.000005</td>
      <td>0.363636</td>
      <td>0.170512</td>
      <td>0.906642</td>
      <td>1.0</td>
      <td>0.088211</td>
      <td>0.610129</td>
      <td>0.8</td>
      <td>0.286286</td>
      <td>brostep</td>
      <td>7vGuf3Y35N4wmASOKLUVVU</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.044177</td>
      <td>0.712690</td>
      <td>0.041918</td>
      <td>0.729</td>
      <td>0.000006</td>
      <td>0.000000</td>
      <td>0.365095</td>
      <td>0.883980</td>
      <td>1.0</td>
      <td>0.032989</td>
      <td>0.459603</td>
      <td>0.8</td>
      <td>0.317317</td>
      <td>dance</td>
      <td>4tERsdVCLtLtrGdFBf9DGC</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.032932</td>
      <td>0.737056</td>
      <td>0.031488</td>
      <td>0.889</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.092778</td>
      <td>0.896287</td>
      <td>1.0</td>
      <td>0.044364</td>
      <td>0.412669</td>
      <td>0.8</td>
      <td>0.649650</td>
      <td>latin</td>
      <td>2hl6q70unbviGo3g1R7uFx</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.234940</td>
      <td>0.613198</td>
      <td>0.040196</td>
      <td>0.661</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.095386</td>
      <td>0.854666</td>
      <td>1.0</td>
      <td>0.038780</td>
      <td>0.459908</td>
      <td>0.8</td>
      <td>0.506507</td>
      <td>edm</td>
      <td>5gIRPQWULwrvIt0F6pY7ph</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>70</th>
      <td>0.245984</td>
      <td>0.634518</td>
      <td>0.035633</td>
      <td>0.548</td>
      <td>0.000000</td>
      <td>0.454545</td>
      <td>0.111334</td>
      <td>0.757476</td>
      <td>0.0</td>
      <td>0.377456</td>
      <td>0.430229</td>
      <td>0.8</td>
      <td>0.517518</td>
      <td>deep</td>
      <td>5A8EYS9Z4hz4EmtU8W8kI3</td>
    </tr>
    <tr>
      <th>71</th>
      <td>0.011245</td>
      <td>0.944162</td>
      <td>0.034196</td>
      <td>0.673</td>
      <td>0.000000</td>
      <td>0.909091</td>
      <td>0.138415</td>
      <td>0.828871</td>
      <td>0.0</td>
      <td>0.075801</td>
      <td>0.580102</td>
      <td>0.8</td>
      <td>0.219219</td>
      <td>pop</td>
      <td>4EsYkJjHKMejYLp54woB9c</td>
    </tr>
    <tr>
      <th>72</th>
      <td>0.092169</td>
      <td>0.703553</td>
      <td>0.046954</td>
      <td>0.682</td>
      <td>0.000000</td>
      <td>0.818182</td>
      <td>0.186560</td>
      <td>0.850668</td>
      <td>1.0</td>
      <td>0.342296</td>
      <td>0.687395</td>
      <td>0.8</td>
      <td>0.193193</td>
      <td>dirty</td>
      <td>5gLJFqhjXwsxJdrugDIQDq</td>
    </tr>
    <tr>
      <th>73</th>
      <td>0.112450</td>
      <td>0.709645</td>
      <td>0.044285</td>
      <td>0.890</td>
      <td>0.000000</td>
      <td>0.545455</td>
      <td>0.120361</td>
      <td>0.898683</td>
      <td>1.0</td>
      <td>0.233713</td>
      <td>0.722291</td>
      <td>0.8</td>
      <td>0.721722</td>
      <td>hip</td>
      <td>1baLpIuaLdNehFwV5N3WUm</td>
    </tr>
    <tr>
      <th>74</th>
      <td>0.151606</td>
      <td>0.776650</td>
      <td>0.049752</td>
      <td>0.561</td>
      <td>0.000000</td>
      <td>0.636364</td>
      <td>0.183551</td>
      <td>0.826403</td>
      <td>0.0</td>
      <td>0.088935</td>
      <td>0.580304</td>
      <td>0.8</td>
      <td>0.202202</td>
      <td>dwn</td>
      <td>2c5D6B8oXAwc6easamdgVA</td>
    </tr>
    <tr>
      <th>75</th>
      <td>0.012349</td>
      <td>0.883249</td>
      <td>0.047314</td>
      <td>0.578</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.053059</td>
      <td>0.806541</td>
      <td>1.0</td>
      <td>0.129266</td>
      <td>0.636069</td>
      <td>0.8</td>
      <td>0.506507</td>
      <td>pop</td>
      <td>6NvSxE4pfP7TcD1OhGgaZW</td>
    </tr>
    <tr>
      <th>76</th>
      <td>0.315261</td>
      <td>0.888325</td>
      <td>0.035329</td>
      <td>0.884</td>
      <td>0.000008</td>
      <td>0.000000</td>
      <td>0.128385</td>
      <td>0.860746</td>
      <td>0.0</td>
      <td>0.222337</td>
      <td>0.558769</td>
      <td>0.8</td>
      <td>0.847848</td>
      <td>dwn</td>
      <td>6l8McJSn8BgCTTewysSusR</td>
    </tr>
    <tr>
      <th>77</th>
      <td>0.011044</td>
      <td>0.960406</td>
      <td>0.021103</td>
      <td>0.520</td>
      <td>0.000006</td>
      <td>0.090909</td>
      <td>0.081645</td>
      <td>0.902662</td>
      <td>1.0</td>
      <td>0.138573</td>
      <td>0.601444</td>
      <td>0.8</td>
      <td>0.859860</td>
      <td>dwn</td>
      <td>7rf7lJOYCRFpWnbRJE4C1w</td>
    </tr>
    <tr>
      <th>78</th>
      <td>0.181727</td>
      <td>0.644670</td>
      <td>0.039741</td>
      <td>0.777</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.358074</td>
      <td>0.852750</td>
      <td>1.0</td>
      <td>0.238883</td>
      <td>0.550475</td>
      <td>0.8</td>
      <td>0.177177</td>
      <td>hip</td>
      <td>0RyA3o15NOLJYtm9NlDu5c</td>
    </tr>
    <tr>
      <th>79</th>
      <td>0.301205</td>
      <td>0.883249</td>
      <td>0.041975</td>
      <td>0.674</td>
      <td>0.000000</td>
      <td>0.545455</td>
      <td>0.107322</td>
      <td>0.866863</td>
      <td>1.0</td>
      <td>0.279214</td>
      <td>0.601822</td>
      <td>0.8</td>
      <td>0.553554</td>
      <td>deep</td>
      <td>24lORMRGMv9sXpZJdN1PVm</td>
    </tr>
    <tr>
      <th>80</th>
      <td>0.000272</td>
      <td>0.511675</td>
      <td>0.040881</td>
      <td>0.841</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.284855</td>
      <td>0.905408</td>
      <td>1.0</td>
      <td>0.122027</td>
      <td>0.605243</td>
      <td>0.8</td>
      <td>0.246246</td>
      <td>crunk</td>
      <td>23eRgjXUAUE9vuZmMm99EN</td>
    </tr>
    <tr>
      <th>81</th>
      <td>0.164659</td>
      <td>0.321827</td>
      <td>0.037512</td>
      <td>0.806</td>
      <td>0.000000</td>
      <td>0.818182</td>
      <td>0.332999</td>
      <td>0.832575</td>
      <td>1.0</td>
      <td>0.387797</td>
      <td>0.663144</td>
      <td>0.8</td>
      <td>0.369369</td>
      <td>crunk</td>
      <td>0NKh1STZG1VgnVwntJF3ze</td>
    </tr>
    <tr>
      <th>82</th>
      <td>0.897590</td>
      <td>0.493401</td>
      <td>0.032236</td>
      <td>0.463</td>
      <td>0.000000</td>
      <td>0.272727</td>
      <td>0.182548</td>
      <td>0.795873</td>
      <td>1.0</td>
      <td>0.089245</td>
      <td>0.623340</td>
      <td>0.8</td>
      <td>0.681682</td>
      <td>crunk</td>
      <td>61fOn76qpka9rHRbQ6Vxxv</td>
    </tr>
    <tr>
      <th>83</th>
      <td>0.157631</td>
      <td>0.704569</td>
      <td>0.034756</td>
      <td>0.720</td>
      <td>0.000021</td>
      <td>0.272727</td>
      <td>0.353059</td>
      <td>0.839668</td>
      <td>0.0</td>
      <td>0.433299</td>
      <td>0.382995</td>
      <td>0.8</td>
      <td>0.329329</td>
      <td>east</td>
      <td>32aYDW8Qdnv1ur89TUlDnm</td>
    </tr>
    <tr>
      <th>84</th>
      <td>0.291165</td>
      <td>0.810152</td>
      <td>0.038728</td>
      <td>0.582</td>
      <td>0.000000</td>
      <td>0.909091</td>
      <td>0.153460</td>
      <td>0.827655</td>
      <td>1.0</td>
      <td>0.106515</td>
      <td>0.541815</td>
      <td>0.8</td>
      <td>0.766767</td>
      <td>hip</td>
      <td>6PGoSes0D9eUDeeAafB2As</td>
    </tr>
    <tr>
      <th>85</th>
      <td>0.007530</td>
      <td>0.678173</td>
      <td>0.044156</td>
      <td>0.503</td>
      <td>0.000000</td>
      <td>0.636364</td>
      <td>0.442327</td>
      <td>0.812621</td>
      <td>0.0</td>
      <td>0.377456</td>
      <td>0.687842</td>
      <td>0.8</td>
      <td>0.333333</td>
      <td>deep</td>
      <td>0luqG2fhUz17ncifrWqrut</td>
    </tr>
    <tr>
      <th>86</th>
      <td>0.229920</td>
      <td>0.732995</td>
      <td>0.032482</td>
      <td>0.716</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.079539</td>
      <td>0.865002</td>
      <td>0.0</td>
      <td>0.196484</td>
      <td>0.691972</td>
      <td>0.8</td>
      <td>0.667668</td>
      <td>dwn</td>
      <td>1iu3UTLNy436K7S8KVOdjS</td>
    </tr>
    <tr>
      <th>87</th>
      <td>0.027811</td>
      <td>0.456853</td>
      <td>0.039648</td>
      <td>0.898</td>
      <td>0.061600</td>
      <td>0.727273</td>
      <td>0.166499</td>
      <td>0.897227</td>
      <td>0.0</td>
      <td>0.040848</td>
      <td>0.550015</td>
      <td>0.8</td>
      <td>0.220220</td>
      <td>big</td>
      <td>5EwwwdsQfKI8ZnFG93j5Zu</td>
    </tr>
    <tr>
      <th>88</th>
      <td>0.107430</td>
      <td>0.482234</td>
      <td>0.030923</td>
      <td>0.739</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.555667</td>
      <td>0.866163</td>
      <td>1.0</td>
      <td>0.051810</td>
      <td>0.550131</td>
      <td>0.8</td>
      <td>0.242242</td>
      <td>big</td>
      <td>3VQDpxMffTaggOHEeur7Tj</td>
    </tr>
    <tr>
      <th>89</th>
      <td>0.001797</td>
      <td>0.661929</td>
      <td>0.032939</td>
      <td>0.855</td>
      <td>0.000000</td>
      <td>0.181818</td>
      <td>0.040522</td>
      <td>0.874602</td>
      <td>1.0</td>
      <td>0.045605</td>
      <td>0.575912</td>
      <td>0.8</td>
      <td>0.279279</td>
      <td>big</td>
      <td>6WbADFqMvR8N5u0BvtsWQE</td>
    </tr>
    <tr>
      <th>90</th>
      <td>0.030924</td>
      <td>0.525888</td>
      <td>0.034957</td>
      <td>0.724</td>
      <td>0.000000</td>
      <td>0.727273</td>
      <td>0.116349</td>
      <td>0.847057</td>
      <td>1.0</td>
      <td>0.086763</td>
      <td>0.523804</td>
      <td>0.8</td>
      <td>0.334334</td>
      <td>big</td>
      <td>7pstxLUhvsJC3M4TxHe2aI</td>
    </tr>
    <tr>
      <th>91</th>
      <td>0.004568</td>
      <td>0.585787</td>
      <td>0.043044</td>
      <td>0.507</td>
      <td>0.000000</td>
      <td>0.909091</td>
      <td>0.149448</td>
      <td>0.834159</td>
      <td>0.0</td>
      <td>0.037849</td>
      <td>0.549792</td>
      <td>0.8</td>
      <td>0.245245</td>
      <td>big</td>
      <td>0ZlJbNS7AU7FUSRF4pl7Sk</td>
    </tr>
    <tr>
      <th>92</th>
      <td>0.369478</td>
      <td>0.581726</td>
      <td>0.035946</td>
      <td>0.742</td>
      <td>0.002390</td>
      <td>0.454545</td>
      <td>0.285858</td>
      <td>0.906716</td>
      <td>1.0</td>
      <td>0.046846</td>
      <td>0.688074</td>
      <td>0.8</td>
      <td>0.330330</td>
      <td>bass</td>
      <td>6QcIPGlQpohjUNPpIv1OsB</td>
    </tr>
    <tr>
      <th>93</th>
      <td>0.229920</td>
      <td>0.605076</td>
      <td>0.042010</td>
      <td>0.731</td>
      <td>0.000000</td>
      <td>0.181818</td>
      <td>0.236710</td>
      <td>0.876942</td>
      <td>1.0</td>
      <td>0.043433</td>
      <td>0.450905</td>
      <td>0.8</td>
      <td>0.369369</td>
      <td>big</td>
      <td>47165S9Ppynsd9rIzVG4uS</td>
    </tr>
    <tr>
      <th>94</th>
      <td>0.436747</td>
      <td>0.534010</td>
      <td>0.033298</td>
      <td>0.520</td>
      <td>0.000000</td>
      <td>0.545455</td>
      <td>0.197593</td>
      <td>0.816509</td>
      <td>0.0</td>
      <td>0.058842</td>
      <td>0.640848</td>
      <td>0.8</td>
      <td>0.130130</td>
      <td>big</td>
      <td>0OlnLZY4cmQzT6ZGttvWBM</td>
    </tr>
    <tr>
      <th>95</th>
      <td>0.280120</td>
      <td>0.573604</td>
      <td>0.029543</td>
      <td>0.626</td>
      <td>0.806000</td>
      <td>1.000000</td>
      <td>0.272818</td>
      <td>0.832685</td>
      <td>0.0</td>
      <td>0.049224</td>
      <td>0.528411</td>
      <td>0.8</td>
      <td>0.511512</td>
      <td>big</td>
      <td>6nsmj4f9Iw3FXCRsNc6JVZ</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0.024598</td>
      <td>0.716751</td>
      <td>0.040639</td>
      <td>0.740</td>
      <td>0.512000</td>
      <td>0.545455</td>
      <td>0.135406</td>
      <td>0.818001</td>
      <td>1.0</td>
      <td>0.033609</td>
      <td>0.524333</td>
      <td>0.6</td>
      <td>0.444444</td>
      <td>deep</td>
      <td>1HrMWH5GUdK6Yi94rbANJA</td>
    </tr>
    <tr>
      <th>97</th>
      <td>0.008785</td>
      <td>0.596954</td>
      <td>0.034687</td>
      <td>0.935</td>
      <td>0.001170</td>
      <td>1.000000</td>
      <td>0.338014</td>
      <td>0.887536</td>
      <td>0.0</td>
      <td>0.049741</td>
      <td>0.541549</td>
      <td>0.8</td>
      <td>0.270270</td>
      <td>big</td>
      <td>2VYJbJpcMciVAE6FcfZnOp</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0.060743</td>
      <td>0.664975</td>
      <td>0.027631</td>
      <td>0.804</td>
      <td>0.000535</td>
      <td>0.727273</td>
      <td>0.382146</td>
      <td>0.866200</td>
      <td>1.0</td>
      <td>0.082523</td>
      <td>0.451029</td>
      <td>0.8</td>
      <td>0.399399</td>
      <td>big</td>
      <td>7xHfguzndSR2jOpMxzqh3r</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0.497992</td>
      <td>0.663959</td>
      <td>0.033429</td>
      <td>0.781</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.104313</td>
      <td>0.878360</td>
      <td>0.0</td>
      <td>0.053154</td>
      <td>0.653285</td>
      <td>0.8</td>
      <td>0.699700</td>
      <td>indie</td>
      <td>4yeWP0EabOteKT2MzPyeJ8</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 15 columns</p>
</div>



#### 2. Select a random song from the given genre



```python
print("Number of genres available: {}".format(len(data['genre'].unique())))
data['genre'].unique()
```


    Number of genres available: 716





    array(['pop', 'dance', 'hip', 'underground', 'bmore', 'dwn', 'latin',
           'rap', 'big', 'brostep', 'edm', 'detroit', 'drill', 'post-teen',
           'dirty', 'deep', 'east', 'ra', 'crunk', 'bass', 'indie',
           'chillwave', 'danish', 'canadian', 'irish', 'blues-rock',
           'alternative', 'alt-indie', 'escape', 'modern', 'emo', 'garage',
           'melodic', 'glam', 'folk-pop', 'contemporary', 'australian', 'lift',
           'christian', 'chicago', 'bachata', 'trap', 'reggaeton', 'cumbia',
           'reggaeto', 'colombian', 'aussietronica', 'house', 'chamber', 'boy',
           'acoustic', 'vapor', 'brooklyn', 'progressive', 'quebecoi',
           'indiecoustic', 'viral', 'channel', 'dreamo', 'folk-po', 'focu',
           'classify', 'compositional', 'new', 'ambient', 'soul', 'nu', 'bow',
           'scorecore', 'focus', 'austindie', 'funk', 'neo', 'folk', 'freak',
           'portland', 'michigan', 'chanson', 'anti-folk', 'vancouver',
           'norwegian', 'seattle', 'electroclash', 'bay', 'adult', 'tropical',
           'melodipo', 'afrobeats', 'grime', 'chillhop', 'uk', 'northern',
           'shiver', 'abstractro', 'mandibl', 'lowercas', 'art', 'acousmatic',
           'drone', 'j-ambien', 'float', 'fluxwork', 'glitc', 'fluxwor',
           'slow', 'album', 'britpop', 'bossa', 'classical', 'hollywood',
           'hollywoo', "children's", 'movie', 'broadway', 'celtic', 'brass',
           'classic', 'bubblegum', 'mellow', 'heavy', 'jazz', 'cabare', 'g',
           'jam', 'po', 'electroni', 'blues', 'christmas', 'metropopolis',
           'brill', 'country', 'boston', 'comedy', 'downtempo', 'ninja',
           'arab', 'electro', 'french', 'madcheste', 'dancehall', 'electronic',
           'speed', 'candy', 'experimental', 'permanent', 'bluegrass', 'grim',
           'dark', 'electroclas', 'avantgarde', 'acid', 'abstract', 'britpo',
           'punk', 'dance-punk', 'c86', 'no', 'madchester', 'british', 'bebop',
           'afrobeat', 'sheffield', 'german', 'shimmer', 'reggae',
           'indiecoustica', 'columbus', 'trip', 'gauze', 'etherpop',
           'indietronica', 'death', 'pixi', 'anthem', 'post-screamo', 'anime',
           'vegas', 'j-meta', 'brazilian', 'hard', 'djen', 'hardcore',
           'swedish', 'post-grung', 'post-scream', 'metalcore', 'gabba',
           'j-metal', 'j-dance', 'j-pop', 'j-pun', 'j-indie', 'j-rock',
           'j-roc', 'j-poproc', 'japanese', 'math', 'post', 'grung', 'pixie',
           'antiviral', 'psychedelic', 'belgian', 'grunge', 'comic', 'dutch',
           'wrestlin', 'ido', 'gothic', 'dream', 'israeli', 'blackgaze',
           'chaotic', 'mathcore', 'djent', 'crossover', 'beatdown', 'skate',
           'ccm', 'redneck', 'rednec', 'vintage', 'appalachian', 'west',
           'hyphy', 'african', 'kwaito', 'azontobeats', 'azont', 'azontobeat',
           'freakbea', 'dancehal', 'azonto', 'soc', 'chicano', 'gangster',
           'francoton', 'comed', 'environmental', 'chillho', 'disco',
           'freestyle', 'gospe', 'gospel', 'vocal', 'fun', 'sky', 'tracestep',
           'melbourne', 'traceste', 'hous', 'fidget', 'lilit', 'minimal',
           'catstep', 'polish', 'zapste', 'chillstep', 'microhouse', 'cantopo',
           'disc', 'filter', 'groove', 'tranc', 'canzone', 'glitch', 'ambeat',
           'chillste', 'bubble', 'balearic', 'dub', 'lati', 'eurodance',
           'complextro', 'drum', 'classif', 'broste', 'tech', 'moombahto',
           'hardstyle', 'electropowerpo', 'hardstyl', 'bouncy', 'chillwav',
           'nordic', 'destroy', 'bassline', 'funky', 'broken', 'traditional',
           'breakbeat', 'glitter', 'dubstep', 'straight', 'singer-songwriter',
           'coverchil', 'c64', 'singer-songwrite', 'jungl', 'tribal',
           'k-indie', 'abstrac', 'kompa', 'komp', 'c-pop', 'tin', 'baleari',
           'lovers', 'afrobea', 'soundtrac', 'cantautor', 'show', 'teen',
           'europop', 'twee', 'power-pop', 'synthpo', 'icelandic', 'stru',
           'broadwa', 'bounc', 'bounce', 'italian', 'strut', 'ectofolk',
           'spanish', 'metropopoli', 'cabaret', 'protopun', 'eurovision',
           'comi', 'merseybeat', 'hauntology', 'avant-garde', 'soundtrack',
           'baroque', 'bolero', 'opera', 'orchestral', 'poetry', 'serialism',
           'choral', 'harpsichor', 'tang', 'tango', 'concert', 'ragtime',
           'composition', 'serialis', 'marching', 'wind', 'light', 'library',
           'romanti', 'avant-gard', 'romantic', 'violin', 'violi', 'theme',
           'baroqu', 'cell', 'malle', 'retro', 'ld', 'early', 'tzadik',
           'laboratori', 'unblack', 'minima', 'clarine', 'consort', 'consor',
           'folklore', 'epicore', 'desi', 'hallowee', 'rockabilly', 'surf',
           'roots', 'psych', 'martial', 'jaz', 'a', 'idol', 'cool', 'cuban',
           'finnish', 'bhangra', 'filmi', 'albuquerque', 'lilith', 'cello',
           'atmospheric', 'substep', 'etherpo', 'eurovisio', 'anti-fol',
           'college', 'austindi', 'aussietronic', 'prever', 'stomp', 'ectofol',
           'fingerstyle', 'old-tim', 'lo', 'neo-rockabilly', 'la', 'mbala',
           'old-time', 'flamenco', 'bluegras', 'scottish', 'welsh', 'fol',
           'jig', 'ceilid', 'melancholi', 'neo-singer-songwrite', 'strid',
           'world', 'rock-and-roll', 'melancholia', 'neofol', 'europo', 'kiwi',
           'post-grunge', 'swiss', 'chanso', 'poetr', 'orator', 'prais',
           'soft', 'eurodanc', 'electronica', 'shant', 'doo-wo', 'doo-wop',
           'jump', 'fingerstyl', 'native', 'healing', 'environmenta',
           'meditation', 'slee', 'drif', 'sleep', 'hawaiian', 'ukulel',
           'hawaiia', 'epicor', 'exotica', 'boogie-woogie', 'turkish',
           'grupera', 'banda', 'axe', 'ecuadoria', 'trova', 'regional',
           'chilean', 'argentine', 'andean', 'j-ra', 'ambea', 'j-danc', 'j-po',
           'korean', 'k-pop', 'k-po', 'k-hop', 'k-ho', 'grave', 'k-indi',
           'freestyl', 'christelijk', 'em', 'warm', 'future', 'drift',
           'outsider', 'footwor', 'footwork', 'austrian', 'beat', 'chiptun',
           'beats', 'rock-and-rol', 'easy', 'rock', 'dixieland', 'enk',
           'swamp', 'beach', 'tribut', 'merseybea', 'freakbeat', 'doom', 'opm',
           'nashville', 'ragga', 'psychobilly', 'austropop', 'miami',
           'electropowerpop', 'melodipop', 'hauntolog', 'triangle', 'noise',
           'string', 'dixielan', 'pagode', 'malagasy', 'persian', 'worl',
           'ethiopian', 'mande', 'polyphon', 'yoi', 'desert', 'fado', 'gamela',
           'throat', 'highlif', 'boogaloo', 'hindustani', 'nueva', 'zi',
           'danspunk', 'quiet', 'perth', 'electrofo', 'ska', 'talent',
           'indian', 'monasti', 'nurser', 'kuduro', 'moombahton', 'black',
           'neo-trad', 'neo-psychedelic', 'brutal', 'thrash-groove',
           'industrial', 'cyber', 'voidgaz', 'blackgaz', 'pagan', 'depressive',
           'post-post-hardcore', 'estonian', 'symphonic', 'metal',
           'instrumental', 'neue', 'funeral', 'aggrotech', 'ethereal',
           'fallen', 'breakcore', 'russian', 'catste', 'nwobhm', 'nwobh',
           'grisly', 'post-metal', 'post-doom', 'neo-progressive', 'sludge',
           'slam', 'technical', 'cryptic', 'charred', 'power', 'reading',
           'readin', 'dram', 'anarcho-punk', 'polk', 'kids', 'scorecor',
           'arabesk', 'fast', 'pinoy', 'post-hardcor', 'lo-fi', 'ye',
           'nerdcor', 'horror', 'orgcor', 'o', 'crack', 'oi', 'euroska',
           'danspun', 'clarinet', 'wonk', 'chill', 'subste', 'coverchill',
           'zydec', 'morn', 'didgerido', 'andea', 'capoeir', 'denver', 'lds',
           'kraut', 'canterbury', 'zeuh', 'zol', 'hoerspiel', 'drama', 'pub',
           'judaic', 'musique', 'experimenta', 'lithumani', 'them', 'guidance',
           'gypsy', 'guidanc', 'motivatio', 'stride', 'orchestra', 'acousmati',
           'laboratorio', 'chinese', 'saxophon', 'chora', 'piedmont', 'athens',
           'entehno', 'e6fi', 'breton', 'darkstep', 'video', 'nintendocor',
           'beman', 'chiptune', 'gamecor', 'liquid', 'vegan', 'nerdcore',
           'flick', 'vocaloi', 'vaporwav', 'turntablis', 'merengue', 'cubato',
           'wroc', 'slavic', 'louisiana', 'guitar', 'polynesian', 'popgaz',
           'electrofox', 'highlife', 'soukous', 'south', 'du', 'jangle',
           'relaxativ', 'beatdow', 'crust', 'fak', 'post-disc', 'wrestling',
           'cc', 'outlaw', 'chalga', 'free', 'steelpa', 'balkan', 'swing',
           'texas', 'smooth', 'cowboy', 'rhythm', 'stoner', 'common', 'rumba',
           'neo-traditional', 'ballroom', 'portuguese', 'musica', 'francoto',
           'downtemp', 'psychil', 'relaxative', 'afrikaans', 'shanty',
           'barnmusik', 'mpb', 'harmonica', 'dansband', 'barnmusi', 'memphis',
           'folkmusik', 'dansban', 'p', 'mashup', 'hungarian', 'ebm',
           'operatic', 'nwothm', 'klezme', 'complextr', 'galego', 'faroese',
           'czech', 'dubste', 'mexican', 'mariachi', 'riddi'], dtype=object)





```python
selected_genre = 'hip'
N = 10

genre_data = data[data.genre==selected_genre]


ind = data[data.genre==selected_genre].index
r = np.random.choice(ind,1)[0]

```




```python
seed = data.iloc[r,:]
seed
```





    acousticness                      0.118474
    danceability                      0.679188
    duration_ms                      0.0570461
    energy                                0.49
    instrumentalness                  2.63e-05
    key                                      1
    liveness                          0.592778
    loudness                          0.819825
    mode                                     0
    speechiness                       0.438469
    tempo                             0.636405
    time_signature                         0.8
    valence                           0.258258
    genre                                  hip
    id                  3DFjTEueCBYqg7YE05eUJP
    Name: 47558, dtype: object





```python
# Getting feature values for our seed song
acousticness = seed.acousticness
danceability = seed.danceability
energy = seed.energy
liveness = seed.liveness
```


#### 3. Get the closest N songs on those features (by euclidean distance)



```python
# Calculating euclidean distance for every song with respect to the seed song
distance = []

for i in genre_data.index:
#     print(i)
    d = np.sqrt((genre_data.loc[i,'acousticness']-acousticness)**2 + (genre_data.loc[i,'danceability']-danceability)**2 + (genre_data.loc[i,'energy']-energy)**2 + (genre_data.loc[i,'liveness']-liveness)**2)
    distance.append(d)
    
distance
```





    [0.59644769566172307,
     0.61979187585450535,
     0.37768696043819994,
     0.49843560852592306,
     0.5153417097485633,
     0.46225083653822974,
     0.94826892189746992,
     0.44972138964520264,
     0.39464180823503359,
     0.42145877414732058,
     0.39309825838246171,
     0.36808859874060956,
     0.62034689830360534,
     0.49602671537480875,
     0.61412009338507889,
     0.53951990398928318,
     0.64390846005622682,
     0.52030401595752451,
     0.80687082500083684,
     0.46659297933294752,
     0.48390858579789298,
     0.32681650701324488,
     0.38813942358010439,
     0.51290289519732191,
     0.37253579273755949,
     0.57635844458091601,
     0.47994412112167167,
     0.42885534303410694,
     0.43911486486192769,
     0.45589878077567225,
     0.53275029809324981,
     0.56474699222150415,
     0.64476731481203509,
     0.57472443023361608,
     0.53045510435315979,
     0.38042844550397215,
     0.49217976999292695,
     0.49303066798414458,
     0.50690442881404441,
     0.4834311665372123,
     0.56997023056999274,
     0.49271528270149328,
     0.56358804198096091,
     0.29453824373255133,
     0.40565444562242242,
     0.58008348439581359,
     0.5103647994572299,
     0.69386747590713371,
     0.57696454778968176,
     0.6893396263479703,
     0.48713421128092893,
     0.64933405472225991,
     0.59150178039704848,
     0.56622350711862979,
     0.81603556361226592,
     0.60023490066678076,
     0.48697459083846434,
     0.65260492254194036,
     0.49381296761464061,
     0.57888356021249554,
     0.31856910506462571,
     0.20388484136212892,
     0.5959549044586997,
     0.48870874616612459,
     0.59047467196466075,
     0.55701356811782399,
     0.58733083379890116,
     0.64376490813295473,
     0.36292775324351872,
     0.58468501038916176,
     0.51452478396926393,
     1.0254416804859241,
     0.57125973307071665,
     0.70627953405560628,
     0.32564463876458138,
     0.59139627199001144,
     0.66383479073098439,
     0.51747217841643067,
     0.51693978768908222,
     0.60181894269780634,
     0.74350354871994484,
     0.62319139597083706,
     0.43612249813061438,
     0.47144313075185262,
     0.44640740168398196,
     0.72273085802720827,
     0.61928392896505624,
     0.53374755288585518,
     0.58602133679115687,
     0.3389604838789666,
     0.45396329469691676,
     0.20537485765705055,
     0.54693892910374264,
     0.69701128560981684,
     0.64433052317683204,
     0.47288500537304246,
     0.4949492385244042,
     0.58913798895005487,
     0.23560375640175407,
     0.38119432634335787,
     0.53267983171106881,
     0.62343517830705475,
     0.55579093742286778,
     0.48753318320456457,
     0.63110795494649463,
     0.29551286768817359,
     0.44694723145336923,
     0.83725613334906956,
     0.48896285190093935,
     0.49649118845863166,
     0.53586288681315819,
     0.89334652597307995,
     0.60436460310121887,
     0.4268242894735797,
     0.76563586239780423,
     0.56518777946966703,
     0.56625963264103418,
     0.54591879644595065,
     0.6517617370965485,
     0.54962241922677602,
     0.36994035405674897,
     0.52781400316341109,
     0.46927901703495922,
     0.57914858669281155,
     0.54999706183623231,
     0.3304216873345785,
     0.60069085527582622,
     0.61484284589375593,
     0.68043623652908436,
     0.53441985990221486,
     0.73194218213544382,
     0.46040460276224365,
     0.22848643324457032,
     0.44975027804607293,
     0.57484443372799732,
     0.38964868159345206,
     0.018790356232333317,
     0.44258274090662808,
     0.42273810624425673,
     0.48445489787011686,
     0.50041858468799638,
     0.61578588689287905,
     0.46200914360808915,
     0.54126265431738463,
     0.44501058776822094,
     0.64281687985268443,
     0.42831350989480405,
     1.0132202120779181,
     0.67291577900473454,
     0.46310527922020256,
     0.42185649826396177,
     0.33152216115313954,
     0.32603127421676675,
     0.59095463370604695,
     0.55825158436716837,
     0.58924304226814461,
     0.68850987758895288,
     0.84404509766122959,
     0.74672935244026595,
     0.64658636740882214,
     0.49381296761464061,
     0.54775544605983606,
     0.52678178217752847,
     0.77497059152966363,
     0.31017412094892866,
     0.56209485827098926,
     0.52064358650546427,
     0.56622350711862979,
     0.58207739794751978,
     0.66142454492973235,
     0.78058076582979119,
     0.6460635311590136,
     0.37253579273755949,
     0.60726069184518394,
     0.59787582669281414,
     0.39671047405480442,
     0.45856898329664531,
     0.56187945239031822,
     0.55261143002039692,
     0.40260168886292991,
     0.47484052627947665,
     0.4672150512559598,
     0.48267271912331311,
     0.53241577690999042,
     0.28390560572925316,
     0.26264552791571516,
     0.46032759841837867,
     0.52535225877753278,
     0.64992739223115659,
     0.56003119590087003,
     0.57497489769639509,
     0.61569993001964141,
     0.5963259301593512,
     0.36509236211978946,
     0.49151689615738758,
     0.41107439588961647,
     0.64790883548248368,
     0.40389423746211095,
     0.47763465245386488,
     0.53497137785481463,
     0.45490605506458137,
     0.22504923048194028,
     0.018790356232333317,
     0.51583351271424971,
     0.49279030144091218,
     0.61174974820579042,
     0.51058986076246615,
     0.47928317014226451,
     0.51481454778833846,
     0.65406176312074027,
     0.40590081990350207,
     0.2808740504108409,
     0.0,
     0.59904709404931755,
     0.49983665802032468,
     0.74692341459648015,
     0.61461758198339023,
     0.41317340736493208,
     0.60055172027347792,
     0.59908002109772718,
     0.17306186800134146,
     0.58143377015940545,
     0.50136981541221959,
     0.735700950410341,
     0.43290244918430321,
     0.59144510982344223,
     0.58658160572921958,
     0.39543129930841608,
     0.22846237172006514,
     0.62393684273220806,
     0.47214209748994435,
     0.56756586688035016,
     0.51949288596647025,
     0.61505284772191737,
     0.50120107349611354,
     0.37436824676333036,
     0.80186991095597704,
     0.28433227863282196,
     0.41867473614271644,
     0.57407236560812913,
     0.69651421913420142,
     0.33697541319834851,
     0.67685004824500883,
     0.522202345678214,
     0.59616103847586499,
     0.57635844458091601,
     0.49853317301026817,
     0.59680524512891497,
     0.63033629650718748,
     0.36169403301620406,
     0.56036650215879014,
     0.78905382517054878,
     0.51872483409286663,
     0.29851934905810656,
     0.47796559133803751,
     0.38380960894277855,
     0.40210487116469795,
     0.63472991715992422,
     0.34005669374783293,
     0.40079873417753847,
     0.66396855457564774,
     0.62978844277619994,
     0.36784353930166591,
     0.46345815126688955,
     0.48438908686371845,
     0.5292050492934669,
     0.5919395951572789,
     0.52956967739986749,
     0.24258066213109511,
     0.6141167070550978,
     0.57911702245003316,
     0.53584293564324625,
     0.54419863133458102,
     0.46078213449088667]





```python
genre_data = genre_data.reset_index(drop=True)
genre_data['distance'] = distance
genre_data
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>acousticness</th>
      <th>danceability</th>
      <th>duration_ms</th>
      <th>energy</th>
      <th>instrumentalness</th>
      <th>key</th>
      <th>liveness</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>valence</th>
      <th>genre</th>
      <th>id</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.004147</td>
      <td>0.662944</td>
      <td>0.036795</td>
      <td>0.718</td>
      <td>0.000000</td>
      <td>0.272727</td>
      <td>0.053862</td>
      <td>0.867232</td>
      <td>0.0</td>
      <td>0.220269</td>
      <td>0.352546</td>
      <td>0.8</td>
      <td>0.216216</td>
      <td>hip</td>
      <td>0OI7AFifLSoGzpb8bdBLLV</td>
      <td>0.596448</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.112450</td>
      <td>0.709645</td>
      <td>0.044285</td>
      <td>0.890</td>
      <td>0.000000</td>
      <td>0.545455</td>
      <td>0.120361</td>
      <td>0.898683</td>
      <td>1.0</td>
      <td>0.233713</td>
      <td>0.722291</td>
      <td>0.8</td>
      <td>0.721722</td>
      <td>hip</td>
      <td>1baLpIuaLdNehFwV5N3WUm</td>
      <td>0.619792</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.181727</td>
      <td>0.644670</td>
      <td>0.039741</td>
      <td>0.777</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.358074</td>
      <td>0.852750</td>
      <td>1.0</td>
      <td>0.238883</td>
      <td>0.550475</td>
      <td>0.8</td>
      <td>0.177177</td>
      <td>hip</td>
      <td>0RyA3o15NOLJYtm9NlDu5c</td>
      <td>0.377687</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.291165</td>
      <td>0.810152</td>
      <td>0.038728</td>
      <td>0.582</td>
      <td>0.000000</td>
      <td>0.909091</td>
      <td>0.153460</td>
      <td>0.827655</td>
      <td>1.0</td>
      <td>0.106515</td>
      <td>0.541815</td>
      <td>0.8</td>
      <td>0.766767</td>
      <td>hip</td>
      <td>6PGoSes0D9eUDeeAafB2As</td>
      <td>0.498436</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.019177</td>
      <td>0.699492</td>
      <td>0.037685</td>
      <td>0.531</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.089168</td>
      <td>0.828521</td>
      <td>0.0</td>
      <td>0.247156</td>
      <td>0.601113</td>
      <td>0.8</td>
      <td>0.366366</td>
      <td>hip</td>
      <td>5QZfSiRJhIgDlolnAK8MQF</td>
      <td>0.515342</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.095984</td>
      <td>0.684264</td>
      <td>0.037675</td>
      <td>0.536</td>
      <td>0.000045</td>
      <td>0.363636</td>
      <td>0.133400</td>
      <td>0.817024</td>
      <td>1.0</td>
      <td>0.088418</td>
      <td>0.369303</td>
      <td>0.8</td>
      <td>0.452452</td>
      <td>hip</td>
      <td>6Gd123r71KDdpH8JRdYvrh</td>
      <td>0.462251</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.862450</td>
      <td>0.389848</td>
      <td>0.041409</td>
      <td>0.633</td>
      <td>0.000000</td>
      <td>0.454545</td>
      <td>0.101304</td>
      <td>0.832004</td>
      <td>0.0</td>
      <td>0.137539</td>
      <td>0.538055</td>
      <td>0.8</td>
      <td>0.360360</td>
      <td>hip</td>
      <td>05pdoheuKPSotkjMgIVX6I</td>
      <td>0.948269</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.498996</td>
      <td>0.661929</td>
      <td>0.046078</td>
      <td>0.477</td>
      <td>0.000004</td>
      <td>0.818182</td>
      <td>0.354062</td>
      <td>0.807333</td>
      <td>1.0</td>
      <td>0.059359</td>
      <td>0.524148</td>
      <td>0.8</td>
      <td>0.443443</td>
      <td>hip</td>
      <td>0Iv5zus2xWVY8fGbMovMGn</td>
      <td>0.449721</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.010944</td>
      <td>0.475127</td>
      <td>0.055591</td>
      <td>0.800</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.512538</td>
      <td>0.840700</td>
      <td>1.0</td>
      <td>0.156153</td>
      <td>0.398552</td>
      <td>1.0</td>
      <td>0.093493</td>
      <td>hip</td>
      <td>1IjxCFAyR1ysajk10iHsKh</td>
      <td>0.394642</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.000419</td>
      <td>0.850761</td>
      <td>0.040986</td>
      <td>0.475</td>
      <td>0.714000</td>
      <td>0.363636</td>
      <td>0.226680</td>
      <td>0.862202</td>
      <td>0.0</td>
      <td>0.037229</td>
      <td>0.456474</td>
      <td>0.8</td>
      <td>0.711712</td>
      <td>hip</td>
      <td>0eEgMbSzOHmkOeVuNC3E0k</td>
      <td>0.421459</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.035040</td>
      <td>0.572589</td>
      <td>0.049412</td>
      <td>0.846</td>
      <td>0.000000</td>
      <td>0.272727</td>
      <td>0.690070</td>
      <td>0.882524</td>
      <td>0.0</td>
      <td>0.339193</td>
      <td>0.653371</td>
      <td>0.8</td>
      <td>0.528529</td>
      <td>hip</td>
      <td>1kMuU3TNQvHbqvXCWBodmP</td>
      <td>0.393098</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.159639</td>
      <td>0.788832</td>
      <td>0.054560</td>
      <td>0.564</td>
      <td>0.000000</td>
      <td>0.545455</td>
      <td>0.251755</td>
      <td>0.779678</td>
      <td>0.0</td>
      <td>0.426060</td>
      <td>0.584125</td>
      <td>0.8</td>
      <td>0.669670</td>
      <td>hip</td>
      <td>2P3SLxeQHPqh8qKB6gtJY2</td>
      <td>0.368089</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.403614</td>
      <td>0.560406</td>
      <td>0.058428</td>
      <td>0.403</td>
      <td>0.000004</td>
      <td>0.181818</td>
      <td>0.061886</td>
      <td>0.746384</td>
      <td>0.0</td>
      <td>0.172699</td>
      <td>0.290219</td>
      <td>0.8</td>
      <td>0.394394</td>
      <td>hip</td>
      <td>5VIUVUvwHPM2vJaQpznd5W</td>
      <td>0.620347</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.113454</td>
      <td>0.763452</td>
      <td>0.036480</td>
      <td>0.547</td>
      <td>0.000249</td>
      <td>0.090909</td>
      <td>0.107322</td>
      <td>0.806006</td>
      <td>1.0</td>
      <td>0.081386</td>
      <td>0.481164</td>
      <td>0.8</td>
      <td>0.161161</td>
      <td>hip</td>
      <td>2kwxN1whHc0YR0cBHR5iOi</td>
      <td>0.496027</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.001948</td>
      <td>0.932995</td>
      <td>0.039911</td>
      <td>0.730</td>
      <td>0.000085</td>
      <td>0.090909</td>
      <td>0.101304</td>
      <td>0.842377</td>
      <td>1.0</td>
      <td>0.091003</td>
      <td>0.408458</td>
      <td>0.8</td>
      <td>0.189189</td>
      <td>hip</td>
      <td>6tUgfsOa8PLkoede5nOR7D</td>
      <td>0.614120</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.005723</td>
      <td>0.883249</td>
      <td>0.034738</td>
      <td>0.618</td>
      <td>0.000057</td>
      <td>0.090909</td>
      <td>0.123370</td>
      <td>0.803445</td>
      <td>1.0</td>
      <td>0.258532</td>
      <td>0.425158</td>
      <td>0.8</td>
      <td>0.646647</td>
      <td>hip</td>
      <td>4wwRROf47MXBX5u5Knwixx</td>
      <td>0.539520</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.295181</td>
      <td>0.694416</td>
      <td>0.046451</td>
      <td>0.839</td>
      <td>0.000001</td>
      <td>0.727273</td>
      <td>0.081545</td>
      <td>0.915910</td>
      <td>1.0</td>
      <td>0.268873</td>
      <td>0.391663</td>
      <td>0.8</td>
      <td>0.786787</td>
      <td>hip</td>
      <td>26jvFY5r6AN5kuJmih4GpF</td>
      <td>0.643908</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.012349</td>
      <td>0.684264</td>
      <td>0.046035</td>
      <td>0.605</td>
      <td>0.000000</td>
      <td>0.727273</td>
      <td>0.096590</td>
      <td>0.842064</td>
      <td>0.0</td>
      <td>0.034643</td>
      <td>0.536852</td>
      <td>0.8</td>
      <td>0.452452</td>
      <td>hip</td>
      <td>7tGlzXJv6GD5e5qlu5YmDg</td>
      <td>0.520304</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.621486</td>
      <td>0.747208</td>
      <td>0.047574</td>
      <td>0.892</td>
      <td>0.000001</td>
      <td>0.090909</td>
      <td>0.111334</td>
      <td>0.867508</td>
      <td>1.0</td>
      <td>0.289555</td>
      <td>0.378718</td>
      <td>0.8</td>
      <td>0.597598</td>
      <td>hip</td>
      <td>2NxIp9jbmsnaIiAExWOEo0</td>
      <td>0.806871</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.398594</td>
      <td>0.786802</td>
      <td>0.044348</td>
      <td>0.707</td>
      <td>0.000000</td>
      <td>0.909091</td>
      <td>0.876630</td>
      <td>0.865260</td>
      <td>0.0</td>
      <td>0.157187</td>
      <td>0.451154</td>
      <td>0.8</td>
      <td>0.780781</td>
      <td>hip</td>
      <td>7L9g4cPfohScjJ8mGwLQWr</td>
      <td>0.466593</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.594378</td>
      <td>0.593909</td>
      <td>0.083830</td>
      <td>0.473</td>
      <td>0.000009</td>
      <td>0.090909</td>
      <td>0.603811</td>
      <td>0.833422</td>
      <td>1.0</td>
      <td>0.310238</td>
      <td>0.643216</td>
      <td>0.8</td>
      <td>0.470470</td>
      <td>hip</td>
      <td>23luOrEVHMfoX0AhfbQuS6</td>
      <td>0.483909</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.087149</td>
      <td>0.448731</td>
      <td>0.042590</td>
      <td>0.718</td>
      <td>0.000000</td>
      <td>0.909091</td>
      <td>0.619860</td>
      <td>0.894463</td>
      <td>0.0</td>
      <td>0.176836</td>
      <td>0.342022</td>
      <td>0.8</td>
      <td>0.540541</td>
      <td>hip</td>
      <td>4KGCSHN3xWAvsijrkhprFA</td>
      <td>0.326817</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.072892</td>
      <td>0.750254</td>
      <td>0.070388</td>
      <td>0.526</td>
      <td>0.000000</td>
      <td>0.363636</td>
      <td>0.215647</td>
      <td>0.827582</td>
      <td>0.0</td>
      <td>0.104447</td>
      <td>0.618626</td>
      <td>0.8</td>
      <td>0.375375</td>
      <td>hip</td>
      <td>0AOvNRgl0SMfOibWA5bP8o</td>
      <td>0.388139</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.004719</td>
      <td>0.646701</td>
      <td>0.033719</td>
      <td>0.514</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.094283</td>
      <td>0.839023</td>
      <td>1.0</td>
      <td>0.377456</td>
      <td>0.601362</td>
      <td>0.8</td>
      <td>0.402402</td>
      <td>hip</td>
      <td>6HZILIRieu8S0iqY8kIKhj</td>
      <td>0.512903</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.256024</td>
      <td>0.509645</td>
      <td>0.044219</td>
      <td>0.786</td>
      <td>0.000000</td>
      <td>0.818182</td>
      <td>0.651956</td>
      <td>0.882966</td>
      <td>0.0</td>
      <td>0.327818</td>
      <td>0.726378</td>
      <td>0.8</td>
      <td>0.739740</td>
      <td>hip</td>
      <td>4dASQiO1Eoo3RJvt74FtXB</td>
      <td>0.372536</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.008293</td>
      <td>0.848731</td>
      <td>0.055507</td>
      <td>0.670</td>
      <td>0.000355</td>
      <td>0.454545</td>
      <td>0.083952</td>
      <td>0.836370</td>
      <td>0.0</td>
      <td>0.237849</td>
      <td>0.459917</td>
      <td>0.8</td>
      <td>0.556557</td>
      <td>hip</td>
      <td>5iUQMwxUPdJBFeGkePtM66</td>
      <td>0.576358</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.111446</td>
      <td>0.823350</td>
      <td>0.043288</td>
      <td>0.691</td>
      <td>0.004830</td>
      <td>0.181818</td>
      <td>0.181545</td>
      <td>0.818443</td>
      <td>1.0</td>
      <td>0.046743</td>
      <td>0.558713</td>
      <td>0.8</td>
      <td>0.840841</td>
      <td>hip</td>
      <td>3f2k8op0nWDoZM4pXim6wG</td>
      <td>0.479944</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.244980</td>
      <td>0.707614</td>
      <td>0.046993</td>
      <td>0.771</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.295888</td>
      <td>0.847646</td>
      <td>0.0</td>
      <td>0.061944</td>
      <td>0.343491</td>
      <td>0.8</td>
      <td>0.617618</td>
      <td>hip</td>
      <td>6H99B8iTsXgCpRzGpyVgCS</td>
      <td>0.428855</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.022490</td>
      <td>0.691371</td>
      <td>0.041276</td>
      <td>0.541</td>
      <td>0.000000</td>
      <td>0.818182</td>
      <td>0.167503</td>
      <td>0.799171</td>
      <td>0.0</td>
      <td>0.056877</td>
      <td>0.476183</td>
      <td>0.8</td>
      <td>0.413413</td>
      <td>hip</td>
      <td>6SwRhMLwNqEi6alNPVG00n</td>
      <td>0.439115</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.090663</td>
      <td>0.635533</td>
      <td>0.039395</td>
      <td>0.452</td>
      <td>0.000000</td>
      <td>0.636364</td>
      <td>0.141424</td>
      <td>0.801842</td>
      <td>0.0</td>
      <td>0.057187</td>
      <td>0.313662</td>
      <td>0.8</td>
      <td>0.118118</td>
      <td>hip</td>
      <td>7M2Y6k2KSpH28HUAFamMg5</td>
      <td>0.455899</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>244</th>
      <td>0.379518</td>
      <td>0.456853</td>
      <td>0.048594</td>
      <td>0.434</td>
      <td>0.002360</td>
      <td>0.000000</td>
      <td>0.108325</td>
      <td>0.798692</td>
      <td>1.0</td>
      <td>0.030507</td>
      <td>0.599699</td>
      <td>0.8</td>
      <td>0.227227</td>
      <td>hip</td>
      <td>1fDm1kCj6dwmviArCtCsOA</td>
      <td>0.596161</td>
    </tr>
    <tr>
      <th>245</th>
      <td>0.008293</td>
      <td>0.848731</td>
      <td>0.055507</td>
      <td>0.670</td>
      <td>0.000355</td>
      <td>0.454545</td>
      <td>0.083952</td>
      <td>0.836370</td>
      <td>0.0</td>
      <td>0.237849</td>
      <td>0.459917</td>
      <td>0.8</td>
      <td>0.556557</td>
      <td>hip</td>
      <td>7L7KnCltNM8yXWs3ozuUY1</td>
      <td>0.576358</td>
    </tr>
    <tr>
      <th>246</th>
      <td>0.290161</td>
      <td>0.813198</td>
      <td>0.038728</td>
      <td>0.580</td>
      <td>0.000000</td>
      <td>0.909091</td>
      <td>0.153460</td>
      <td>0.828927</td>
      <td>1.0</td>
      <td>0.103413</td>
      <td>0.541759</td>
      <td>0.8</td>
      <td>0.777778</td>
      <td>hip</td>
      <td>5WkPWduQVV8fh7pppm1fqg</td>
      <td>0.498533</td>
    </tr>
    <tr>
      <th>247</th>
      <td>0.036647</td>
      <td>0.721827</td>
      <td>0.038076</td>
      <td>0.949</td>
      <td>0.000007</td>
      <td>1.000000</td>
      <td>0.222668</td>
      <td>0.900691</td>
      <td>1.0</td>
      <td>0.136505</td>
      <td>0.438278</td>
      <td>0.8</td>
      <td>0.711712</td>
      <td>hip</td>
      <td>519fys5P2mce1fLjig0z6l</td>
      <td>0.596805</td>
    </tr>
    <tr>
      <th>248</th>
      <td>0.333333</td>
      <td>0.867005</td>
      <td>0.048728</td>
      <td>0.759</td>
      <td>0.000377</td>
      <td>0.636364</td>
      <td>0.099298</td>
      <td>0.865058</td>
      <td>0.0</td>
      <td>0.034333</td>
      <td>0.532502</td>
      <td>0.8</td>
      <td>0.844845</td>
      <td>hip</td>
      <td>3dq91g5o0wcFKQPdpaUSlr</td>
      <td>0.630336</td>
    </tr>
    <tr>
      <th>249</th>
      <td>0.095281</td>
      <td>0.640609</td>
      <td>0.052650</td>
      <td>0.304</td>
      <td>0.000000</td>
      <td>0.909091</td>
      <td>0.285858</td>
      <td>0.738130</td>
      <td>0.0</td>
      <td>0.036091</td>
      <td>0.550170</td>
      <td>0.8</td>
      <td>0.183183</td>
      <td>hip</td>
      <td>0OH5ZegnMYV119OvFiKT9Z</td>
      <td>0.361694</td>
    </tr>
    <tr>
      <th>250</th>
      <td>0.087952</td>
      <td>0.562437</td>
      <td>0.049188</td>
      <td>0.808</td>
      <td>0.004020</td>
      <td>0.818182</td>
      <td>0.147442</td>
      <td>0.854132</td>
      <td>1.0</td>
      <td>0.315408</td>
      <td>0.397894</td>
      <td>0.8</td>
      <td>0.337337</td>
      <td>hip</td>
      <td>3XyMPyuWwHcGMvXeJY6Qt0</td>
      <td>0.560367</td>
    </tr>
    <tr>
      <th>251</th>
      <td>0.734940</td>
      <td>0.604061</td>
      <td>0.045268</td>
      <td>0.507</td>
      <td>0.000020</td>
      <td>0.636364</td>
      <td>0.106319</td>
      <td>0.818222</td>
      <td>1.0</td>
      <td>0.063909</td>
      <td>0.370016</td>
      <td>0.8</td>
      <td>0.337337</td>
      <td>hip</td>
      <td>4JOP8ELK6AaeySe7sKe996</td>
      <td>0.789054</td>
    </tr>
    <tr>
      <th>252</th>
      <td>0.006978</td>
      <td>0.665990</td>
      <td>0.045005</td>
      <td>0.446</td>
      <td>0.000419</td>
      <td>0.090909</td>
      <td>0.088265</td>
      <td>0.854979</td>
      <td>1.0</td>
      <td>0.274043</td>
      <td>0.320353</td>
      <td>0.8</td>
      <td>0.203203</td>
      <td>hip</td>
      <td>7bre6yd84LZ6MFoTppmHja</td>
      <td>0.518725</td>
    </tr>
    <tr>
      <th>253</th>
      <td>0.013554</td>
      <td>0.598985</td>
      <td>0.056486</td>
      <td>0.702</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.429288</td>
      <td>0.796739</td>
      <td>1.0</td>
      <td>0.084488</td>
      <td>0.575972</td>
      <td>0.8</td>
      <td>0.124124</td>
      <td>hip</td>
      <td>5h5tBFnbcVioFXiOixTn6E</td>
      <td>0.298519</td>
    </tr>
    <tr>
      <th>254</th>
      <td>0.317269</td>
      <td>0.973604</td>
      <td>0.039165</td>
      <td>0.524</td>
      <td>0.000000</td>
      <td>0.818182</td>
      <td>0.274824</td>
      <td>0.813561</td>
      <td>0.0</td>
      <td>0.283351</td>
      <td>0.515420</td>
      <td>0.8</td>
      <td>0.840841</td>
      <td>hip</td>
      <td>6jl01C724Mk68qlOQLco5I</td>
      <td>0.477966</td>
    </tr>
    <tr>
      <th>255</th>
      <td>0.074197</td>
      <td>0.726904</td>
      <td>0.070388</td>
      <td>0.530</td>
      <td>0.000000</td>
      <td>0.636364</td>
      <td>0.216650</td>
      <td>0.827655</td>
      <td>1.0</td>
      <td>0.136505</td>
      <td>0.309343</td>
      <td>0.8</td>
      <td>0.362362</td>
      <td>hip</td>
      <td>0UtnpKaReKUg2GquaSxCyD</td>
      <td>0.383810</td>
    </tr>
    <tr>
      <th>256</th>
      <td>0.384538</td>
      <td>0.626396</td>
      <td>0.058484</td>
      <td>0.589</td>
      <td>0.000000</td>
      <td>0.727273</td>
      <td>0.312939</td>
      <td>0.824855</td>
      <td>1.0</td>
      <td>0.124095</td>
      <td>0.331630</td>
      <td>0.8</td>
      <td>0.373373</td>
      <td>hip</td>
      <td>6pDdJM1QjrJdnIBrDN00yX</td>
      <td>0.402105</td>
    </tr>
    <tr>
      <th>257</th>
      <td>0.513052</td>
      <td>0.779695</td>
      <td>0.048716</td>
      <td>0.432</td>
      <td>0.000000</td>
      <td>0.636364</td>
      <td>0.109328</td>
      <td>0.794952</td>
      <td>0.0</td>
      <td>0.387797</td>
      <td>0.567162</td>
      <td>0.8</td>
      <td>0.672673</td>
      <td>hip</td>
      <td>2sNAjuCXxyj8jHt93t9IJ9</td>
      <td>0.634730</td>
    </tr>
    <tr>
      <th>258</th>
      <td>0.243976</td>
      <td>0.741117</td>
      <td>0.031824</td>
      <td>0.780</td>
      <td>0.000000</td>
      <td>0.636364</td>
      <td>0.483450</td>
      <td>0.835265</td>
      <td>1.0</td>
      <td>0.245088</td>
      <td>0.412020</td>
      <td>0.8</td>
      <td>0.832833</td>
      <td>hip</td>
      <td>17IuQqSZsngvxzK8mN7OeC</td>
      <td>0.340057</td>
    </tr>
    <tr>
      <th>259</th>
      <td>0.165663</td>
      <td>0.656853</td>
      <td>0.055637</td>
      <td>0.765</td>
      <td>0.000038</td>
      <td>0.454545</td>
      <td>0.305918</td>
      <td>0.832391</td>
      <td>0.0</td>
      <td>0.039193</td>
      <td>0.635889</td>
      <td>0.8</td>
      <td>0.718719</td>
      <td>hip</td>
      <td>5UwXd4AktGqBXmRJe5RvBP</td>
      <td>0.400799</td>
    </tr>
    <tr>
      <th>260</th>
      <td>0.061345</td>
      <td>0.708629</td>
      <td>0.053769</td>
      <td>0.925</td>
      <td>0.000000</td>
      <td>0.272727</td>
      <td>0.095286</td>
      <td>0.886080</td>
      <td>0.0</td>
      <td>0.229576</td>
      <td>0.416662</td>
      <td>0.8</td>
      <td>0.676677</td>
      <td>hip</td>
      <td>0xikWgPgYN9BEes0ieZ8Co</td>
      <td>0.663969</td>
    </tr>
    <tr>
      <th>261</th>
      <td>0.121486</td>
      <td>0.849746</td>
      <td>0.054825</td>
      <td>0.925</td>
      <td>0.000000</td>
      <td>0.818182</td>
      <td>0.170512</td>
      <td>0.854777</td>
      <td>0.0</td>
      <td>0.141675</td>
      <td>0.412493</td>
      <td>0.8</td>
      <td>0.737738</td>
      <td>hip</td>
      <td>0lV740rDXmB83eOTChsuQ2</td>
      <td>0.629788</td>
    </tr>
    <tr>
      <th>262</th>
      <td>0.154618</td>
      <td>0.787817</td>
      <td>0.054560</td>
      <td>0.552</td>
      <td>0.000000</td>
      <td>0.545455</td>
      <td>0.248746</td>
      <td>0.779383</td>
      <td>0.0</td>
      <td>0.385729</td>
      <td>0.584185</td>
      <td>0.8</td>
      <td>0.667668</td>
      <td>hip</td>
      <td>2AbEkP9A6XQ7iZv6zuM2EM</td>
      <td>0.367844</td>
    </tr>
    <tr>
      <th>263</th>
      <td>0.000243</td>
      <td>0.516751</td>
      <td>0.029671</td>
      <td>0.832</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.353059</td>
      <td>0.843427</td>
      <td>0.0</td>
      <td>0.172699</td>
      <td>0.650453</td>
      <td>0.8</td>
      <td>0.232232</td>
      <td>hip</td>
      <td>7Kac5ZeOc3HAFfbk3yosdp</td>
      <td>0.463458</td>
    </tr>
    <tr>
      <th>264</th>
      <td>0.474900</td>
      <td>0.482234</td>
      <td>0.035092</td>
      <td>0.667</td>
      <td>0.000000</td>
      <td>0.727273</td>
      <td>0.399198</td>
      <td>0.804459</td>
      <td>0.0</td>
      <td>0.439504</td>
      <td>0.795045</td>
      <td>0.8</td>
      <td>0.703704</td>
      <td>hip</td>
      <td>4ZQyCb1wKXCpTxSGDuWKga</td>
      <td>0.484389</td>
    </tr>
    <tr>
      <th>265</th>
      <td>0.014759</td>
      <td>0.645685</td>
      <td>0.043030</td>
      <td>0.631</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.094483</td>
      <td>0.881161</td>
      <td>0.0</td>
      <td>0.030920</td>
      <td>0.386738</td>
      <td>0.8</td>
      <td>0.561562</td>
      <td>hip</td>
      <td>1n0Tvb6HY6EGjp2WHFHtAW</td>
      <td>0.529205</td>
    </tr>
    <tr>
      <th>266</th>
      <td>0.104418</td>
      <td>0.919797</td>
      <td>0.039507</td>
      <td>0.647</td>
      <td>0.000131</td>
      <td>0.545455</td>
      <td>0.075426</td>
      <td>0.867877</td>
      <td>0.0</td>
      <td>0.113754</td>
      <td>0.455675</td>
      <td>0.8</td>
      <td>0.848849</td>
      <td>hip</td>
      <td>4Eap0FiMiGB2gqjk1D4cXo</td>
      <td>0.591940</td>
    </tr>
    <tr>
      <th>267</th>
      <td>0.066064</td>
      <td>0.731980</td>
      <td>0.038390</td>
      <td>0.518</td>
      <td>0.000113</td>
      <td>0.818182</td>
      <td>0.069208</td>
      <td>0.823252</td>
      <td>1.0</td>
      <td>0.074354</td>
      <td>0.416812</td>
      <td>0.8</td>
      <td>0.231231</td>
      <td>hip</td>
      <td>5FOAptZGxeMAO0BJSQh2fc</td>
      <td>0.529570</td>
    </tr>
    <tr>
      <th>268</th>
      <td>0.007430</td>
      <td>0.856853</td>
      <td>0.053861</td>
      <td>0.609</td>
      <td>0.000000</td>
      <td>0.909091</td>
      <td>0.564694</td>
      <td>0.766946</td>
      <td>1.0</td>
      <td>0.224405</td>
      <td>0.391581</td>
      <td>0.8</td>
      <td>0.725726</td>
      <td>hip</td>
      <td>5thts3213xwSroRd11fv5A</td>
      <td>0.242581</td>
    </tr>
    <tr>
      <th>269</th>
      <td>0.054618</td>
      <td>0.789848</td>
      <td>0.041215</td>
      <td>0.739</td>
      <td>0.000004</td>
      <td>0.454545</td>
      <td>0.046138</td>
      <td>0.869240</td>
      <td>1.0</td>
      <td>0.134436</td>
      <td>0.408415</td>
      <td>0.8</td>
      <td>0.613614</td>
      <td>hip</td>
      <td>5KxMtkdx2tBBsFVVgtQl8R</td>
      <td>0.614117</td>
    </tr>
    <tr>
      <th>270</th>
      <td>0.516064</td>
      <td>0.771574</td>
      <td>0.058417</td>
      <td>0.454</td>
      <td>0.000324</td>
      <td>0.272727</td>
      <td>0.183551</td>
      <td>0.795762</td>
      <td>1.0</td>
      <td>0.192347</td>
      <td>0.390795</td>
      <td>0.8</td>
      <td>0.670671</td>
      <td>hip</td>
      <td>5CBiUwCW1j6v5ITZYK9khy</td>
      <td>0.579117</td>
    </tr>
    <tr>
      <th>271</th>
      <td>0.130522</td>
      <td>0.623350</td>
      <td>0.045655</td>
      <td>0.326</td>
      <td>0.000318</td>
      <td>0.818182</td>
      <td>0.085858</td>
      <td>0.800018</td>
      <td>1.0</td>
      <td>0.044674</td>
      <td>0.634918</td>
      <td>0.6</td>
      <td>0.631632</td>
      <td>hip</td>
      <td>7Cbdk6g8ll93qL9QT66ABu</td>
      <td>0.535843</td>
    </tr>
    <tr>
      <th>272</th>
      <td>0.024699</td>
      <td>0.761421</td>
      <td>0.042110</td>
      <td>0.884</td>
      <td>0.000004</td>
      <td>0.090909</td>
      <td>0.238716</td>
      <td>0.864228</td>
      <td>1.0</td>
      <td>0.065564</td>
      <td>0.523937</td>
      <td>0.8</td>
      <td>0.867868</td>
      <td>hip</td>
      <td>69x16JmnJxuLVP9ELBoY03</td>
      <td>0.544199</td>
    </tr>
    <tr>
      <th>273</th>
      <td>0.017269</td>
      <td>0.735025</td>
      <td>0.039565</td>
      <td>0.691</td>
      <td>0.000017</td>
      <td>0.545455</td>
      <td>0.194584</td>
      <td>0.859254</td>
      <td>0.0</td>
      <td>0.033506</td>
      <td>0.541502</td>
      <td>0.8</td>
      <td>0.236236</td>
      <td>hip</td>
      <td>7rdjfrTBMNt3KaaGvSv3YG</td>
      <td>0.460782</td>
    </tr>
  </tbody>
</table>
<p>274 rows × 16 columns</p>
</div>





```python
genre_data = genre_data.sort_values(by=['distance'], ascending=True)
```




```python
playlist = genre_data.iloc[:N,:]
print(playlist.shape)
playlist
```


    (10, 16)





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>acousticness</th>
      <th>danceability</th>
      <th>duration_ms</th>
      <th>energy</th>
      <th>instrumentalness</th>
      <th>key</th>
      <th>liveness</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>valence</th>
      <th>genre</th>
      <th>id</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>212</th>
      <td>0.118474</td>
      <td>0.679188</td>
      <td>0.057046</td>
      <td>0.490</td>
      <td>0.000026</td>
      <td>1.000000</td>
      <td>0.592778</td>
      <td>0.819825</td>
      <td>0.0</td>
      <td>0.438469</td>
      <td>0.636405</td>
      <td>0.8</td>
      <td>0.258258</td>
      <td>hip</td>
      <td>3DFjTEueCBYqg7YE05eUJP</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>136</th>
      <td>0.119478</td>
      <td>0.690355</td>
      <td>0.057046</td>
      <td>0.489</td>
      <td>0.000029</td>
      <td>1.000000</td>
      <td>0.607823</td>
      <td>0.819235</td>
      <td>0.0</td>
      <td>0.446743</td>
      <td>0.318192</td>
      <td>0.8</td>
      <td>0.266266</td>
      <td>hip</td>
      <td>5ujh1I7NZH5agbwf7Hp8Hc</td>
      <td>0.018790</td>
    </tr>
    <tr>
      <th>202</th>
      <td>0.119478</td>
      <td>0.690355</td>
      <td>0.057046</td>
      <td>0.489</td>
      <td>0.000029</td>
      <td>1.000000</td>
      <td>0.607823</td>
      <td>0.819235</td>
      <td>0.0</td>
      <td>0.446743</td>
      <td>0.318192</td>
      <td>0.8</td>
      <td>0.266266</td>
      <td>hip</td>
      <td>41UYoaW5ErVF7bH4wuXwvo</td>
      <td>0.018790</td>
    </tr>
    <tr>
      <th>220</th>
      <td>0.011847</td>
      <td>0.809137</td>
      <td>0.055657</td>
      <td>0.521</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.565697</td>
      <td>0.812363</td>
      <td>1.0</td>
      <td>0.210962</td>
      <td>0.446882</td>
      <td>0.8</td>
      <td>0.391391</td>
      <td>hip</td>
      <td>6lVJb47gQEh3PV585qgRoy</td>
      <td>0.173062</td>
    </tr>
    <tr>
      <th>61</th>
      <td>0.037450</td>
      <td>0.714721</td>
      <td>0.080453</td>
      <td>0.606</td>
      <td>0.000000</td>
      <td>0.636364</td>
      <td>0.735206</td>
      <td>0.807370</td>
      <td>1.0</td>
      <td>0.358842</td>
      <td>0.626714</td>
      <td>0.8</td>
      <td>0.627628</td>
      <td>hip</td>
      <td>21FjKQQHLuF5jMw5F83Gb9</td>
      <td>0.203885</td>
    </tr>
    <tr>
      <th>91</th>
      <td>0.228916</td>
      <td>0.660914</td>
      <td>0.057893</td>
      <td>0.616</td>
      <td>0.000000</td>
      <td>0.818182</td>
      <td>0.710130</td>
      <td>0.826845</td>
      <td>0.0</td>
      <td>0.309204</td>
      <td>0.357059</td>
      <td>0.8</td>
      <td>0.518519</td>
      <td>hip</td>
      <td>2J2JIGPDrPip1reebfz2BL</td>
      <td>0.205375</td>
    </tr>
    <tr>
      <th>201</th>
      <td>0.001888</td>
      <td>0.702538</td>
      <td>0.044426</td>
      <td>0.661</td>
      <td>0.000000</td>
      <td>0.636364</td>
      <td>0.678034</td>
      <td>0.874325</td>
      <td>1.0</td>
      <td>0.127198</td>
      <td>0.354459</td>
      <td>0.8</td>
      <td>0.703704</td>
      <td>hip</td>
      <td>04QTusNVhoUHOx7L9jHRHZ</td>
      <td>0.225049</td>
    </tr>
    <tr>
      <th>228</th>
      <td>0.037751</td>
      <td>0.530964</td>
      <td>0.040512</td>
      <td>0.592</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.477432</td>
      <td>0.848181</td>
      <td>1.0</td>
      <td>0.500517</td>
      <td>0.337303</td>
      <td>1.0</td>
      <td>0.880881</td>
      <td>hip</td>
      <td>3Eq7yD58dIXqOgw1j7NFhY</td>
      <td>0.228462</td>
    </tr>
    <tr>
      <th>132</th>
      <td>0.114458</td>
      <td>0.772589</td>
      <td>0.047027</td>
      <td>0.696</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.560682</td>
      <td>0.856564</td>
      <td>0.0</td>
      <td>0.249224</td>
      <td>0.494035</td>
      <td>0.8</td>
      <td>0.272272</td>
      <td>hip</td>
      <td>3u9HxfcMCFYwJ2R0nkpDWV</td>
      <td>0.228486</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0.001376</td>
      <td>0.701523</td>
      <td>0.044429</td>
      <td>0.653</td>
      <td>0.000000</td>
      <td>0.636364</td>
      <td>0.714142</td>
      <td>0.873791</td>
      <td>1.0</td>
      <td>0.094105</td>
      <td>0.354601</td>
      <td>0.8</td>
      <td>0.702703</td>
      <td>hip</td>
      <td>7kMOzDgfcS6qXQHvfXfByU</td>
      <td>0.235604</td>
    </tr>
  </tbody>
</table>
</div>


