---
title: Advanced Models
notebook: Assessing alternative models - Andres.ipynb
nav_include: 3
---

## Contents
{:.no_toc}
*  
{: toc}




### 1. Importing libraries



```python
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


### 2. Loading the dataset



```python
data = pd.read_csv('data_spotify_v2.csv', index_col= 0)
print(data.shape)
print(data.columns.values)
data.head(20)
```


    (1669, 42)
    ['acousticness_avg' 'acousticness_std' 'artist_genres'
     'artist_popularity_avg' 'artist_popularity_std' 'danceability_avg'
     'danceability_std' 'duration_ms_avg' 'duration_ms_std' 'energy_avg'
     'energy_std' 'first_update' 'followers' 'instrumentalness_avg'
     'instrumentalness_std' 'is_collaborative' 'is_public' 'key_avg' 'key_std'
     'last_update' 'liveness_avg' 'liveness_std' 'loudness_avg' 'loudness_std'
     'mode_avg' 'mode_std' 'num_of_artists' 'num_of_markets_avg' 'num_of_songs'
     'playlist_name_length' 'song_duration_avg' 'song_duration_std'
     'song_popularity_avg' 'song_popularity_std' 'speechiness_avg'
     'speechiness_std' 'tempo_avg' 'tempo_std' 'time_signature_avg'
     'time_signature_std' 'valence_avg' 'valence_std']





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
      <th>acousticness_avg</th>
      <th>acousticness_std</th>
      <th>artist_genres</th>
      <th>artist_popularity_avg</th>
      <th>artist_popularity_std</th>
      <th>danceability_avg</th>
      <th>danceability_std</th>
      <th>duration_ms_avg</th>
      <th>duration_ms_std</th>
      <th>energy_avg</th>
      <th>...</th>
      <th>song_popularity_avg</th>
      <th>song_popularity_std</th>
      <th>speechiness_avg</th>
      <th>speechiness_std</th>
      <th>tempo_avg</th>
      <th>tempo_std</th>
      <th>time_signature_avg</th>
      <th>time_signature_std</th>
      <th>valence_avg</th>
      <th>valence_std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.201550</td>
      <td>0.210138</td>
      <td>35</td>
      <td>89.831579</td>
      <td>5.489996</td>
      <td>0.672320</td>
      <td>0.131615</td>
      <td>207939.980000</td>
      <td>31152.236684</td>
      <td>0.657740</td>
      <td>...</td>
      <td>80.220000</td>
      <td>16.306183</td>
      <td>0.080678</td>
      <td>0.050808</td>
      <td>120.039820</td>
      <td>27.196016</td>
      <td>4.000000</td>
      <td>0.200000</td>
      <td>0.419074</td>
      <td>0.182246</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.166715</td>
      <td>0.199305</td>
      <td>26</td>
      <td>86.978261</td>
      <td>9.056559</td>
      <td>0.748429</td>
      <td>0.133955</td>
      <td>213376.040816</td>
      <td>40248.254855</td>
      <td>0.627816</td>
      <td>...</td>
      <td>71.960000</td>
      <td>20.083785</td>
      <td>0.225245</td>
      <td>0.134817</td>
      <td>128.362633</td>
      <td>29.037111</td>
      <td>4.020408</td>
      <td>0.246593</td>
      <td>0.450592</td>
      <td>0.208395</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.115828</td>
      <td>0.128663</td>
      <td>60</td>
      <td>68.231481</td>
      <td>16.317797</td>
      <td>0.608870</td>
      <td>0.146127</td>
      <td>200221.962963</td>
      <td>45993.631187</td>
      <td>0.767889</td>
      <td>...</td>
      <td>56.714286</td>
      <td>22.988906</td>
      <td>0.061541</td>
      <td>0.041325</td>
      <td>125.398907</td>
      <td>25.782681</td>
      <td>3.888889</td>
      <td>0.566558</td>
      <td>0.368354</td>
      <td>0.202621</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.237346</td>
      <td>0.209753</td>
      <td>25</td>
      <td>79.058824</td>
      <td>12.442671</td>
      <td>0.630980</td>
      <td>0.127485</td>
      <td>227396.918367</td>
      <td>43566.385885</td>
      <td>0.543082</td>
      <td>...</td>
      <td>62.367347</td>
      <td>15.297984</td>
      <td>0.127398</td>
      <td>0.102994</td>
      <td>115.759082</td>
      <td>29.865996</td>
      <td>3.918367</td>
      <td>0.528252</td>
      <td>0.434865</td>
      <td>0.191858</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.049196</td>
      <td>0.124658</td>
      <td>70</td>
      <td>72.625000</td>
      <td>9.734993</td>
      <td>0.530340</td>
      <td>0.120331</td>
      <td>221541.943396</td>
      <td>28843.710922</td>
      <td>0.787226</td>
      <td>...</td>
      <td>59.415094</td>
      <td>10.342136</td>
      <td>0.063791</td>
      <td>0.049612</td>
      <td>126.416849</td>
      <td>25.883036</td>
      <td>3.905660</td>
      <td>0.445699</td>
      <td>0.508849</td>
      <td>0.204985</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.166265</td>
      <td>0.170598</td>
      <td>19</td>
      <td>73.586207</td>
      <td>8.202259</td>
      <td>0.597327</td>
      <td>0.090946</td>
      <td>195403.230769</td>
      <td>23419.612551</td>
      <td>0.706231</td>
      <td>...</td>
      <td>64.403846</td>
      <td>18.024833</td>
      <td>0.042362</td>
      <td>0.017098</td>
      <td>118.156308</td>
      <td>31.329474</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.581635</td>
      <td>0.159798</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.156190</td>
      <td>0.119429</td>
      <td>26</td>
      <td>88.039604</td>
      <td>7.315954</td>
      <td>0.729420</td>
      <td>0.064479</td>
      <td>214807.100000</td>
      <td>27359.847317</td>
      <td>0.768000</td>
      <td>...</td>
      <td>79.000000</td>
      <td>13.043006</td>
      <td>0.087244</td>
      <td>0.046341</td>
      <td>119.366280</td>
      <td>32.962056</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.662620</td>
      <td>0.176820</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.242398</td>
      <td>0.265416</td>
      <td>98</td>
      <td>67.247191</td>
      <td>21.125245</td>
      <td>0.605346</td>
      <td>0.162943</td>
      <td>213859.269231</td>
      <td>39274.983379</td>
      <td>0.623885</td>
      <td>...</td>
      <td>21.017241</td>
      <td>19.137877</td>
      <td>0.109023</td>
      <td>0.113717</td>
      <td>128.130865</td>
      <td>35.325679</td>
      <td>3.807692</td>
      <td>0.520298</td>
      <td>0.423398</td>
      <td>0.210862</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.786123</td>
      <td>0.199674</td>
      <td>59</td>
      <td>61.746988</td>
      <td>9.530981</td>
      <td>0.525704</td>
      <td>0.101879</td>
      <td>228501.950617</td>
      <td>50612.227546</td>
      <td>0.325552</td>
      <td>...</td>
      <td>51.074074</td>
      <td>16.487764</td>
      <td>0.033222</td>
      <td>0.006567</td>
      <td>114.429407</td>
      <td>28.322789</td>
      <td>3.913580</td>
      <td>0.449898</td>
      <td>0.320828</td>
      <td>0.129866</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.987340</td>
      <td>0.013527</td>
      <td>33</td>
      <td>64.047619</td>
      <td>8.160743</td>
      <td>0.375910</td>
      <td>0.119037</td>
      <td>175618.000000</td>
      <td>79896.697492</td>
      <td>0.067467</td>
      <td>...</td>
      <td>59.000000</td>
      <td>14.951923</td>
      <td>0.046537</td>
      <td>0.012607</td>
      <td>108.401150</td>
      <td>32.522634</td>
      <td>3.540000</td>
      <td>0.853464</td>
      <td>0.169196</td>
      <td>0.123953</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.560321</td>
      <td>0.301546</td>
      <td>69</td>
      <td>61.923810</td>
      <td>9.727423</td>
      <td>0.516560</td>
      <td>0.117606</td>
      <td>232499.400000</td>
      <td>40843.198297</td>
      <td>0.448435</td>
      <td>...</td>
      <td>50.640000</td>
      <td>13.230661</td>
      <td>0.036958</td>
      <td>0.012719</td>
      <td>115.489590</td>
      <td>31.718457</td>
      <td>3.860000</td>
      <td>0.548088</td>
      <td>0.334927</td>
      <td>0.176004</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.372465</td>
      <td>0.337896</td>
      <td>36</td>
      <td>33.390805</td>
      <td>16.108366</td>
      <td>0.502805</td>
      <td>0.111165</td>
      <td>216742.060976</td>
      <td>54364.200539</td>
      <td>0.573552</td>
      <td>...</td>
      <td>21.341463</td>
      <td>16.624416</td>
      <td>0.047393</td>
      <td>0.028185</td>
      <td>128.784439</td>
      <td>30.242586</td>
      <td>3.902439</td>
      <td>0.296720</td>
      <td>0.512171</td>
      <td>0.230420</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.220964</td>
      <td>0.234015</td>
      <td>22</td>
      <td>62.184211</td>
      <td>17.324846</td>
      <td>0.741000</td>
      <td>0.127863</td>
      <td>199848.680000</td>
      <td>25681.652485</td>
      <td>0.591960</td>
      <td>...</td>
      <td>42.000000</td>
      <td>20.135541</td>
      <td>0.188500</td>
      <td>0.114445</td>
      <td>124.331680</td>
      <td>31.597291</td>
      <td>3.880000</td>
      <td>0.587878</td>
      <td>0.502200</td>
      <td>0.192124</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.344474</td>
      <td>0.329582</td>
      <td>76</td>
      <td>48.535714</td>
      <td>14.838372</td>
      <td>0.552673</td>
      <td>0.168766</td>
      <td>218424.181818</td>
      <td>51075.544286</td>
      <td>0.567402</td>
      <td>...</td>
      <td>30.600000</td>
      <td>18.547825</td>
      <td>0.072133</td>
      <td>0.077668</td>
      <td>121.398164</td>
      <td>29.889103</td>
      <td>3.872727</td>
      <td>0.506487</td>
      <td>0.392365</td>
      <td>0.211918</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.253763</td>
      <td>0.302475</td>
      <td>73</td>
      <td>63.942857</td>
      <td>24.654113</td>
      <td>0.612220</td>
      <td>0.133341</td>
      <td>212476.780488</td>
      <td>33483.650010</td>
      <td>0.620046</td>
      <td>...</td>
      <td>23.533333</td>
      <td>14.706914</td>
      <td>0.087276</td>
      <td>0.079625</td>
      <td>119.415976</td>
      <td>27.369151</td>
      <td>3.902439</td>
      <td>0.369896</td>
      <td>0.445907</td>
      <td>0.242811</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.143884</td>
      <td>0.135943</td>
      <td>29</td>
      <td>68.750000</td>
      <td>14.247807</td>
      <td>0.807476</td>
      <td>0.122141</td>
      <td>234515.666667</td>
      <td>37237.477180</td>
      <td>0.628619</td>
      <td>...</td>
      <td>48.142857</td>
      <td>13.445784</td>
      <td>0.235300</td>
      <td>0.115372</td>
      <td>103.087476</td>
      <td>20.696904</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.646048</td>
      <td>0.166843</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.218213</td>
      <td>0.271245</td>
      <td>87</td>
      <td>57.393939</td>
      <td>15.424183</td>
      <td>0.573109</td>
      <td>0.187072</td>
      <td>219494.618182</td>
      <td>38949.349059</td>
      <td>0.667089</td>
      <td>...</td>
      <td>42.200000</td>
      <td>19.470631</td>
      <td>0.097662</td>
      <td>0.095417</td>
      <td>121.318527</td>
      <td>28.182618</td>
      <td>3.854545</td>
      <td>0.615502</td>
      <td>0.472185</td>
      <td>0.227651</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.181231</td>
      <td>0.149650</td>
      <td>61</td>
      <td>79.303030</td>
      <td>13.265322</td>
      <td>0.698060</td>
      <td>0.089967</td>
      <td>202996.200000</td>
      <td>25632.338107</td>
      <td>0.653340</td>
      <td>...</td>
      <td>72.440000</td>
      <td>14.886450</td>
      <td>0.086456</td>
      <td>0.065620</td>
      <td>112.946660</td>
      <td>18.855509</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.463562</td>
      <td>0.183308</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.200536</td>
      <td>0.266862</td>
      <td>45</td>
      <td>45.661017</td>
      <td>17.081668</td>
      <td>0.540706</td>
      <td>0.160960</td>
      <td>214371.745098</td>
      <td>37873.507204</td>
      <td>0.692353</td>
      <td>...</td>
      <td>29.019608</td>
      <td>16.364105</td>
      <td>0.098094</td>
      <td>0.090684</td>
      <td>129.127647</td>
      <td>27.876211</td>
      <td>3.980392</td>
      <td>0.138648</td>
      <td>0.429692</td>
      <td>0.228722</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.776473</td>
      <td>0.265833</td>
      <td>69</td>
      <td>27.880952</td>
      <td>15.472252</td>
      <td>0.297418</td>
      <td>0.148548</td>
      <td>332391.075000</td>
      <td>191073.114544</td>
      <td>0.337120</td>
      <td>...</td>
      <td>18.400000</td>
      <td>13.628646</td>
      <td>0.049460</td>
      <td>0.029320</td>
      <td>108.700025</td>
      <td>37.311863</td>
      <td>3.875000</td>
      <td>0.713705</td>
      <td>0.132685</td>
      <td>0.133143</td>
    </tr>
  </tbody>
</table>
<p>20 rows × 42 columns</p>
</div>





```python
# Dropping null values
data.dropna(how='any', inplace= True)
data.shape
```





    (1669, 42)





```python
data.describe()
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
      <th>acousticness_avg</th>
      <th>acousticness_std</th>
      <th>artist_genres</th>
      <th>artist_popularity_avg</th>
      <th>artist_popularity_std</th>
      <th>danceability_avg</th>
      <th>danceability_std</th>
      <th>duration_ms_avg</th>
      <th>duration_ms_std</th>
      <th>energy_avg</th>
      <th>...</th>
      <th>song_popularity_avg</th>
      <th>song_popularity_std</th>
      <th>speechiness_avg</th>
      <th>speechiness_std</th>
      <th>tempo_avg</th>
      <th>tempo_std</th>
      <th>time_signature_avg</th>
      <th>time_signature_std</th>
      <th>valence_avg</th>
      <th>valence_std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1669.000000</td>
      <td>1669.000000</td>
      <td>1669.000000</td>
      <td>1669.000000</td>
      <td>1669.000000</td>
      <td>1669.000000</td>
      <td>1669.000000</td>
      <td>1.669000e+03</td>
      <td>1.669000e+03</td>
      <td>1669.000000</td>
      <td>...</td>
      <td>1669.000000</td>
      <td>1669.000000</td>
      <td>1669.000000</td>
      <td>1669.000000</td>
      <td>1669.000000</td>
      <td>1669.000000</td>
      <td>1669.000000</td>
      <td>1669.000000</td>
      <td>1669.000000</td>
      <td>1669.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.346797</td>
      <td>0.199493</td>
      <td>47.663871</td>
      <td>59.591774</td>
      <td>12.743078</td>
      <td>0.553370</td>
      <td>0.124922</td>
      <td>2.626815e+05</td>
      <td>9.177707e+04</td>
      <td>0.577372</td>
      <td>...</td>
      <td>37.835228</td>
      <td>15.441499</td>
      <td>0.115955</td>
      <td>0.066065</td>
      <td>118.171600</td>
      <td>27.524807</td>
      <td>3.878445</td>
      <td>0.355986</td>
      <td>0.459183</td>
      <td>0.192708</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.277742</td>
      <td>0.090449</td>
      <td>34.619345</td>
      <td>16.475860</td>
      <td>5.372761</td>
      <td>0.123583</td>
      <td>0.028849</td>
      <td>1.452194e+05</td>
      <td>1.363047e+05</td>
      <td>0.222004</td>
      <td>...</td>
      <td>15.966070</td>
      <td>5.877576</td>
      <td>0.167718</td>
      <td>0.063607</td>
      <td>10.986408</td>
      <td>6.198193</td>
      <td>0.216045</td>
      <td>0.287961</td>
      <td>0.164723</td>
      <td>0.044049</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000249</td>
      <td>0.000760</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.962915e+04</td>
      <td>1.253131e+04</td>
      <td>0.023507</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.129468</td>
      <td>0.136567</td>
      <td>20.000000</td>
      <td>49.904110</td>
      <td>9.727423</td>
      <td>0.493714</td>
      <td>0.107594</td>
      <td>2.152009e+05</td>
      <td>3.995136e+04</td>
      <td>0.446835</td>
      <td>...</td>
      <td>27.040000</td>
      <td>11.368817</td>
      <td>0.047339</td>
      <td>0.026198</td>
      <td>113.432680</td>
      <td>25.097422</td>
      <td>3.840000</td>
      <td>0.156125</td>
      <td>0.358124</td>
      <td>0.170908</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.248812</td>
      <td>0.210138</td>
      <td>44.000000</td>
      <td>60.826087</td>
      <td>12.665039</td>
      <td>0.567103</td>
      <td>0.124317</td>
      <td>2.338687e+05</td>
      <td>5.337463e+04</td>
      <td>0.633250</td>
      <td>...</td>
      <td>39.200000</td>
      <td>16.060908</td>
      <td>0.066684</td>
      <td>0.049319</td>
      <td>119.038150</td>
      <td>28.229578</td>
      <td>3.936508</td>
      <td>0.294884</td>
      <td>0.466459</td>
      <td>0.197602</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.543012</td>
      <td>0.266696</td>
      <td>68.000000</td>
      <td>71.470588</td>
      <td>15.792832</td>
      <td>0.640423</td>
      <td>0.142906</td>
      <td>2.611716e+05</td>
      <td>8.071371e+04</td>
      <td>0.735967</td>
      <td>...</td>
      <td>48.632653</td>
      <td>19.942139</td>
      <td>0.097518</td>
      <td>0.083669</td>
      <td>124.541882</td>
      <td>30.933774</td>
      <td>4.000000</td>
      <td>0.521978</td>
      <td>0.573760</td>
      <td>0.219549</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.991452</td>
      <td>0.446182</td>
      <td>254.000000</td>
      <td>97.136364</td>
      <td>37.491844</td>
      <td>0.816880</td>
      <td>0.294881</td>
      <td>2.386798e+06</td>
      <td>1.430093e+06</td>
      <td>0.968585</td>
      <td>...</td>
      <td>80.220000</td>
      <td>31.302945</td>
      <td>0.935176</td>
      <td>0.434830</td>
      <td>167.132520</td>
      <td>61.910298</td>
      <td>4.166667</td>
      <td>1.785258</td>
      <td>0.877350</td>
      <td>0.340069</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 42 columns</p>
</div>



### 3. Data Transformation and creating of Train and Test sets



```python
# np.mean(data['is_public'])
```




```python
# Dropping features that present zero variance
data.drop(['is_collaborative', 'is_public'], axis =1 , inplace= True)
```




```python
# Getting features
data1 = data.copy()
X = data1.drop(['followers'], axis = 1)

# Scaling X
scaler = MinMaxScaler().fit(X)
data1 = scaler.transform(X)
data1 = pd.DataFrame(data1, columns= X.columns)
data1['followers'] = data['followers']


# Creating a dataset with ln(Y)
data3 = data1.copy()
data3['followers'] = np.log(1 + data3['followers'])


y = (data1['followers'])
X = data1.drop(['followers'], axis = 1)

y2 = (data3['followers'])
X2 = data3.drop(['followers'], axis = 1)
X2.shape, y2.shape


# Baseline model

X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.33, random_state=42)

```


### 4. Advanced Models

#### Model 1: Random Forest



```python
from collections import OrderedDict
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


param_dict = OrderedDict(
    n_estimators = [10, 30, 50, 100, 200, 500],
    max_features = [0.2, 0.4, 0.6]
)

rf = RandomForestRegressor(oob_score=False)
gs = GridSearchCV(rf, param_grid = param_dict, cv=5, n_jobs=-1)
gs.fit(X_train, y_train)
r2_test = gs.score(X_test, y_test)

print("The test R2 for the Random Forest is {:.2f}%".format(r2_test*100))
```


    The test R2 for the Random Forest is 62.53%


##### Feature Importance



```python
gs.best_estimator_.feature_importances_
plt.figure(figsize=(10,10))
pd.Series(gs.best_estimator_.feature_importances_, index=X_train.columns.values).sort_values()[27:].plot(kind="barh")
plt.title("Feature Importance")
plt.show()
```



![png](Assessing%20alternative%20models%20-%20Andres_files/Assessing%20alternative%20models%20-%20Andres_14_0.png)


####  Model 2: Gradient Boosted Regression Tree



```python
# Gradient Boosting Regression Trees (first run with one set of hyperparameters)
from sklearn.ensemble import GradientBoostingRegressor

gb = GradientBoostingRegressor(n_estimators=100, max_depth=1)
gb.fit(X_train, y_train)
```





    GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                 learning_rate=0.1, loss='ls', max_depth=1, max_features=None,
                 max_leaf_nodes=None, min_impurity_split=1e-07,
                 min_samples_leaf=1, min_samples_split=2,
                 min_weight_fraction_leaf=0.0, n_estimators=100,
                 presort='auto', random_state=None, subsample=1.0, verbose=0,
                 warm_start=False)





```python
GradientBoostingRegressor()
```





    GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                 learning_rate=0.1, loss='ls', max_depth=3, max_features=None,
                 max_leaf_nodes=None, min_impurity_split=1e-07,
                 min_samples_leaf=1, min_samples_split=2,
                 min_weight_fraction_leaf=0.0, n_estimators=100,
                 presort='auto', random_state=None, subsample=1.0, verbose=0,
                 warm_start=False)





```python
from sklearn.model_selection import GridSearchCV

param_grid = {'learning_rate': [0.1, 0.01],
              'max_depth': [3, 6],
              'min_samples_leaf': [3, 5],  ## depends on the num of training examples
              'max_features': [0.2, 0.6]
              }

gb = GradientBoostingRegressor(n_estimators=600, loss='huber')
gb_cv = GridSearchCV(gb, param_grid, cv=3, n_jobs=-1)
```




```python
gb_cv.fit(X_train, y_train)
```





    GridSearchCV(cv=3, error_score='raise',
           estimator=GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                 learning_rate=0.1, loss='huber', max_depth=3,
                 max_features=None, max_leaf_nodes=None,
                 min_impurity_split=1e-07, min_samples_leaf=1,
                 min_samples_split=2, min_weight_fraction_leaf=0.0,
                 n_estimators=600, presort='auto', random_state=None,
                 subsample=1.0, verbose=0, warm_start=False),
           fit_params={}, iid=True, n_jobs=-1,
           param_grid={'learning_rate': [0.1, 0.01], 'max_depth': [3, 6], 'min_samples_leaf': [3, 5], 'max_features': [0.2, 0.6]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
           scoring=None, verbose=0)





```python
gb_cv.best_estimator_
```





    GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                 learning_rate=0.01, loss='huber', max_depth=6,
                 max_features=0.6, max_leaf_nodes=None,
                 min_impurity_split=1e-07, min_samples_leaf=3,
                 min_samples_split=2, min_weight_fraction_leaf=0.0,
                 n_estimators=600, presort='auto', random_state=None,
                 subsample=1.0, verbose=0, warm_start=False)





```python
from sklearn.metrics import r2_score

r2_score_test = r2_score(y_test, gb_cv.predict(X_test))

print("The train R2 for the Grandient Boosted Regression Tree is {:.2f}%".format(r2_score_test*100))
```


    The train R2 for the Grandient Boosted Regression Tree is 64.44%


##### Feature Importance



```python
plt.figure(figsize=(10,10))
pd.Series(gb_cv.best_estimator_.feature_importances_, index=X_train.columns.values).sort_values()[27:].plot(kind="barh")
plt.title("Feature Importance")
plt.show()
```



![png](Assessing%20alternative%20models%20-%20Andres_files/Assessing%20alternative%20models%20-%20Andres_23_0.png)




```python
relevant = pd.Series(gb_cv.best_estimator_.feature_importances_, index=X_train.columns.values).sort_values()[27:].index
relevant
```





    Index(['danceability_avg', 'artist_popularity_avg', 'song_popularity_std',
           'artist_genres', 'tempo_std', 'num_of_artists', 'num_of_markets_avg',
           'artist_popularity_std', 'num_of_songs', 'song_popularity_avg',
           'first_update', 'last_update'],
          dtype='object')





```python
plt.figure(figsize=(20,20))

i=1

for r in relevant:
    
    plt.subplot(5,3,i)
    plt.scatter(data[r], data['followers'])
    plt.yscale('log')
    plt.ylim(ymin=1)
    plt.xlabel(r)
    plt.ylabel('Number of Followers')
        
    i += 1
```



![png](Assessing%20alternative%20models%20-%20Andres_files/Assessing%20alternative%20models%20-%20Andres_25_0.png)


The graphs above allow us to understand not only the features that are important for predicting the number of followers of a playlist, but also in what direction:
- The last and first update are very important: when these features are larger, the higher the amount of followers it will have on average. This can be related to a measure of "freshness" that can be encouranged also by Spotify discoverability algorithms.
- [COMPLETE!]

### 5. Adding new features [COMPLETE!!!]

We have added a new feature for these playlists, which reflect the decade of the songs. These features were taken from other data sources other than Spotify's API. So we are going to run the best model from the previous section (Gradient Boosted Regression Trees) with this new dataset to see if these new features can improve our predictions.

##### Loading Data



```python
data = pd.read_csv('data_spotify_v3.csv', index_col= 0)
print(data.shape)
print(data.columns.values)
print("\nPercentage of the playlists for the different decades")
np.mean(data.iloc[:,8:20], axis=0)

data.iloc[:,8:20].head(200)
```


    (1636, 55)
    ['acousticness_avg' 'acousticness_std' 'active_period' 'artist_genres'
     'artist_popularity_avg' 'artist_popularity_std' 'danceability_avg'
     'danceability_std' 'decade_1900' 'decade_1910' 'decade_1920' 'decade_1930'
     'decade_1940' 'decade_1950' 'decade_1960' 'decade_1970' 'decade_1980'
     'decade_1990' 'decade_2000' 'decade_2010' 'duration_ms_avg'
     'duration_ms_std' 'energy_avg' 'energy_std' 'first_update' 'followers'
     'instrumentalness_avg' 'instrumentalness_std' 'is_collaborative'
     'is_public' 'key_avg' 'key_std' 'last_update' 'liveness_avg'
     'liveness_std' 'loudness_avg' 'loudness_std' 'mode_avg' 'mode_std'
     'num_of_artists' 'num_of_markets_avg' 'num_of_songs'
     'playlist_name_length' 'song_duration_avg' 'song_duration_std'
     'song_popularity_avg' 'song_popularity_std' 'speechiness_avg'
     'speechiness_std' 'tempo_avg' 'tempo_std' 'time_signature_avg'
     'time_signature_std' 'valence_avg' 'valence_std']
    
    Percentage of the playlists for the different decades





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
      <th>decade_1900</th>
      <th>decade_1910</th>
      <th>decade_1920</th>
      <th>decade_1930</th>
      <th>decade_1940</th>
      <th>decade_1950</th>
      <th>decade_1960</th>
      <th>decade_1970</th>
      <th>decade_1980</th>
      <th>decade_1990</th>
      <th>decade_2000</th>
      <th>decade_2010</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.074074</td>
      <td>0.925926</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.960000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012195</td>
      <td>0.097561</td>
      <td>0.890244</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.047619</td>
      <td>0.666667</td>
      <td>0.238095</td>
      <td>0.047619</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.100000</td>
      <td>0.900000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.090909</td>
      <td>0.045455</td>
      <td>0.136364</td>
      <td>0.272727</td>
      <td>0.454545</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.122449</td>
      <td>0.877551</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.224138</td>
      <td>0.155172</td>
      <td>0.620690</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.50000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.060000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.428571</td>
      <td>0.228571</td>
      <td>0.085714</td>
      <td>0.114286</td>
      <td>0.142857</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.360000</td>
      <td>0.640000</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.538462</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.384615</td>
      <td>0.076923</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.428571</td>
      <td>0.333333</td>
      <td>0.238095</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.26000</td>
      <td>0.180000</td>
      <td>0.100000</td>
      <td>0.120000</td>
      <td>0.220000</td>
      <td>0.120000</td>
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
    </tr>
    <tr>
      <th>170</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>171</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>172</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>173</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.516129</td>
      <td>0.483871</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>174</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>175</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>176</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>177</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.800000</td>
    </tr>
    <tr>
      <th>178</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.066667</td>
      <td>0.050000</td>
      <td>0.883333</td>
    </tr>
    <tr>
      <th>179</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.020000</td>
      <td>0.120000</td>
      <td>0.820000</td>
    </tr>
    <tr>
      <th>180</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.133333</td>
      <td>0.866667</td>
    </tr>
    <tr>
      <th>181</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.280000</td>
      <td>0.720000</td>
    </tr>
    <tr>
      <th>182</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.476190</td>
      <td>0.523810</td>
    </tr>
    <tr>
      <th>183</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.066667</td>
      <td>0.933333</td>
    </tr>
    <tr>
      <th>184</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.057143</td>
      <td>0.028571</td>
      <td>0.914286</td>
    </tr>
    <tr>
      <th>185</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>186</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.47619</td>
      <td>0.047619</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.428571</td>
      <td>0.047619</td>
    </tr>
    <tr>
      <th>187</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>188</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>189</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>190</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.022222</td>
      <td>0.111111</td>
      <td>0.866667</td>
    </tr>
    <tr>
      <th>191</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>192</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>193</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.020000</td>
      <td>0.200000</td>
      <td>0.200000</td>
      <td>0.160000</td>
      <td>0.420000</td>
    </tr>
    <tr>
      <th>194</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.020408</td>
      <td>0.367347</td>
      <td>0.387755</td>
      <td>0.224490</td>
    </tr>
    <tr>
      <th>195</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.100000</td>
      <td>0.040000</td>
      <td>0.860000</td>
    </tr>
    <tr>
      <th>196</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.020833</td>
      <td>0.854167</td>
      <td>0.104167</td>
      <td>0.020833</td>
    </tr>
    <tr>
      <th>197</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.044444</td>
      <td>0.022222</td>
      <td>0.177778</td>
      <td>0.444444</td>
      <td>0.311111</td>
    </tr>
    <tr>
      <th>198</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.960000</td>
    </tr>
    <tr>
      <th>199</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.02000</td>
      <td>0.260000</td>
      <td>0.060000</td>
      <td>0.220000</td>
      <td>0.380000</td>
      <td>0.060000</td>
    </tr>
  </tbody>
</table>
<p>200 rows × 12 columns</p>
</div>



##### Data Transformation and creating of Train and Test sets



```python
# Dropping null values
data.dropna(how='any', inplace= True)
data.shape

# Dropping features that present zero variance
data.drop(['is_collaborative', 'is_public','decade_1910'], axis =1 , inplace= True)

# Getting features
data1 = data.copy()
X = data1.drop(['followers'], axis = 1)

# Scaling X
scaler = MinMaxScaler().fit(X)
data1 = scaler.transform(X)
data1 = pd.DataFrame(data1, columns= X.columns)
data1['followers'] = data['followers']


# Creating a dataset with ln(Y)
data3 = data1.copy()
data3['followers'] = np.log(1 + data3['followers'])


y = (data1['followers'])
X = data1.drop(['followers'], axis = 1)

y2 = (data3['followers'])
X2 = data3.drop(['followers'], axis = 1)
X2.shape, y2.shape

# Creating Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.33, random_state=42)

```


##### Fitting the model



```python
param_grid = {'learning_rate': [0.1, 0.01],
              'max_depth': [3, 6],
              'min_samples_leaf': [3, 5],
              'max_features': [0.2, 0.6]
              }

gb = GradientBoostingRegressor(n_estimators=600, loss='huber', random_state=1111)
gb_cv = GridSearchCV(gb, param_grid, cv=3, n_jobs=-1)

gb_cv.fit(X_train, y_train)
```





    GridSearchCV(cv=3, error_score='raise',
           estimator=GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                 learning_rate=0.1, loss='huber', max_depth=3,
                 max_features=None, max_leaf_nodes=None,
                 min_impurity_split=1e-07, min_samples_leaf=1,
                 min_samples_split=2, min_weight_fraction_leaf=0.0,
                 n_estimators=600, presort='auto', random_state=1111,
                 subsample=1.0, verbose=0, warm_start=False),
           fit_params={}, iid=True, n_jobs=-1,
           param_grid={'learning_rate': [0.1, 0.01], 'max_depth': [3, 6], 'min_samples_leaf': [3, 5], 'max_features': [0.2, 0.6]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
           scoring=None, verbose=0)





```python
gb_cv.best_estimator_
```





    GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                 learning_rate=0.01, loss='huber', max_depth=6,
                 max_features=0.2, max_leaf_nodes=None,
                 min_impurity_split=1e-07, min_samples_leaf=5,
                 min_samples_split=2, min_weight_fraction_leaf=0.0,
                 n_estimators=600, presort='auto', random_state=1111,
                 subsample=1.0, verbose=0, warm_start=False)





```python
from sklearn.metrics import r2_score

r2_score_test = r2_score(y_test, gb_cv.predict(X_test))

print("The train R2 for the Grandient Boosted Regression Tree is {:.2f}%".format(r2_score_test*100))
```


    The train R2 for the Grandient Boosted Regression Tree is 62.53%


From the results above we see that adding these new features does not succeed in improving our model.

### 6. Adding more features

##### Loading Data



```python
data = pd.read_csv('data_spotify_v3.csv', index_col= 0)
data2 = pd.read_csv('EveryNoise_playlists_data.csv')
print("Shape for first file {}".format(data.shape))
print("Shape for second file {}".format(data2.shape))
```


    Shape for first file (1636, 55)
    Shape for second file (1497, 56)




```python
# Concatenating these two datasets
frames = [data, data2]

all = pd.concat(frames)
print("Shape for concatenated dataset {}".format(all.shape))

all.head(100)
```


    Shape for concatenated dataset (3133, 56)





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
      <th>acousticness_avg</th>
      <th>acousticness_std</th>
      <th>active_period</th>
      <th>artist_genres</th>
      <th>artist_popularity_avg</th>
      <th>artist_popularity_std</th>
      <th>danceability_avg</th>
      <th>danceability_std</th>
      <th>decade_1900</th>
      <th>decade_1910</th>
      <th>...</th>
      <th>song_popularity_avg</th>
      <th>song_popularity_std</th>
      <th>speechiness_avg</th>
      <th>speechiness_std</th>
      <th>tempo_avg</th>
      <th>tempo_std</th>
      <th>time_signature_avg</th>
      <th>time_signature_std</th>
      <th>valence_avg</th>
      <th>valence_std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.201550</td>
      <td>0.210138</td>
      <td>0.0</td>
      <td>35</td>
      <td>89.757895</td>
      <td>5.358153</td>
      <td>0.672320</td>
      <td>0.131615</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>82.860000</td>
      <td>10.148911</td>
      <td>0.080678</td>
      <td>0.050808</td>
      <td>120.039820</td>
      <td>27.196016</td>
      <td>4.000000</td>
      <td>0.200000</td>
      <td>0.419074</td>
      <td>0.182246</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.162225</td>
      <td>0.196876</td>
      <td>0.0</td>
      <td>26</td>
      <td>86.815217</td>
      <td>9.026446</td>
      <td>0.748647</td>
      <td>0.139444</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>75.313725</td>
      <td>11.412910</td>
      <td>0.224157</td>
      <td>0.133022</td>
      <td>128.579686</td>
      <td>28.443222</td>
      <td>4.019608</td>
      <td>0.241742</td>
      <td>0.461216</td>
      <td>0.213021</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.110528</td>
      <td>0.125271</td>
      <td>257482.0</td>
      <td>55</td>
      <td>69.589286</td>
      <td>15.207231</td>
      <td>0.620873</td>
      <td>0.115776</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>60.807018</td>
      <td>17.420938</td>
      <td>0.074180</td>
      <td>0.070625</td>
      <td>127.025945</td>
      <td>19.637540</td>
      <td>3.963636</td>
      <td>0.187193</td>
      <td>0.391784</td>
      <td>0.200177</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.237346</td>
      <td>0.209753</td>
      <td>0.0</td>
      <td>25</td>
      <td>79.014706</td>
      <td>12.330608</td>
      <td>0.630980</td>
      <td>0.127485</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>63.938776</td>
      <td>12.573714</td>
      <td>0.127398</td>
      <td>0.102994</td>
      <td>115.759082</td>
      <td>29.865996</td>
      <td>3.918367</td>
      <td>0.528252</td>
      <td>0.434865</td>
      <td>0.191858</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.049196</td>
      <td>0.124658</td>
      <td>0.0</td>
      <td>70</td>
      <td>72.660714</td>
      <td>9.728964</td>
      <td>0.530340</td>
      <td>0.120331</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>59.981132</td>
      <td>8.962165</td>
      <td>0.063791</td>
      <td>0.049612</td>
      <td>126.416849</td>
      <td>25.883036</td>
      <td>3.905660</td>
      <td>0.445699</td>
      <td>0.508849</td>
      <td>0.204985</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.166265</td>
      <td>0.170598</td>
      <td>0.0</td>
      <td>19</td>
      <td>73.551724</td>
      <td>8.225638</td>
      <td>0.597327</td>
      <td>0.090946</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>65.423077</td>
      <td>15.686038</td>
      <td>0.042362</td>
      <td>0.017098</td>
      <td>118.156308</td>
      <td>31.329474</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.581635</td>
      <td>0.159798</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.156190</td>
      <td>0.119429</td>
      <td>0.0</td>
      <td>26</td>
      <td>88.059406</td>
      <td>7.321232</td>
      <td>0.729420</td>
      <td>0.064479</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>78.960000</td>
      <td>13.181745</td>
      <td>0.087244</td>
      <td>0.046341</td>
      <td>119.366280</td>
      <td>32.962056</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.662620</td>
      <td>0.176820</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.237830</td>
      <td>0.264956</td>
      <td>41875.0</td>
      <td>98</td>
      <td>70.741573</td>
      <td>15.071014</td>
      <td>0.603434</td>
      <td>0.161986</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>54.051724</td>
      <td>9.990379</td>
      <td>0.109192</td>
      <td>0.112646</td>
      <td>128.370547</td>
      <td>35.033491</td>
      <td>3.811321</td>
      <td>0.516030</td>
      <td>0.420051</td>
      <td>0.210253</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.786123</td>
      <td>0.199674</td>
      <td>0.0</td>
      <td>59</td>
      <td>61.638554</td>
      <td>9.529382</td>
      <td>0.525704</td>
      <td>0.101879</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>50.901235</td>
      <td>16.504848</td>
      <td>0.033222</td>
      <td>0.006567</td>
      <td>114.429407</td>
      <td>28.322789</td>
      <td>3.913580</td>
      <td>0.449898</td>
      <td>0.320828</td>
      <td>0.129866</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.987340</td>
      <td>0.013527</td>
      <td>0.0</td>
      <td>33</td>
      <td>63.866667</td>
      <td>8.138172</td>
      <td>0.375910</td>
      <td>0.119037</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>58.920000</td>
      <td>14.958396</td>
      <td>0.046537</td>
      <td>0.012607</td>
      <td>108.401150</td>
      <td>32.522634</td>
      <td>3.540000</td>
      <td>0.853464</td>
      <td>0.169196</td>
      <td>0.123953</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.560321</td>
      <td>0.301546</td>
      <td>0.0</td>
      <td>69</td>
      <td>61.819048</td>
      <td>9.734358</td>
      <td>0.516560</td>
      <td>0.117606</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>50.510000</td>
      <td>13.116779</td>
      <td>0.036958</td>
      <td>0.012719</td>
      <td>115.489590</td>
      <td>31.718457</td>
      <td>3.860000</td>
      <td>0.548088</td>
      <td>0.334927</td>
      <td>0.176004</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.372465</td>
      <td>0.337896</td>
      <td>0.0</td>
      <td>36</td>
      <td>33.344828</td>
      <td>16.062255</td>
      <td>0.502805</td>
      <td>0.111165</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>21.280488</td>
      <td>16.571555</td>
      <td>0.047393</td>
      <td>0.028185</td>
      <td>128.784439</td>
      <td>30.242586</td>
      <td>3.902439</td>
      <td>0.296720</td>
      <td>0.512171</td>
      <td>0.230420</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.220964</td>
      <td>0.234015</td>
      <td>1228282.0</td>
      <td>22</td>
      <td>62.078947</td>
      <td>17.390835</td>
      <td>0.741000</td>
      <td>0.127863</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>41.920000</td>
      <td>20.145312</td>
      <td>0.188500</td>
      <td>0.114445</td>
      <td>124.331680</td>
      <td>31.597291</td>
      <td>3.880000</td>
      <td>0.587878</td>
      <td>0.502200</td>
      <td>0.192124</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.344474</td>
      <td>0.329582</td>
      <td>0.0</td>
      <td>76</td>
      <td>48.357143</td>
      <td>14.886989</td>
      <td>0.552673</td>
      <td>0.168766</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>30.545455</td>
      <td>18.374704</td>
      <td>0.072133</td>
      <td>0.077668</td>
      <td>121.398164</td>
      <td>29.889103</td>
      <td>3.872727</td>
      <td>0.506487</td>
      <td>0.392365</td>
      <td>0.211918</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.253763</td>
      <td>0.302475</td>
      <td>3743.0</td>
      <td>73</td>
      <td>69.057143</td>
      <td>16.689505</td>
      <td>0.612220</td>
      <td>0.133341</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>55.466667</td>
      <td>9.227013</td>
      <td>0.087276</td>
      <td>0.079625</td>
      <td>119.415976</td>
      <td>27.369151</td>
      <td>3.902439</td>
      <td>0.369896</td>
      <td>0.445907</td>
      <td>0.242811</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.143884</td>
      <td>0.135943</td>
      <td>0.0</td>
      <td>29</td>
      <td>68.781250</td>
      <td>14.284464</td>
      <td>0.807476</td>
      <td>0.122141</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>48.142857</td>
      <td>13.445784</td>
      <td>0.235300</td>
      <td>0.115372</td>
      <td>103.087476</td>
      <td>20.696904</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.646048</td>
      <td>0.166843</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.218213</td>
      <td>0.271245</td>
      <td>0.0</td>
      <td>87</td>
      <td>57.287879</td>
      <td>15.370934</td>
      <td>0.573109</td>
      <td>0.187072</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>42.109091</td>
      <td>19.591903</td>
      <td>0.097662</td>
      <td>0.095417</td>
      <td>121.318527</td>
      <td>28.182618</td>
      <td>3.854545</td>
      <td>0.615502</td>
      <td>0.472185</td>
      <td>0.227651</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.181231</td>
      <td>0.149650</td>
      <td>0.0</td>
      <td>61</td>
      <td>79.262626</td>
      <td>13.197479</td>
      <td>0.698060</td>
      <td>0.089967</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>72.360000</td>
      <td>14.964972</td>
      <td>0.086456</td>
      <td>0.065620</td>
      <td>112.946660</td>
      <td>18.855509</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.463562</td>
      <td>0.183308</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.200536</td>
      <td>0.266862</td>
      <td>1569935.0</td>
      <td>45</td>
      <td>45.593220</td>
      <td>17.136658</td>
      <td>0.540706</td>
      <td>0.160960</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>28.960784</td>
      <td>16.399379</td>
      <td>0.098094</td>
      <td>0.090684</td>
      <td>129.127647</td>
      <td>27.876211</td>
      <td>3.980392</td>
      <td>0.138648</td>
      <td>0.429692</td>
      <td>0.228722</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.776473</td>
      <td>0.265833</td>
      <td>0.0</td>
      <td>69</td>
      <td>27.880952</td>
      <td>15.453775</td>
      <td>0.297418</td>
      <td>0.148548</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>18.425000</td>
      <td>13.529389</td>
      <td>0.049460</td>
      <td>0.029320</td>
      <td>108.700025</td>
      <td>37.311863</td>
      <td>3.875000</td>
      <td>0.713705</td>
      <td>0.132685</td>
      <td>0.133143</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.348700</td>
      <td>0.336479</td>
      <td>708716.0</td>
      <td>56</td>
      <td>61.764706</td>
      <td>16.356432</td>
      <td>0.606864</td>
      <td>0.154810</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>42.227273</td>
      <td>19.187903</td>
      <td>0.088873</td>
      <td>0.082685</td>
      <td>110.400409</td>
      <td>30.695100</td>
      <td>3.818182</td>
      <td>0.649221</td>
      <td>0.398350</td>
      <td>0.225276</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.361513</td>
      <td>0.313367</td>
      <td>0.0</td>
      <td>26</td>
      <td>61.122807</td>
      <td>5.331371</td>
      <td>0.490653</td>
      <td>0.096412</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>43.020408</td>
      <td>7.358155</td>
      <td>0.033718</td>
      <td>0.012102</td>
      <td>112.118735</td>
      <td>26.646985</td>
      <td>3.897959</td>
      <td>0.363930</td>
      <td>0.423435</td>
      <td>0.175933</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.607912</td>
      <td>0.273498</td>
      <td>22125662.0</td>
      <td>74</td>
      <td>60.855932</td>
      <td>14.753497</td>
      <td>0.471172</td>
      <td>0.163675</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>41.706897</td>
      <td>23.982759</td>
      <td>0.075200</td>
      <td>0.081810</td>
      <td>110.555552</td>
      <td>26.639681</td>
      <td>3.862069</td>
      <td>0.680980</td>
      <td>0.456105</td>
      <td>0.247278</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.350730</td>
      <td>0.246802</td>
      <td>0.0</td>
      <td>8</td>
      <td>75.636364</td>
      <td>7.474474</td>
      <td>0.556000</td>
      <td>0.151256</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>51.320000</td>
      <td>7.859873</td>
      <td>0.045362</td>
      <td>0.023430</td>
      <td>119.263040</td>
      <td>25.939101</td>
      <td>3.980000</td>
      <td>0.140000</td>
      <td>0.617310</td>
      <td>0.261826</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.043497</td>
      <td>0.087279</td>
      <td>0.0</td>
      <td>5</td>
      <td>86.000000</td>
      <td>0.000000</td>
      <td>0.508886</td>
      <td>0.098307</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>61.542857</td>
      <td>6.855000</td>
      <td>0.082583</td>
      <td>0.056962</td>
      <td>126.481629</td>
      <td>19.287092</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.486486</td>
      <td>0.156637</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.464827</td>
      <td>0.386639</td>
      <td>0.0</td>
      <td>4</td>
      <td>85.888889</td>
      <td>7.612530</td>
      <td>0.521480</td>
      <td>0.146393</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>63.800000</td>
      <td>6.864401</td>
      <td>0.039696</td>
      <td>0.016275</td>
      <td>120.981760</td>
      <td>30.289289</td>
      <td>3.760000</td>
      <td>0.649923</td>
      <td>0.307544</td>
      <td>0.144813</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.330031</td>
      <td>0.200522</td>
      <td>534711.0</td>
      <td>24</td>
      <td>70.967742</td>
      <td>3.897757</td>
      <td>0.663500</td>
      <td>0.131860</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>22.153846</td>
      <td>19.738078</td>
      <td>0.064419</td>
      <td>0.051717</td>
      <td>114.878615</td>
      <td>24.266813</td>
      <td>3.846154</td>
      <td>0.360801</td>
      <td>0.632692</td>
      <td>0.191927</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.524209</td>
      <td>0.307408</td>
      <td>0.0</td>
      <td>2</td>
      <td>80.000000</td>
      <td>0.000000</td>
      <td>0.482478</td>
      <td>0.125586</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>52.739130</td>
      <td>6.891819</td>
      <td>0.041061</td>
      <td>0.009764</td>
      <td>126.467348</td>
      <td>22.975332</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.288730</td>
      <td>0.218672</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.041955</td>
      <td>0.052746</td>
      <td>0.0</td>
      <td>5</td>
      <td>64.000000</td>
      <td>0.000000</td>
      <td>0.703190</td>
      <td>0.088061</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>31.952381</td>
      <td>10.218919</td>
      <td>0.048476</td>
      <td>0.023498</td>
      <td>129.197190</td>
      <td>14.407778</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.719190</td>
      <td>0.216465</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.375784</td>
      <td>0.271836</td>
      <td>6632.0</td>
      <td>28</td>
      <td>75.111111</td>
      <td>0.808901</td>
      <td>0.586380</td>
      <td>0.137303</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>42.200000</td>
      <td>8.611620</td>
      <td>0.060648</td>
      <td>0.046618</td>
      <td>118.037160</td>
      <td>24.547195</td>
      <td>3.800000</td>
      <td>0.400000</td>
      <td>0.616958</td>
      <td>0.248554</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>70</th>
      <td>0.170804</td>
      <td>0.204121</td>
      <td>9146785.0</td>
      <td>8</td>
      <td>81.962963</td>
      <td>5.534280</td>
      <td>0.724133</td>
      <td>0.085263</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>52.088889</td>
      <td>18.875996</td>
      <td>0.070369</td>
      <td>0.044385</td>
      <td>116.794044</td>
      <td>21.020270</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.683000</td>
      <td>0.215844</td>
    </tr>
    <tr>
      <th>71</th>
      <td>0.171916</td>
      <td>0.196605</td>
      <td>0.0</td>
      <td>10</td>
      <td>80.000000</td>
      <td>0.000000</td>
      <td>0.508037</td>
      <td>0.121582</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>43.574074</td>
      <td>16.406415</td>
      <td>0.046409</td>
      <td>0.028838</td>
      <td>121.435741</td>
      <td>19.762645</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.596613</td>
      <td>0.230023</td>
    </tr>
    <tr>
      <th>72</th>
      <td>0.294437</td>
      <td>0.272726</td>
      <td>2926.0</td>
      <td>25</td>
      <td>67.625000</td>
      <td>4.509365</td>
      <td>0.546361</td>
      <td>0.125170</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>36.305556</td>
      <td>11.034842</td>
      <td>0.050411</td>
      <td>0.035866</td>
      <td>129.262056</td>
      <td>29.710585</td>
      <td>3.916667</td>
      <td>0.276385</td>
      <td>0.517361</td>
      <td>0.209269</td>
    </tr>
    <tr>
      <th>73</th>
      <td>0.663727</td>
      <td>0.215775</td>
      <td>0.0</td>
      <td>17</td>
      <td>73.000000</td>
      <td>0.000000</td>
      <td>0.640033</td>
      <td>0.089896</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>30.066667</td>
      <td>10.508515</td>
      <td>0.073345</td>
      <td>0.043828</td>
      <td>128.814833</td>
      <td>33.108323</td>
      <td>3.883333</td>
      <td>0.450617</td>
      <td>0.827667</td>
      <td>0.151128</td>
    </tr>
    <tr>
      <th>74</th>
      <td>0.171539</td>
      <td>0.242918</td>
      <td>0.0</td>
      <td>7</td>
      <td>84.216216</td>
      <td>7.400274</td>
      <td>0.727500</td>
      <td>0.121891</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>56.633333</td>
      <td>11.686127</td>
      <td>0.098507</td>
      <td>0.078849</td>
      <td>121.886500</td>
      <td>19.056821</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.536173</td>
      <td>0.301165</td>
    </tr>
    <tr>
      <th>75</th>
      <td>0.323057</td>
      <td>0.257864</td>
      <td>0.0</td>
      <td>9</td>
      <td>79.022222</td>
      <td>6.485844</td>
      <td>0.545867</td>
      <td>0.135151</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>50.977778</td>
      <td>9.298719</td>
      <td>0.033627</td>
      <td>0.011723</td>
      <td>117.100733</td>
      <td>33.179938</td>
      <td>3.933333</td>
      <td>0.326599</td>
      <td>0.612331</td>
      <td>0.234149</td>
    </tr>
    <tr>
      <th>76</th>
      <td>0.329725</td>
      <td>0.240271</td>
      <td>20744.0</td>
      <td>9</td>
      <td>74.814815</td>
      <td>6.755174</td>
      <td>0.615292</td>
      <td>0.139306</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>46.708333</td>
      <td>16.736883</td>
      <td>0.044571</td>
      <td>0.009804</td>
      <td>115.432917</td>
      <td>23.182625</td>
      <td>3.958333</td>
      <td>0.199826</td>
      <td>0.683750</td>
      <td>0.238982</td>
    </tr>
    <tr>
      <th>77</th>
      <td>0.264612</td>
      <td>0.261657</td>
      <td>0.0</td>
      <td>15</td>
      <td>80.666667</td>
      <td>5.064034</td>
      <td>0.537756</td>
      <td>0.129263</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>40.390244</td>
      <td>17.418630</td>
      <td>0.039959</td>
      <td>0.024551</td>
      <td>129.026390</td>
      <td>26.487054</td>
      <td>3.951220</td>
      <td>0.215409</td>
      <td>0.553024</td>
      <td>0.267704</td>
    </tr>
    <tr>
      <th>78</th>
      <td>0.336361</td>
      <td>0.343847</td>
      <td>27933688.0</td>
      <td>19</td>
      <td>73.210526</td>
      <td>9.052938</td>
      <td>0.566577</td>
      <td>0.129767</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>41.653846</td>
      <td>12.306791</td>
      <td>0.086819</td>
      <td>0.079577</td>
      <td>109.456885</td>
      <td>18.480213</td>
      <td>4.000000</td>
      <td>0.679366</td>
      <td>0.338631</td>
      <td>0.211459</td>
    </tr>
    <tr>
      <th>79</th>
      <td>0.129660</td>
      <td>0.220695</td>
      <td>0.0</td>
      <td>0</td>
      <td>73.826667</td>
      <td>12.897931</td>
      <td>0.651800</td>
      <td>0.173518</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>58.950000</td>
      <td>7.999844</td>
      <td>0.107725</td>
      <td>0.096842</td>
      <td>123.973950</td>
      <td>26.957362</td>
      <td>3.950000</td>
      <td>0.217945</td>
      <td>0.589902</td>
      <td>0.234257</td>
    </tr>
    <tr>
      <th>80</th>
      <td>0.131307</td>
      <td>0.171251</td>
      <td>835.0</td>
      <td>21</td>
      <td>95.355556</td>
      <td>5.869528</td>
      <td>0.743837</td>
      <td>0.117768</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>70.632653</td>
      <td>4.406368</td>
      <td>0.215851</td>
      <td>0.133073</td>
      <td>125.712857</td>
      <td>27.472401</td>
      <td>4.000000</td>
      <td>0.202031</td>
      <td>0.386631</td>
      <td>0.165846</td>
    </tr>
    <tr>
      <th>81</th>
      <td>0.425065</td>
      <td>0.273396</td>
      <td>0.0</td>
      <td>15</td>
      <td>59.277778</td>
      <td>11.854702</td>
      <td>0.548759</td>
      <td>0.110318</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>23.907407</td>
      <td>20.042795</td>
      <td>0.040674</td>
      <td>0.011530</td>
      <td>117.569130</td>
      <td>29.465935</td>
      <td>3.740741</td>
      <td>0.724829</td>
      <td>0.629130</td>
      <td>0.207903</td>
    </tr>
    <tr>
      <th>82</th>
      <td>0.770706</td>
      <td>0.221449</td>
      <td>0.0</td>
      <td>31</td>
      <td>67.901961</td>
      <td>0.693242</td>
      <td>0.533020</td>
      <td>0.133961</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>38.080000</td>
      <td>10.409304</td>
      <td>0.036702</td>
      <td>0.015057</td>
      <td>110.695940</td>
      <td>26.479487</td>
      <td>3.740000</td>
      <td>0.558928</td>
      <td>0.438650</td>
      <td>0.244187</td>
    </tr>
    <tr>
      <th>83</th>
      <td>0.200652</td>
      <td>0.245417</td>
      <td>0.0</td>
      <td>6</td>
      <td>86.702128</td>
      <td>2.009709</td>
      <td>0.615732</td>
      <td>0.150058</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>56.951220</td>
      <td>16.963305</td>
      <td>0.098237</td>
      <td>0.086599</td>
      <td>125.299098</td>
      <td>33.987598</td>
      <td>4.000000</td>
      <td>0.220863</td>
      <td>0.570463</td>
      <td>0.209605</td>
    </tr>
    <tr>
      <th>84</th>
      <td>0.588340</td>
      <td>0.284677</td>
      <td>0.0</td>
      <td>5</td>
      <td>73.500000</td>
      <td>7.500000</td>
      <td>0.553960</td>
      <td>0.116514</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>32.760000</td>
      <td>21.128710</td>
      <td>0.040768</td>
      <td>0.011791</td>
      <td>117.495240</td>
      <td>24.477120</td>
      <td>4.040000</td>
      <td>0.195959</td>
      <td>0.191344</td>
      <td>0.103988</td>
    </tr>
    <tr>
      <th>85</th>
      <td>0.383068</td>
      <td>0.262870</td>
      <td>0.0</td>
      <td>27</td>
      <td>68.615385</td>
      <td>5.728316</td>
      <td>0.506960</td>
      <td>0.152132</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>32.440000</td>
      <td>12.083311</td>
      <td>0.049386</td>
      <td>0.030989</td>
      <td>115.770920</td>
      <td>31.008568</td>
      <td>3.960000</td>
      <td>0.280000</td>
      <td>0.496200</td>
      <td>0.271303</td>
    </tr>
    <tr>
      <th>86</th>
      <td>0.058763</td>
      <td>0.077298</td>
      <td>0.0</td>
      <td>47</td>
      <td>81.480769</td>
      <td>14.424192</td>
      <td>0.624114</td>
      <td>0.118631</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>55.500000</td>
      <td>19.363156</td>
      <td>0.069455</td>
      <td>0.041568</td>
      <td>128.429841</td>
      <td>26.611255</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.560023</td>
      <td>0.182972</td>
    </tr>
    <tr>
      <th>87</th>
      <td>0.313152</td>
      <td>0.240917</td>
      <td>0.0</td>
      <td>11</td>
      <td>65.708333</td>
      <td>3.397047</td>
      <td>0.636348</td>
      <td>0.131574</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>44.347826</td>
      <td>6.913180</td>
      <td>0.078035</td>
      <td>0.051791</td>
      <td>110.965435</td>
      <td>23.195828</td>
      <td>3.956522</td>
      <td>0.203931</td>
      <td>0.529304</td>
      <td>0.206123</td>
    </tr>
    <tr>
      <th>88</th>
      <td>0.485162</td>
      <td>0.365019</td>
      <td>0.0</td>
      <td>5</td>
      <td>32.615385</td>
      <td>25.784726</td>
      <td>0.594692</td>
      <td>0.198696</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>22.730769</td>
      <td>16.372016</td>
      <td>0.286208</td>
      <td>0.376609</td>
      <td>119.083077</td>
      <td>50.672600</td>
      <td>3.076923</td>
      <td>1.298611</td>
      <td>0.559981</td>
      <td>0.270660</td>
    </tr>
    <tr>
      <th>89</th>
      <td>0.339376</td>
      <td>0.336902</td>
      <td>0.0</td>
      <td>21</td>
      <td>65.132075</td>
      <td>9.484919</td>
      <td>0.416373</td>
      <td>0.135674</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>14.098039</td>
      <td>16.763938</td>
      <td>0.049425</td>
      <td>0.026301</td>
      <td>125.188373</td>
      <td>27.461389</td>
      <td>3.901961</td>
      <td>0.408484</td>
      <td>0.391833</td>
      <td>0.211810</td>
    </tr>
    <tr>
      <th>90</th>
      <td>0.007387</td>
      <td>0.016235</td>
      <td>25738267.0</td>
      <td>4</td>
      <td>81.000000</td>
      <td>0.000000</td>
      <td>0.547057</td>
      <td>0.063853</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>57.371429</td>
      <td>7.305910</td>
      <td>0.049300</td>
      <td>0.019699</td>
      <td>130.260971</td>
      <td>27.365907</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.570829</td>
      <td>0.214851</td>
    </tr>
    <tr>
      <th>91</th>
      <td>0.362674</td>
      <td>0.352012</td>
      <td>0.0</td>
      <td>6</td>
      <td>82.000000</td>
      <td>0.000000</td>
      <td>0.392894</td>
      <td>0.157086</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>55.980000</td>
      <td>7.347081</td>
      <td>0.046036</td>
      <td>0.036181</td>
      <td>121.525740</td>
      <td>31.164356</td>
      <td>3.900000</td>
      <td>0.412311</td>
      <td>0.308902</td>
      <td>0.232509</td>
    </tr>
    <tr>
      <th>92</th>
      <td>0.206874</td>
      <td>0.254887</td>
      <td>0.0</td>
      <td>14</td>
      <td>77.000000</td>
      <td>0.000000</td>
      <td>0.462452</td>
      <td>0.145691</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>37.770492</td>
      <td>8.017181</td>
      <td>0.039857</td>
      <td>0.019745</td>
      <td>128.846852</td>
      <td>32.800561</td>
      <td>3.885246</td>
      <td>0.447153</td>
      <td>0.566180</td>
      <td>0.232916</td>
    </tr>
    <tr>
      <th>93</th>
      <td>0.176970</td>
      <td>0.250899</td>
      <td>30594374.0</td>
      <td>19</td>
      <td>92.860000</td>
      <td>5.385202</td>
      <td>0.621694</td>
      <td>0.137182</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>64.361111</td>
      <td>11.282121</td>
      <td>0.077903</td>
      <td>0.055491</td>
      <td>119.365111</td>
      <td>31.462769</td>
      <td>3.944444</td>
      <td>0.229061</td>
      <td>0.503286</td>
      <td>0.201224</td>
    </tr>
    <tr>
      <th>94</th>
      <td>0.127340</td>
      <td>0.152750</td>
      <td>0.0</td>
      <td>7</td>
      <td>72.000000</td>
      <td>0.000000</td>
      <td>0.559060</td>
      <td>0.097893</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>38.640000</td>
      <td>12.819922</td>
      <td>0.033608</td>
      <td>0.018354</td>
      <td>122.708660</td>
      <td>30.468187</td>
      <td>3.960000</td>
      <td>0.195959</td>
      <td>0.517354</td>
      <td>0.239521</td>
    </tr>
    <tr>
      <th>95</th>
      <td>0.498856</td>
      <td>0.348871</td>
      <td>10819213.0</td>
      <td>24</td>
      <td>69.520000</td>
      <td>10.284435</td>
      <td>0.602257</td>
      <td>0.168387</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>47.285714</td>
      <td>12.641364</td>
      <td>0.079317</td>
      <td>0.084052</td>
      <td>123.014314</td>
      <td>24.364596</td>
      <td>4.028571</td>
      <td>0.166599</td>
      <td>0.309343</td>
      <td>0.163024</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0.131547</td>
      <td>0.162981</td>
      <td>5272318.0</td>
      <td>20</td>
      <td>65.616279</td>
      <td>14.669355</td>
      <td>0.724478</td>
      <td>0.107335</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>41.717391</td>
      <td>13.885656</td>
      <td>0.218870</td>
      <td>0.107657</td>
      <td>135.767283</td>
      <td>15.883254</td>
      <td>4.000000</td>
      <td>0.208514</td>
      <td>0.503391</td>
      <td>0.226026</td>
    </tr>
    <tr>
      <th>97</th>
      <td>0.118612</td>
      <td>0.134351</td>
      <td>120866.0</td>
      <td>9</td>
      <td>60.863636</td>
      <td>15.826604</td>
      <td>0.646698</td>
      <td>0.109951</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>43.465116</td>
      <td>10.897725</td>
      <td>0.050535</td>
      <td>0.037262</td>
      <td>119.930721</td>
      <td>27.581378</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.647798</td>
      <td>0.231619</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0.341188</td>
      <td>0.309220</td>
      <td>0.0</td>
      <td>37</td>
      <td>74.089286</td>
      <td>2.572358</td>
      <td>0.564080</td>
      <td>0.154997</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>38.280000</td>
      <td>16.128286</td>
      <td>0.057232</td>
      <td>0.060146</td>
      <td>111.431700</td>
      <td>27.232007</td>
      <td>3.920000</td>
      <td>0.440000</td>
      <td>0.462840</td>
      <td>0.228857</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0.144365</td>
      <td>0.196054</td>
      <td>0.0</td>
      <td>9</td>
      <td>67.571429</td>
      <td>2.744196</td>
      <td>0.556450</td>
      <td>0.117271</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>36.050000</td>
      <td>13.965941</td>
      <td>0.047760</td>
      <td>0.032568</td>
      <td>119.853600</td>
      <td>25.414710</td>
      <td>3.975000</td>
      <td>0.272718</td>
      <td>0.521375</td>
      <td>0.244545</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 56 columns</p>
</div>



##### Data Transformation and creating of Train and Test sets



```python
# Dropping name_score
all.drop(['name_score'], axis =1 , inplace= True)

# Dropping null values
all.dropna(how='any', inplace= True)
print("Shape of data after dropna {}".format(all.shape))

all.reset_index(drop=True, inplace=True)


# Dropping features that present zero variance
np.sum(all['decade_1910'])
all.drop(['is_collaborative', 'is_public','decade_1910'], axis =1 , inplace= True)

# Getting features
all1 = all.copy()
X = all1.drop(['followers'], axis = 1)

# Scaling X
scaler = MinMaxScaler().fit(X)
all1 = scaler.transform(X)
all1 = pd.DataFrame(all1, columns= X.columns)
all1['followers'] = all['followers']


# Creating a dataset with ln(Y)
all3 = all1.copy()
all3['followers'] = np.log(1 + all3['followers'])


y = (all1['followers'])
X = all1.drop(['followers'], axis = 1)

y2 = (all3['followers'])
X2 = all3.drop(['followers'], axis = 1)
X2.shape, y2.shape

# Creating Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.33, random_state=42)

print("Shape of train set {}".format(X_train.shape))
print("Shape of test set {}".format(X_test.shape))
```


    Shape of data after dropna (3133, 55)
    Shape of train set (2099, 51)
    Shape of test set (1034, 51)


##### Fitting the model



```python
param_grid = {'learning_rate': [0.1, 0.01],
              'max_depth': [3, 6],
              'min_samples_leaf': [3, 5],
              'max_features': [0.2, 0.6]
              }

gb = GradientBoostingRegressor(n_estimators=600, loss='huber', random_state=1111)
gb_cv = GridSearchCV(gb, param_grid, cv=3, n_jobs=-1)

gb_cv.fit(X_train, y_train)
```





    GridSearchCV(cv=3, error_score='raise',
           estimator=GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                 learning_rate=0.1, loss='huber', max_depth=3,
                 max_features=None, max_leaf_nodes=None,
                 min_impurity_split=1e-07, min_samples_leaf=1,
                 min_samples_split=2, min_weight_fraction_leaf=0.0,
                 n_estimators=600, presort='auto', random_state=1111,
                 subsample=1.0, verbose=0, warm_start=False),
           fit_params={}, iid=True, n_jobs=-1,
           param_grid={'learning_rate': [0.1, 0.01], 'max_depth': [3, 6], 'min_samples_leaf': [3, 5], 'max_features': [0.2, 0.6]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
           scoring=None, verbose=0)





```python
gb_cv.best_estimator_
```





    GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                 learning_rate=0.01, loss='huber', max_depth=6,
                 max_features=0.2, max_leaf_nodes=None,
                 min_impurity_split=1e-07, min_samples_leaf=3,
                 min_samples_split=2, min_weight_fraction_leaf=0.0,
                 n_estimators=600, presort='auto', random_state=1111,
                 subsample=1.0, verbose=0, warm_start=False)





```python
r2_score_test = r2_score(y_test, gb_cv.predict(X_test))

print("The train R2 for the Grandient Boosted Regression Tree is {:.2f}%".format(r2_score_test*100))
```


    The train R2 for the Grandient Boosted Regression Tree is 75.56%


##### Feature Importance



```python
plt.figure(figsize=(10,10))
pd.Series(gb_cv.best_estimator_.feature_importances_, index=X_train.columns.values).sort_values()[27:].plot(kind="barh")
plt.title("Feature Importance")
plt.show()
```



![png](Assessing%20alternative%20models%20-%20Andres_files/Assessing%20alternative%20models%20-%20Andres_49_0.png)


The most important features are:
- first_update: [EXPLAIN!!!]
- last_update
- song_popularity
- active_period
- artist_popularity_avg
- num_of_songs
- artist_popularity_std
- artist_genres

It's interesting to notice that some of these features are characteristics of the playlist (for example: first and last updates, active period, number of songs, standard deviation of artist popularity) while others are characteristis of the songs that make up the playlist (song popularity, artist popularity). This distiction will be especially important when we build our model for generating playlists for different genres.


### 7. Conclusions

[Explanation of results]
