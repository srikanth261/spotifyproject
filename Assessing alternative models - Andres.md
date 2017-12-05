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


    /Users/andreslindner/anaconda/envs/py36/lib/python3.6/site-packages/statsmodels/compat/pandas.py:56: FutureWarning: The pandas.core.datetools module is deprecated and will be removed in a future version. Please use the pandas.tseries module instead.
      from pandas.core import datetools


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





    1.0





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


    The test R2 for the Random Forest is 63.84%


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





    0.61532503813160289





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
                 min_impurity_split=1e-07, min_samples_leaf=5,
                 min_samples_split=2, min_weight_fraction_leaf=0.0,
                 n_estimators=600, presort='auto', random_state=None,
                 subsample=1.0, verbose=0, warm_start=False)





```python
from sklearn.metrics import r2_score

r2_score_test = r2_score(y_test, gb_cv.predict(X_test))

print("The train R2 for the Grandient Boosted Regression Tree is {:.2f}%".format(r2_score_test*100))
```


    The train R2 for the Grandient Boosted Regression Tree is 64.18%


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





    Index(['song_popularity_std', 'danceability_std', 'num_of_artists',
           'artist_popularity_avg', 'artist_genres', 'num_of_markets_avg',
           'tempo_std', 'artist_popularity_std', 'num_of_songs',
           'song_popularity_avg', 'first_update', 'last_update'],
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

### 5. Conclusions

[Explanation of results]
