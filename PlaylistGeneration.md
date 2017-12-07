---
title: Generation Algorithm
notebook: PlaylistGeneration.ipynb
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
path = '/Users/andreslindner/Google Drive/HES/Introduction to Data Science/Project - Spotify/'
songs = pd.read_csv(path+'Data Files/EveryNoise_tracks.1.4.csv')
print(songs.shape)
songs.head(10)
```


    (385722, 23)





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
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>...</th>
      <th>uri</th>
      <th>track_href</th>
      <th>analysis_url</th>
      <th>duration_ms</th>
      <th>time_signature</th>
      <th>genres</th>
      <th>playlist_name</th>
      <th>popularity</th>
      <th>artist_popularity</th>
      <th>explicit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.628</td>
      <td>0.797</td>
      <td>11</td>
      <td>-3.953</td>
      <td>0</td>
      <td>0.0596</td>
      <td>0.03640</td>
      <td>0.000000</td>
      <td>0.1040</td>
      <td>0.321</td>
      <td>...</td>
      <td>spotify:track:7EI6Iki24tBHAMxtb4xQN2</td>
      <td>https://api.spotify.com/v1/tracks/7EI6Iki24tBH...</td>
      <td>https://api.spotify.com/v1/audio-analysis/7EI6...</td>
      <td>215064</td>
      <td>4</td>
      <td>dance pop|pop|pop rap|post-teen pop|tropical h...</td>
      <td>Pop</td>
      <td>88</td>
      <td>89.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.782</td>
      <td>0.600</td>
      <td>7</td>
      <td>-6.532</td>
      <td>1</td>
      <td>0.0416</td>
      <td>0.01620</td>
      <td>0.000002</td>
      <td>0.1080</td>
      <td>0.612</td>
      <td>...</td>
      <td>spotify:track:20I6sIOMTCkB6w7ryavxtO</td>
      <td>https://api.spotify.com/v1/tracks/20I6sIOMTCkB...</td>
      <td>https://api.spotify.com/v1/audio-analysis/20I6...</td>
      <td>193400</td>
      <td>4</td>
      <td>canadian pop|dance pop|indie poptimism|pop|pop...</td>
      <td>Pop</td>
      <td>72</td>
      <td>77.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.632</td>
      <td>0.730</td>
      <td>6</td>
      <td>-4.075</td>
      <td>0</td>
      <td>0.0338</td>
      <td>0.00113</td>
      <td>0.000000</td>
      <td>0.0826</td>
      <td>0.311</td>
      <td>...</td>
      <td>spotify:track:6Ll3qUTssdGWMxbwhMud2N</td>
      <td>https://api.spotify.com/v1/tracks/6Ll3qUTssdGW...</td>
      <td>https://api.spotify.com/v1/audio-analysis/6Ll3...</td>
      <td>197893</td>
      <td>4</td>
      <td>dance pop|pop|pop christmas|post-teen pop|trop...</td>
      <td>Pop</td>
      <td>77</td>
      <td>81.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.565</td>
      <td>0.381</td>
      <td>10</td>
      <td>-7.204</td>
      <td>1</td>
      <td>0.0302</td>
      <td>0.48000</td>
      <td>0.000000</td>
      <td>0.1090</td>
      <td>0.325</td>
      <td>...</td>
      <td>spotify:track:79NlESqzFSW0hdBWgls4FX</td>
      <td>https://api.spotify.com/v1/tracks/79NlESqzFSW0...</td>
      <td>https://api.spotify.com/v1/audio-analysis/79Nl...</td>
      <td>230269</td>
      <td>4</td>
      <td>dance pop|pop|pop rap|post-teen pop|r&amp;b</td>
      <td>Pop</td>
      <td>81</td>
      <td>86.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.711</td>
      <td>0.774</td>
      <td>0</td>
      <td>-3.391</td>
      <td>0</td>
      <td>0.0922</td>
      <td>0.25900</td>
      <td>0.000000</td>
      <td>0.3710</td>
      <td>0.715</td>
      <td>...</td>
      <td>spotify:track:3SaIsrEzrQGDcG1jCeaK8q</td>
      <td>https://api.spotify.com/v1/tracks/3SaIsrEzrQGD...</td>
      <td>https://api.spotify.com/v1/audio-analysis/3SaI...</td>
      <td>199387</td>
      <td>4</td>
      <td>dance pop|hip pop|pop|pop christmas|pop rap|po...</td>
      <td>Pop</td>
      <td>69</td>
      <td>88.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.585</td>
      <td>0.303</td>
      <td>4</td>
      <td>-10.058</td>
      <td>1</td>
      <td>0.0398</td>
      <td>0.69400</td>
      <td>0.000000</td>
      <td>0.1150</td>
      <td>0.142</td>
      <td>...</td>
      <td>spotify:track:05pKAafT85jeeNhZ6kq7HT</td>
      <td>https://api.spotify.com/v1/tracks/05pKAafT85je...</td>
      <td>https://api.spotify.com/v1/audio-analysis/05pK...</td>
      <td>240166</td>
      <td>3</td>
      <td>acoustic pop|folk-pop|neo mellow|pop|pop chris...</td>
      <td>Pop</td>
      <td>75</td>
      <td>81.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.694</td>
      <td>0.891</td>
      <td>9</td>
      <td>-2.940</td>
      <td>1</td>
      <td>0.0949</td>
      <td>0.05620</td>
      <td>0.000000</td>
      <td>0.5610</td>
      <td>0.563</td>
      <td>...</td>
      <td>spotify:track:6E11E0lT5Zy7yb6iT3y8DN</td>
      <td>https://api.spotify.com/v1/tracks/6E11E0lT5Zy7...</td>
      <td>https://api.spotify.com/v1/audio-analysis/6E11...</td>
      <td>214573</td>
      <td>4</td>
      <td>dance pop|pop|post-teen pop|talent show|teen p...</td>
      <td>Pop</td>
      <td>63</td>
      <td>67.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.630</td>
      <td>0.733</td>
      <td>1</td>
      <td>-4.338</td>
      <td>1</td>
      <td>0.0423</td>
      <td>0.05510</td>
      <td>0.000000</td>
      <td>0.1020</td>
      <td>0.486</td>
      <td>...</td>
      <td>spotify:track:5Gz1PxujU62sfecQpB0Stm</td>
      <td>https://api.spotify.com/v1/tracks/5Gz1PxujU62s...</td>
      <td>https://api.spotify.com/v1/audio-analysis/5Gz1...</td>
      <td>199586</td>
      <td>4</td>
      <td>dance pop|europop|neo mellow|pop|post-teen pop...</td>
      <td>Pop</td>
      <td>72</td>
      <td>72.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.811</td>
      <td>0.596</td>
      <td>11</td>
      <td>-5.428</td>
      <td>1</td>
      <td>0.0621</td>
      <td>0.00755</td>
      <td>0.000012</td>
      <td>0.0588</td>
      <td>0.673</td>
      <td>...</td>
      <td>spotify:track:0o7glLPfZgns8ZF14Wlpf7</td>
      <td>https://api.spotify.com/v1/tracks/0o7glLPfZgns...</td>
      <td>https://api.spotify.com/v1/audio-analysis/0o7g...</td>
      <td>172599</td>
      <td>4</td>
      <td>dance pop|neo mellow|pop|pop christmas|pop roc...</td>
      <td>Pop</td>
      <td>74</td>
      <td>83.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.659</td>
      <td>0.717</td>
      <td>8</td>
      <td>-4.202</td>
      <td>1</td>
      <td>0.0581</td>
      <td>0.03930</td>
      <td>0.000000</td>
      <td>0.0769</td>
      <td>0.710</td>
      <td>...</td>
      <td>spotify:track:5AEtlRudpgdT5FtNiuly6Y</td>
      <td>https://api.spotify.com/v1/tracks/5AEtlRudpgdT...</td>
      <td>https://api.spotify.com/v1/audio-analysis/5AEt...</td>
      <td>174800</td>
      <td>4</td>
      <td>big room|contemporary country|country road|dan...</td>
      <td>Pop</td>
      <td>90</td>
      <td>87.5</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 23 columns</p>
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
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>duration_ms</th>
      <th>time_signature</th>
      <th>popularity</th>
      <th>artist_popularity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>385722.000000</td>
      <td>385722.000000</td>
      <td>385722.000000</td>
      <td>385722.000000</td>
      <td>385722.000000</td>
      <td>385722.000000</td>
      <td>385722.000000</td>
      <td>385722.000000</td>
      <td>385722.000000</td>
      <td>385722.000000</td>
      <td>385722.000000</td>
      <td>3.857220e+05</td>
      <td>385722.000000</td>
      <td>385722.000000</td>
      <td>385722.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.532061</td>
      <td>0.633183</td>
      <td>5.314281</td>
      <td>-9.105949</td>
      <td>0.626869</td>
      <td>0.079648</td>
      <td>0.323429</td>
      <td>0.248311</td>
      <td>0.199941</td>
      <td>0.485182</td>
      <td>122.099861</td>
      <td>2.627601e+05</td>
      <td>3.895308</td>
      <td>15.986750</td>
      <td>22.076962</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.179091</td>
      <td>0.262507</td>
      <td>3.550942</td>
      <td>5.164573</td>
      <td>0.483637</td>
      <td>0.094026</td>
      <td>0.351417</td>
      <td>0.357795</td>
      <td>0.167794</td>
      <td>0.273700</td>
      <td>29.235690</td>
      <td>1.052649e+05</td>
      <td>0.425754</td>
      <td>14.847646</td>
      <td>17.469293</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.054900</td>
      <td>0.000155</td>
      <td>0.000000</td>
      <td>-47.904000</td>
      <td>0.000000</td>
      <td>0.021700</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.009120</td>
      <td>0.000010</td>
      <td>34.010000</td>
      <td>3.029700e+04</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.414000</td>
      <td>0.446000</td>
      <td>2.000000</td>
      <td>-11.195000</td>
      <td>0.000000</td>
      <td>0.035800</td>
      <td>0.007280</td>
      <td>0.000002</td>
      <td>0.095300</td>
      <td>0.255000</td>
      <td>99.824000</td>
      <td>1.940930e+05</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.547000</td>
      <td>0.676000</td>
      <td>5.000000</td>
      <td>-7.862000</td>
      <td>1.000000</td>
      <td>0.048000</td>
      <td>0.157000</td>
      <td>0.002770</td>
      <td>0.130000</td>
      <td>0.482000</td>
      <td>121.499000</td>
      <td>2.389330e+05</td>
      <td>4.000000</td>
      <td>12.000000</td>
      <td>21.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.666000</td>
      <td>0.861000</td>
      <td>9.000000</td>
      <td>-5.632000</td>
      <td>1.000000</td>
      <td>0.081300</td>
      <td>0.641000</td>
      <td>0.586000</td>
      <td>0.263000</td>
      <td>0.713000</td>
      <td>139.984000</td>
      <td>3.057870e+05</td>
      <td>4.000000</td>
      <td>27.000000</td>
      <td>36.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.975000</td>
      <td>1.000000</td>
      <td>11.000000</td>
      <td>5.959000</td>
      <td>1.000000</td>
      <td>0.968000</td>
      <td>0.996000</td>
      <td>0.998000</td>
      <td>0.995000</td>
      <td>0.999000</td>
      <td>247.985000</td>
      <td>4.766720e+06</td>
      <td>5.000000</td>
      <td>94.000000</td>
      <td>93.000000</td>
    </tr>
  </tbody>
</table>
</div>





```python
# Removing duplicate rows and rows with null values
print("Original shape: {}".format(songs.shape))
songs.drop_duplicates(subset='id', inplace=True)
print("Shape of dataset after removing duplicates: {}".format(songs.shape))
songs.dropna(how='any', inplace=True)
print("Shape of dataset after dropna: {}".format(songs.shape))
```


    Original shape: (385722, 23)
    Shape of dataset after removing duplicates: (97913, 23)
    Shape of dataset after dropna: (97708, 23)




```python
# Getting genres (taking the first genre of the list)
genre = []

for s in songs['genres']:
    g = s[:s.find("|")]
    genre.append(g)
#     print(s)
    
songs['sub_genre'] = genre
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
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>...</th>
      <th>track_href</th>
      <th>analysis_url</th>
      <th>duration_ms</th>
      <th>time_signature</th>
      <th>genres</th>
      <th>playlist_name</th>
      <th>popularity</th>
      <th>artist_popularity</th>
      <th>explicit</th>
      <th>sub_genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.628</td>
      <td>0.797</td>
      <td>11</td>
      <td>-3.953</td>
      <td>0</td>
      <td>0.0596</td>
      <td>0.03640</td>
      <td>0.000000</td>
      <td>0.1040</td>
      <td>0.321</td>
      <td>...</td>
      <td>https://api.spotify.com/v1/tracks/7EI6Iki24tBH...</td>
      <td>https://api.spotify.com/v1/audio-analysis/7EI6...</td>
      <td>215064</td>
      <td>4</td>
      <td>dance pop|pop|pop rap|post-teen pop|tropical h...</td>
      <td>Pop</td>
      <td>88</td>
      <td>89.0</td>
      <td>False</td>
      <td>dance pop</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.782</td>
      <td>0.600</td>
      <td>7</td>
      <td>-6.532</td>
      <td>1</td>
      <td>0.0416</td>
      <td>0.01620</td>
      <td>0.000002</td>
      <td>0.1080</td>
      <td>0.612</td>
      <td>...</td>
      <td>https://api.spotify.com/v1/tracks/20I6sIOMTCkB...</td>
      <td>https://api.spotify.com/v1/audio-analysis/20I6...</td>
      <td>193400</td>
      <td>4</td>
      <td>canadian pop|dance pop|indie poptimism|pop|pop...</td>
      <td>Pop</td>
      <td>72</td>
      <td>77.0</td>
      <td>False</td>
      <td>canadian pop</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.632</td>
      <td>0.730</td>
      <td>6</td>
      <td>-4.075</td>
      <td>0</td>
      <td>0.0338</td>
      <td>0.00113</td>
      <td>0.000000</td>
      <td>0.0826</td>
      <td>0.311</td>
      <td>...</td>
      <td>https://api.spotify.com/v1/tracks/6Ll3qUTssdGW...</td>
      <td>https://api.spotify.com/v1/audio-analysis/6Ll3...</td>
      <td>197893</td>
      <td>4</td>
      <td>dance pop|pop|pop christmas|post-teen pop|trop...</td>
      <td>Pop</td>
      <td>77</td>
      <td>81.0</td>
      <td>False</td>
      <td>dance pop</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.565</td>
      <td>0.381</td>
      <td>10</td>
      <td>-7.204</td>
      <td>1</td>
      <td>0.0302</td>
      <td>0.48000</td>
      <td>0.000000</td>
      <td>0.1090</td>
      <td>0.325</td>
      <td>...</td>
      <td>https://api.spotify.com/v1/tracks/79NlESqzFSW0...</td>
      <td>https://api.spotify.com/v1/audio-analysis/79Nl...</td>
      <td>230269</td>
      <td>4</td>
      <td>dance pop|pop|pop rap|post-teen pop|r&amp;b</td>
      <td>Pop</td>
      <td>81</td>
      <td>86.0</td>
      <td>False</td>
      <td>dance pop</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.711</td>
      <td>0.774</td>
      <td>0</td>
      <td>-3.391</td>
      <td>0</td>
      <td>0.0922</td>
      <td>0.25900</td>
      <td>0.000000</td>
      <td>0.3710</td>
      <td>0.715</td>
      <td>...</td>
      <td>https://api.spotify.com/v1/tracks/3SaIsrEzrQGD...</td>
      <td>https://api.spotify.com/v1/audio-analysis/3SaI...</td>
      <td>199387</td>
      <td>4</td>
      <td>dance pop|hip pop|pop|pop christmas|pop rap|po...</td>
      <td>Pop</td>
      <td>69</td>
      <td>88.0</td>
      <td>False</td>
      <td>dance pop</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.585</td>
      <td>0.303</td>
      <td>4</td>
      <td>-10.058</td>
      <td>1</td>
      <td>0.0398</td>
      <td>0.69400</td>
      <td>0.000000</td>
      <td>0.1150</td>
      <td>0.142</td>
      <td>...</td>
      <td>https://api.spotify.com/v1/tracks/05pKAafT85je...</td>
      <td>https://api.spotify.com/v1/audio-analysis/05pK...</td>
      <td>240166</td>
      <td>3</td>
      <td>acoustic pop|folk-pop|neo mellow|pop|pop chris...</td>
      <td>Pop</td>
      <td>75</td>
      <td>81.0</td>
      <td>False</td>
      <td>acoustic pop</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.694</td>
      <td>0.891</td>
      <td>9</td>
      <td>-2.940</td>
      <td>1</td>
      <td>0.0949</td>
      <td>0.05620</td>
      <td>0.000000</td>
      <td>0.5610</td>
      <td>0.563</td>
      <td>...</td>
      <td>https://api.spotify.com/v1/tracks/6E11E0lT5Zy7...</td>
      <td>https://api.spotify.com/v1/audio-analysis/6E11...</td>
      <td>214573</td>
      <td>4</td>
      <td>dance pop|pop|post-teen pop|talent show|teen p...</td>
      <td>Pop</td>
      <td>63</td>
      <td>67.0</td>
      <td>False</td>
      <td>dance pop</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.630</td>
      <td>0.733</td>
      <td>1</td>
      <td>-4.338</td>
      <td>1</td>
      <td>0.0423</td>
      <td>0.05510</td>
      <td>0.000000</td>
      <td>0.1020</td>
      <td>0.486</td>
      <td>...</td>
      <td>https://api.spotify.com/v1/tracks/5Gz1PxujU62s...</td>
      <td>https://api.spotify.com/v1/audio-analysis/5Gz1...</td>
      <td>199586</td>
      <td>4</td>
      <td>dance pop|europop|neo mellow|pop|post-teen pop...</td>
      <td>Pop</td>
      <td>72</td>
      <td>72.0</td>
      <td>False</td>
      <td>dance pop</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.811</td>
      <td>0.596</td>
      <td>11</td>
      <td>-5.428</td>
      <td>1</td>
      <td>0.0621</td>
      <td>0.00755</td>
      <td>0.000012</td>
      <td>0.0588</td>
      <td>0.673</td>
      <td>...</td>
      <td>https://api.spotify.com/v1/tracks/0o7glLPfZgns...</td>
      <td>https://api.spotify.com/v1/audio-analysis/0o7g...</td>
      <td>172599</td>
      <td>4</td>
      <td>dance pop|neo mellow|pop|pop christmas|pop roc...</td>
      <td>Pop</td>
      <td>74</td>
      <td>83.0</td>
      <td>False</td>
      <td>dance pop</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.659</td>
      <td>0.717</td>
      <td>8</td>
      <td>-4.202</td>
      <td>1</td>
      <td>0.0581</td>
      <td>0.03930</td>
      <td>0.000000</td>
      <td>0.0769</td>
      <td>0.710</td>
      <td>...</td>
      <td>https://api.spotify.com/v1/tracks/5AEtlRudpgdT...</td>
      <td>https://api.spotify.com/v1/audio-analysis/5AEt...</td>
      <td>174800</td>
      <td>4</td>
      <td>big room|contemporary country|country road|dan...</td>
      <td>Pop</td>
      <td>90</td>
      <td>87.5</td>
      <td>False</td>
      <td>big room</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 24 columns</p>
</div>



The feature called **playlist_name** contains the genre, together with the feature **sub_genre** that we created, we can get a fairly good understanding of the genres and sub-genres that the different tracks belong to.



```python
songs = songs.reset_index(drop=True)
songs.shape
```





    (97708, 24)



### 3. Building Generative algorithm for Playlists

As we learnt from our predictive model on playlist, we know that some of the song related features that are correlated with popular playlists are:
- Song popularity
- Artist popularity

In addition to these two features, we are going to use also some accoustic features to define similarity, since this could also help us build new playlist accoustically similar to existing successful playlists (there are some studies showing that measures based on accoustic similarity can be very similar to people's perception on similarity between songs). Particularly, we are going to use speechiness (since it came out to be of the most important accoustic factors in our predictive model).


#### Algorithm to find similar songs

We need to take a couple of steps:
1. Scale all the features
2. Select some genre
3. Select a song for that particular genre, which has a high song popularity and artist popularity 
4. Get the closest 100 songs as defined by the euclidean distance of the features mentioned above.




#### 1. Scaling the data



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
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>duration_ms</th>
      <th>time_signature</th>
      <th>popularity</th>
      <th>artist_popularity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>97708.000000</td>
      <td>97708.000000</td>
      <td>97708.000000</td>
      <td>97708.000000</td>
      <td>97708.000000</td>
      <td>97708.000000</td>
      <td>97708.000000</td>
      <td>97708.000000</td>
      <td>97708.000000</td>
      <td>97708.000000</td>
      <td>97708.000000</td>
      <td>9.770800e+04</td>
      <td>97708.000000</td>
      <td>97708.000000</td>
      <td>97708.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.537381</td>
      <td>0.629727</td>
      <td>5.309258</td>
      <td>-9.118232</td>
      <td>0.629887</td>
      <td>0.081546</td>
      <td>0.325928</td>
      <td>0.240615</td>
      <td>0.198867</td>
      <td>0.490976</td>
      <td>122.081159</td>
      <td>2.608718e+05</td>
      <td>3.896099</td>
      <td>13.687671</td>
      <td>19.017381</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.175444</td>
      <td>0.259715</td>
      <td>3.554209</td>
      <td>5.093413</td>
      <td>0.482837</td>
      <td>0.100844</td>
      <td>0.351134</td>
      <td>0.353896</td>
      <td>0.166733</td>
      <td>0.270341</td>
      <td>29.040300</td>
      <td>1.095310e+05</td>
      <td>0.424432</td>
      <td>14.254682</td>
      <td>17.083217</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.054900</td>
      <td>0.000155</td>
      <td>0.000000</td>
      <td>-47.904000</td>
      <td>0.000000</td>
      <td>0.021700</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.009120</td>
      <td>0.000010</td>
      <td>34.010000</td>
      <td>3.029700e+04</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.422000</td>
      <td>0.445000</td>
      <td>2.000000</td>
      <td>-11.206000</td>
      <td>0.000000</td>
      <td>0.035700</td>
      <td>0.008380</td>
      <td>0.000002</td>
      <td>0.095300</td>
      <td>0.267000</td>
      <td>99.940000</td>
      <td>1.930105e+05</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.551000</td>
      <td>0.671000</td>
      <td>5.000000</td>
      <td>-7.906000</td>
      <td>1.000000</td>
      <td>0.047900</td>
      <td>0.163000</td>
      <td>0.002220</td>
      <td>0.129000</td>
      <td>0.490000</td>
      <td>121.733500</td>
      <td>2.366000e+05</td>
      <td>4.000000</td>
      <td>9.000000</td>
      <td>15.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.668000</td>
      <td>0.854000</td>
      <td>9.000000</td>
      <td>-5.674000</td>
      <td>1.000000</td>
      <td>0.081800</td>
      <td>0.645000</td>
      <td>0.545000</td>
      <td>0.261000</td>
      <td>0.715000</td>
      <td>139.972250</td>
      <td>3.020930e+05</td>
      <td>4.000000</td>
      <td>23.000000</td>
      <td>32.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.975000</td>
      <td>1.000000</td>
      <td>11.000000</td>
      <td>5.959000</td>
      <td>1.000000</td>
      <td>0.968000</td>
      <td>0.996000</td>
      <td>0.998000</td>
      <td>0.995000</td>
      <td>0.999000</td>
      <td>247.985000</td>
      <td>4.766720e+06</td>
      <td>5.000000</td>
      <td>94.000000</td>
      <td>93.000000</td>
    </tr>
  </tbody>
</table>
</div>





```python
# Getting features
features = songs[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
       'duration_ms','time_signature', 'popularity','artist_popularity', 'explicit']]


# Scaling featues
scaler = MinMaxScaler().fit(features)
data = scaler.transform(features)
data = pd.DataFrame(data, columns= features.columns)
data['genre'] = songs['playlist_name']
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
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>duration_ms</th>
      <th>time_signature</th>
      <th>popularity</th>
      <th>artist_popularity</th>
      <th>explicit</th>
      <th>genre</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.622867</td>
      <td>0.796969</td>
      <td>1.000000</td>
      <td>0.815978</td>
      <td>0.0</td>
      <td>0.040051</td>
      <td>0.036546</td>
      <td>0.000000</td>
      <td>0.096239</td>
      <td>0.321315</td>
      <td>0.340787</td>
      <td>0.039010</td>
      <td>0.8</td>
      <td>0.936170</td>
      <td>0.956989</td>
      <td>0.0</td>
      <td>Pop</td>
      <td>7EI6Iki24tBHAMxtb4xQN2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.790240</td>
      <td>0.599938</td>
      <td>0.636364</td>
      <td>0.768097</td>
      <td>1.0</td>
      <td>0.021029</td>
      <td>0.016265</td>
      <td>0.000002</td>
      <td>0.100296</td>
      <td>0.612609</td>
      <td>0.402103</td>
      <td>0.034436</td>
      <td>0.8</td>
      <td>0.765957</td>
      <td>0.827957</td>
      <td>0.0</td>
      <td>Pop</td>
      <td>20I6sIOMTCkB6w7ryavxtO</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.627214</td>
      <td>0.729958</td>
      <td>0.545455</td>
      <td>0.813713</td>
      <td>0.0</td>
      <td>0.012787</td>
      <td>0.001135</td>
      <td>0.000000</td>
      <td>0.074532</td>
      <td>0.311304</td>
      <td>0.429877</td>
      <td>0.035385</td>
      <td>0.8</td>
      <td>0.819149</td>
      <td>0.870968</td>
      <td>0.0</td>
      <td>Pop</td>
      <td>6Ll3qUTssdGWMxbwhMud2N</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.554396</td>
      <td>0.380904</td>
      <td>0.909091</td>
      <td>0.755621</td>
      <td>1.0</td>
      <td>0.008982</td>
      <td>0.481928</td>
      <td>0.000000</td>
      <td>0.101311</td>
      <td>0.325319</td>
      <td>0.184475</td>
      <td>0.042220</td>
      <td>0.8</td>
      <td>0.861702</td>
      <td>0.924731</td>
      <td>0.0</td>
      <td>Pop</td>
      <td>79NlESqzFSW0hdBWgls4FX</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.713075</td>
      <td>0.773965</td>
      <td>0.000000</td>
      <td>0.826411</td>
      <td>0.0</td>
      <td>0.074501</td>
      <td>0.260040</td>
      <td>0.000000</td>
      <td>0.367063</td>
      <td>0.715713</td>
      <td>0.542222</td>
      <td>0.035700</td>
      <td>0.8</td>
      <td>0.734043</td>
      <td>0.946237</td>
      <td>0.0</td>
      <td>Pop</td>
      <td>3SaIsrEzrQGDcG1jCeaK8q</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.576133</td>
      <td>0.302892</td>
      <td>0.363636</td>
      <td>0.702634</td>
      <td>1.0</td>
      <td>0.019127</td>
      <td>0.696787</td>
      <td>0.000000</td>
      <td>0.107396</td>
      <td>0.142134</td>
      <td>0.479930</td>
      <td>0.044310</td>
      <td>0.6</td>
      <td>0.797872</td>
      <td>0.870968</td>
      <td>0.0</td>
      <td>Pop</td>
      <td>05pKAafT85jeeNhZ6kq7HT</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.694598</td>
      <td>0.890983</td>
      <td>0.818182</td>
      <td>0.834785</td>
      <td>1.0</td>
      <td>0.077354</td>
      <td>0.056426</td>
      <td>0.000000</td>
      <td>0.559784</td>
      <td>0.563559</td>
      <td>0.298769</td>
      <td>0.038906</td>
      <td>0.8</td>
      <td>0.670213</td>
      <td>0.720430</td>
      <td>0.0</td>
      <td>Pop</td>
      <td>6E11E0lT5Zy7yb6iT3y8DN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.625041</td>
      <td>0.732959</td>
      <td>0.090909</td>
      <td>0.808830</td>
      <td>1.0</td>
      <td>0.021769</td>
      <td>0.055321</td>
      <td>0.000000</td>
      <td>0.094210</td>
      <td>0.486481</td>
      <td>0.327127</td>
      <td>0.035742</td>
      <td>0.8</td>
      <td>0.765957</td>
      <td>0.774194</td>
      <td>0.0</td>
      <td>Pop</td>
      <td>5Gz1PxujU62sfecQpB0Stm</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.821759</td>
      <td>0.595937</td>
      <td>1.000000</td>
      <td>0.788593</td>
      <td>1.0</td>
      <td>0.042693</td>
      <td>0.007580</td>
      <td>0.000012</td>
      <td>0.050392</td>
      <td>0.673670</td>
      <td>0.448744</td>
      <td>0.030044</td>
      <td>0.8</td>
      <td>0.787234</td>
      <td>0.892473</td>
      <td>0.0</td>
      <td>Pop</td>
      <td>0o7glLPfZgns8ZF14Wlpf7</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.656559</td>
      <td>0.716956</td>
      <td>0.727273</td>
      <td>0.811355</td>
      <td>1.0</td>
      <td>0.038466</td>
      <td>0.039458</td>
      <td>0.000000</td>
      <td>0.068751</td>
      <td>0.710708</td>
      <td>0.322589</td>
      <td>0.030509</td>
      <td>0.8</td>
      <td>0.957447</td>
      <td>0.940860</td>
      <td>0.0</td>
      <td>Pop</td>
      <td>5AEtlRudpgdT5FtNiuly6Y</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.915227</td>
      <td>0.633943</td>
      <td>0.363636</td>
      <td>0.773072</td>
      <td>0.0</td>
      <td>0.147205</td>
      <td>0.004147</td>
      <td>0.000000</td>
      <td>0.060332</td>
      <td>0.506502</td>
      <td>0.364706</td>
      <td>0.038366</td>
      <td>0.8</td>
      <td>0.904255</td>
      <td>0.956989</td>
      <td>0.0</td>
      <td>Pop</td>
      <td>6eUncuJutsFi9BGO1JaBHh</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.446799</td>
      <td>0.758963</td>
      <td>0.727273</td>
      <td>0.796948</td>
      <td>0.0</td>
      <td>0.187361</td>
      <td>0.052309</td>
      <td>0.000000</td>
      <td>0.050290</td>
      <td>0.328322</td>
      <td>0.588961</td>
      <td>0.049420</td>
      <td>0.8</td>
      <td>0.734043</td>
      <td>0.892473</td>
      <td>0.0</td>
      <td>Pop</td>
      <td>519tXYJVcrpEMqV2BMbh6E</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.551136</td>
      <td>0.856978</td>
      <td>0.636364</td>
      <td>0.816479</td>
      <td>0.0</td>
      <td>0.061714</td>
      <td>0.038454</td>
      <td>0.000000</td>
      <td>0.102325</td>
      <td>0.539535</td>
      <td>0.434037</td>
      <td>0.035394</td>
      <td>0.8</td>
      <td>0.744681</td>
      <td>0.774194</td>
      <td>0.0</td>
      <td>Pop</td>
      <td>6nHqns54LRqDNjeqKDF3v8</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.707640</td>
      <td>0.800969</td>
      <td>0.454545</td>
      <td>0.771791</td>
      <td>1.0</td>
      <td>0.019233</td>
      <td>0.005793</td>
      <td>0.000000</td>
      <td>0.074127</td>
      <td>0.630627</td>
      <td>0.308461</td>
      <td>0.048356</td>
      <td>0.8</td>
      <td>0.787234</td>
      <td>0.774194</td>
      <td>0.0</td>
      <td>Pop</td>
      <td>1D1nixOVWOxvNfWi0UD7VX</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.717422</td>
      <td>0.801969</td>
      <td>1.000000</td>
      <td>0.803093</td>
      <td>0.0</td>
      <td>0.018704</td>
      <td>0.139558</td>
      <td>0.000000</td>
      <td>0.170284</td>
      <td>0.327321</td>
      <td>0.425185</td>
      <td>0.035406</td>
      <td>0.8</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>Pop</td>
      <td>7EmGUiUaOSGDnUUQUDrOXC</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.625041</td>
      <td>0.529927</td>
      <td>0.000000</td>
      <td>0.754600</td>
      <td>1.0</td>
      <td>0.022931</td>
      <td>0.401606</td>
      <td>0.000000</td>
      <td>0.170284</td>
      <td>0.417412</td>
      <td>0.345966</td>
      <td>0.041119</td>
      <td>0.6</td>
      <td>0.840426</td>
      <td>0.897849</td>
      <td>0.0</td>
      <td>Pop</td>
      <td>2YlZnw2ikdb837oKMKjBkW</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.284860</td>
      <td>0.569933</td>
      <td>0.000000</td>
      <td>0.790394</td>
      <td>1.0</td>
      <td>0.011307</td>
      <td>0.242972</td>
      <td>0.000000</td>
      <td>0.137826</td>
      <td>0.322316</td>
      <td>0.234808</td>
      <td>0.038236</td>
      <td>0.8</td>
      <td>0.659574</td>
      <td>0.784946</td>
      <td>0.0</td>
      <td>Pop</td>
      <td>1Yw6ViCo3tuufI0Hg4mzSU</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.863058</td>
      <td>0.498922</td>
      <td>0.181818</td>
      <td>0.743219</td>
      <td>1.0</td>
      <td>0.263447</td>
      <td>0.117470</td>
      <td>0.000050</td>
      <td>0.262588</td>
      <td>0.504500</td>
      <td>0.275602</td>
      <td>0.069604</td>
      <td>0.8</td>
      <td>0.606383</td>
      <td>0.774194</td>
      <td>1.0</td>
      <td>Pop</td>
      <td>0aULRU35N9kTj6O1xMULRR</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.640256</td>
      <td>0.231881</td>
      <td>0.181818</td>
      <td>0.619628</td>
      <td>1.0</td>
      <td>0.024094</td>
      <td>0.396586</td>
      <td>0.000031</td>
      <td>0.376192</td>
      <td>0.063754</td>
      <td>0.503274</td>
      <td>0.048244</td>
      <td>0.8</td>
      <td>0.053191</td>
      <td>0.107527</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>0z7Y93Wajc4LYQ0gT5VkRJ</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.252255</td>
      <td>0.594937</td>
      <td>0.545455</td>
      <td>0.780387</td>
      <td>1.0</td>
      <td>0.012681</td>
      <td>0.191767</td>
      <td>0.005170</td>
      <td>0.110439</td>
      <td>0.459454</td>
      <td>0.429938</td>
      <td>0.033006</td>
      <td>1.0</td>
      <td>0.319149</td>
      <td>0.301075</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>4NyhYkg3Joy19fgllXw06I</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.700033</td>
      <td>0.763963</td>
      <td>0.818182</td>
      <td>0.780629</td>
      <td>1.0</td>
      <td>0.004227</td>
      <td>0.335341</td>
      <td>0.000122</td>
      <td>0.096239</td>
      <td>0.783782</td>
      <td>0.364127</td>
      <td>0.033974</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>08TcCPQrnDpqwBuFS4p4GF</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.640256</td>
      <td>0.808970</td>
      <td>0.636364</td>
      <td>0.747043</td>
      <td>1.0</td>
      <td>0.009088</td>
      <td>0.425703</td>
      <td>0.000032</td>
      <td>0.132754</td>
      <td>0.625622</td>
      <td>0.364416</td>
      <td>0.041589</td>
      <td>0.8</td>
      <td>0.021277</td>
      <td>0.053763</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>7gNr0qWQkKa9k5o3eRzIBd</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.647864</td>
      <td>0.647945</td>
      <td>0.636364</td>
      <td>0.700611</td>
      <td>1.0</td>
      <td>0.007820</td>
      <td>0.388554</td>
      <td>0.006543</td>
      <td>0.326490</td>
      <td>0.679676</td>
      <td>0.299063</td>
      <td>0.072352</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>13Bc0coBNXQvU5piWZ5Mnj</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.285947</td>
      <td>0.472918</td>
      <td>0.181818</td>
      <td>0.744760</td>
      <td>1.0</td>
      <td>0.007186</td>
      <td>0.566265</td>
      <td>0.000008</td>
      <td>0.112468</td>
      <td>0.171163</td>
      <td>0.361715</td>
      <td>0.032327</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>5q15P5oyKyGeDztezzsU4W</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.487012</td>
      <td>0.815971</td>
      <td>0.363636</td>
      <td>0.779459</td>
      <td>1.0</td>
      <td>0.016485</td>
      <td>0.015763</td>
      <td>0.000041</td>
      <td>0.226072</td>
      <td>0.788787</td>
      <td>0.438621</td>
      <td>0.043410</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>0MDU04AXM27es0bE1RT2l1</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.370721</td>
      <td>0.587936</td>
      <td>0.818182</td>
      <td>0.746152</td>
      <td>1.0</td>
      <td>0.006023</td>
      <td>0.003886</td>
      <td>0.000000</td>
      <td>0.096239</td>
      <td>0.374368</td>
      <td>0.409623</td>
      <td>0.037955</td>
      <td>0.8</td>
      <td>0.010638</td>
      <td>0.032258</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>6mA1t4Nu5aGA5IgnlxnXpm</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.627214</td>
      <td>0.822973</td>
      <td>0.000000</td>
      <td>0.758331</td>
      <td>1.0</td>
      <td>0.006446</td>
      <td>0.001074</td>
      <td>0.851703</td>
      <td>0.077778</td>
      <td>0.609606</td>
      <td>0.401481</td>
      <td>0.050459</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>5Mw0EfaaMC7S8Y0afXjRJL</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.483752</td>
      <td>0.797969</td>
      <td>0.181818</td>
      <td>0.807920</td>
      <td>1.0</td>
      <td>0.012258</td>
      <td>0.000751</td>
      <td>0.010922</td>
      <td>0.108411</td>
      <td>0.534530</td>
      <td>0.441748</td>
      <td>0.035300</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>3qwA9pfwuYzAsSUdaiLQp7</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.342463</td>
      <td>0.522926</td>
      <td>0.181818</td>
      <td>0.719418</td>
      <td>1.0</td>
      <td>0.006552</td>
      <td>0.280120</td>
      <td>0.496994</td>
      <td>0.063172</td>
      <td>0.649646</td>
      <td>0.439888</td>
      <td>0.061489</td>
      <td>0.8</td>
      <td>0.393617</td>
      <td>0.387097</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>34IhFakqoA6aKnCkv97buS</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.695685</td>
      <td>0.637944</td>
      <td>0.272727</td>
      <td>0.716466</td>
      <td>1.0</td>
      <td>0.010673</td>
      <td>0.789157</td>
      <td>0.120240</td>
      <td>0.062462</td>
      <td>0.550546</td>
      <td>0.524000</td>
      <td>0.039275</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>7409wUufqO6hrn8EQw7f62</td>
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
    </tr>
    <tr>
      <th>70</th>
      <td>0.362026</td>
      <td>0.647945</td>
      <td>0.363636</td>
      <td>0.729907</td>
      <td>1.0</td>
      <td>0.034450</td>
      <td>0.002681</td>
      <td>0.000005</td>
      <td>0.082546</td>
      <td>0.270263</td>
      <td>0.167239</td>
      <td>0.040761</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.021505</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>4DzN7qW3imCimdL2rZVbb6</td>
    </tr>
    <tr>
      <th>71</th>
      <td>0.209868</td>
      <td>0.800969</td>
      <td>0.363636</td>
      <td>0.771179</td>
      <td>0.0</td>
      <td>0.017331</td>
      <td>0.001225</td>
      <td>0.023948</td>
      <td>0.117540</td>
      <td>0.278271</td>
      <td>0.370282</td>
      <td>0.059482</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>2REUCwPU2AA2WxNtmZIlS1</td>
    </tr>
    <tr>
      <th>72</th>
      <td>0.654385</td>
      <td>0.373903</td>
      <td>0.545455</td>
      <td>0.742068</td>
      <td>1.0</td>
      <td>0.011307</td>
      <td>0.858434</td>
      <td>0.000000</td>
      <td>0.301132</td>
      <td>0.608605</td>
      <td>0.403561</td>
      <td>0.026030</td>
      <td>0.8</td>
      <td>0.308511</td>
      <td>0.634409</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>32ua6sylmpqljnQ9BxzQim</td>
    </tr>
    <tr>
      <th>73</th>
      <td>0.242474</td>
      <td>0.568933</td>
      <td>0.000000</td>
      <td>0.713644</td>
      <td>1.0</td>
      <td>0.008877</td>
      <td>0.045683</td>
      <td>0.004609</td>
      <td>0.768735</td>
      <td>0.390384</td>
      <td>0.672518</td>
      <td>0.037707</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>74YxbycicrGZGuNTwxG3hM</td>
    </tr>
    <tr>
      <th>74</th>
      <td>0.619607</td>
      <td>0.489921</td>
      <td>0.636364</td>
      <td>0.679576</td>
      <td>1.0</td>
      <td>0.012575</td>
      <td>0.116466</td>
      <td>0.000084</td>
      <td>0.098268</td>
      <td>0.599596</td>
      <td>0.331698</td>
      <td>0.051934</td>
      <td>0.8</td>
      <td>0.053191</td>
      <td>0.021505</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>6ihGMpxNtDTjCjoBF8pOW9</td>
    </tr>
    <tr>
      <th>75</th>
      <td>0.258776</td>
      <td>0.743960</td>
      <td>0.545455</td>
      <td>0.781186</td>
      <td>1.0</td>
      <td>0.036458</td>
      <td>0.004880</td>
      <td>0.235471</td>
      <td>0.278817</td>
      <td>0.568564</td>
      <td>0.778913</td>
      <td>0.023139</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>3deLs3c6QlXvo2wDK2BlJm</td>
    </tr>
    <tr>
      <th>76</th>
      <td>0.506575</td>
      <td>0.370902</td>
      <td>0.181818</td>
      <td>0.738615</td>
      <td>1.0</td>
      <td>0.002431</td>
      <td>0.520080</td>
      <td>0.000003</td>
      <td>0.099282</td>
      <td>0.301294</td>
      <td>0.321472</td>
      <td>0.056430</td>
      <td>0.6</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>7i3sOjAuqGOx37AHtUnF1J</td>
    </tr>
    <tr>
      <th>77</th>
      <td>0.516357</td>
      <td>0.840975</td>
      <td>0.363636</td>
      <td>0.777751</td>
      <td>1.0</td>
      <td>0.019338</td>
      <td>0.055823</td>
      <td>0.000023</td>
      <td>0.091472</td>
      <td>0.471466</td>
      <td>0.433280</td>
      <td>0.037989</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>12570XYqlHZhSeeqM16Ykk</td>
    </tr>
    <tr>
      <th>78</th>
      <td>0.401152</td>
      <td>0.789967</td>
      <td>0.636364</td>
      <td>0.743256</td>
      <td>1.0</td>
      <td>0.011836</td>
      <td>0.017068</td>
      <td>0.917836</td>
      <td>0.341705</td>
      <td>0.746744</td>
      <td>0.564253</td>
      <td>0.031002</td>
      <td>0.8</td>
      <td>0.010638</td>
      <td>0.064516</td>
      <td>1.0</td>
      <td>Athens Indie</td>
      <td>7yTwOwXrzruSjrePPrD1iX</td>
    </tr>
    <tr>
      <th>79</th>
      <td>0.501141</td>
      <td>0.865979</td>
      <td>0.090909</td>
      <td>0.781371</td>
      <td>0.0</td>
      <td>0.008771</td>
      <td>0.183735</td>
      <td>0.081363</td>
      <td>0.605429</td>
      <td>0.670667</td>
      <td>0.324893</td>
      <td>0.046127</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>0UwZQseG7mersNz0P711DD</td>
    </tr>
    <tr>
      <th>80</th>
      <td>0.516357</td>
      <td>0.618941</td>
      <td>0.000000</td>
      <td>0.687503</td>
      <td>1.0</td>
      <td>0.040685</td>
      <td>0.099197</td>
      <td>0.755511</td>
      <td>0.107396</td>
      <td>0.574570</td>
      <td>0.401327</td>
      <td>0.056734</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>1CMdGBh8g6QM4NrVyIoUX0</td>
    </tr>
    <tr>
      <th>81</th>
      <td>0.665254</td>
      <td>0.626942</td>
      <td>0.818182</td>
      <td>0.755621</td>
      <td>0.0</td>
      <td>0.005072</td>
      <td>0.138554</td>
      <td>0.582164</td>
      <td>0.179413</td>
      <td>0.459454</td>
      <td>0.431410</td>
      <td>0.059397</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>3RLXwMZ9VvLaZ3LMfHGy8N</td>
    </tr>
    <tr>
      <th>82</th>
      <td>0.360939</td>
      <td>0.963994</td>
      <td>0.636364</td>
      <td>0.779570</td>
      <td>1.0</td>
      <td>0.026419</td>
      <td>0.041767</td>
      <td>0.185371</td>
      <td>0.318375</td>
      <td>0.888888</td>
      <td>0.408566</td>
      <td>0.033045</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>30syBNGvNPA4kN0vRTVEIz</td>
    </tr>
    <tr>
      <th>83</th>
      <td>0.466362</td>
      <td>0.610940</td>
      <td>0.909091</td>
      <td>0.743423</td>
      <td>1.0</td>
      <td>0.004861</td>
      <td>0.091466</td>
      <td>0.105210</td>
      <td>0.333590</td>
      <td>0.630627</td>
      <td>0.495478</td>
      <td>0.041848</td>
      <td>0.8</td>
      <td>0.159574</td>
      <td>0.129032</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>3E2vzyoYuXNdCfB169nSx0</td>
    </tr>
    <tr>
      <th>84</th>
      <td>0.691338</td>
      <td>0.824973</td>
      <td>0.000000</td>
      <td>0.768728</td>
      <td>1.0</td>
      <td>0.013526</td>
      <td>0.251004</td>
      <td>0.122244</td>
      <td>0.048667</td>
      <td>0.938938</td>
      <td>0.488398</td>
      <td>0.028307</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>0VjbMCQBiJhJvAUOTKqqV6</td>
    </tr>
    <tr>
      <th>85</th>
      <td>0.522878</td>
      <td>0.430912</td>
      <td>0.909091</td>
      <td>0.769359</td>
      <td>1.0</td>
      <td>0.007820</td>
      <td>0.402610</td>
      <td>0.000001</td>
      <td>0.208829</td>
      <td>0.681678</td>
      <td>0.184657</td>
      <td>0.042470</td>
      <td>0.8</td>
      <td>0.010638</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>14I8vttEbaQmYsWne0pYD3</td>
    </tr>
    <tr>
      <th>86</th>
      <td>0.565265</td>
      <td>0.653946</td>
      <td>0.818182</td>
      <td>0.679409</td>
      <td>1.0</td>
      <td>0.024834</td>
      <td>0.002269</td>
      <td>0.000012</td>
      <td>0.097253</td>
      <td>0.554550</td>
      <td>0.583596</td>
      <td>0.050431</td>
      <td>0.8</td>
      <td>0.148936</td>
      <td>0.204301</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>6HjU8Eo9DXWqDtgjlpI5c2</td>
    </tr>
    <tr>
      <th>87</th>
      <td>0.413107</td>
      <td>0.573934</td>
      <td>0.636364</td>
      <td>0.682788</td>
      <td>1.0</td>
      <td>0.005389</td>
      <td>0.485944</td>
      <td>0.528056</td>
      <td>0.149998</td>
      <td>0.243236</td>
      <td>0.481084</td>
      <td>0.041702</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>0akHwpm72qdOIPZcH5OKhU</td>
    </tr>
    <tr>
      <th>88</th>
      <td>0.582654</td>
      <td>0.799969</td>
      <td>0.181818</td>
      <td>0.749605</td>
      <td>1.0</td>
      <td>0.020078</td>
      <td>0.291165</td>
      <td>0.416834</td>
      <td>0.142898</td>
      <td>0.625622</td>
      <td>0.434864</td>
      <td>0.054839</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>0bRMsq5beAH609g1mPaLm3</td>
    </tr>
    <tr>
      <th>89</th>
      <td>0.578307</td>
      <td>0.883982</td>
      <td>0.454545</td>
      <td>0.792065</td>
      <td>1.0</td>
      <td>0.020078</td>
      <td>0.218876</td>
      <td>0.000005</td>
      <td>0.286931</td>
      <td>0.605602</td>
      <td>0.280318</td>
      <td>0.046957</td>
      <td>0.8</td>
      <td>0.010638</td>
      <td>0.010753</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>5qie33Dvuvsu2ZzXOw1bQo</td>
    </tr>
    <tr>
      <th>90</th>
      <td>0.245734</td>
      <td>0.403908</td>
      <td>0.000000</td>
      <td>0.748677</td>
      <td>1.0</td>
      <td>0.006975</td>
      <td>0.719880</td>
      <td>0.003407</td>
      <td>0.242301</td>
      <td>0.240233</td>
      <td>0.223208</td>
      <td>0.052455</td>
      <td>0.8</td>
      <td>0.223404</td>
      <td>0.494624</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>4wqoHJMex8lTiWdOxJkH6Q</td>
    </tr>
    <tr>
      <th>91</th>
      <td>0.435931</td>
      <td>0.704954</td>
      <td>0.181818</td>
      <td>0.745168</td>
      <td>1.0</td>
      <td>0.018176</td>
      <td>0.666667</td>
      <td>0.164329</td>
      <td>0.087211</td>
      <td>0.321315</td>
      <td>0.351401</td>
      <td>0.040716</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>2YpjHCOBNoXDq56z1lCDiz</td>
    </tr>
    <tr>
      <th>92</th>
      <td>0.667427</td>
      <td>0.652946</td>
      <td>1.000000</td>
      <td>0.782411</td>
      <td>1.0</td>
      <td>0.015746</td>
      <td>0.416667</td>
      <td>0.000000</td>
      <td>0.193614</td>
      <td>0.963964</td>
      <td>0.486136</td>
      <td>0.031353</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>69guG7XNldszA0qoZKeyyj</td>
    </tr>
    <tr>
      <th>93</th>
      <td>0.354418</td>
      <td>0.688952</td>
      <td>0.181818</td>
      <td>0.755546</td>
      <td>1.0</td>
      <td>0.011413</td>
      <td>0.201807</td>
      <td>0.003317</td>
      <td>0.332576</td>
      <td>0.371365</td>
      <td>0.664423</td>
      <td>0.049108</td>
      <td>0.8</td>
      <td>0.010638</td>
      <td>0.053763</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>5cSGT0sBGDGb3BXGk0qs55</td>
    </tr>
    <tr>
      <th>94</th>
      <td>0.351157</td>
      <td>0.700954</td>
      <td>0.000000</td>
      <td>0.764996</td>
      <td>1.0</td>
      <td>0.012153</td>
      <td>0.001315</td>
      <td>0.275551</td>
      <td>0.337648</td>
      <td>0.673670</td>
      <td>0.805767</td>
      <td>0.042141</td>
      <td>0.8</td>
      <td>0.053191</td>
      <td>0.075269</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>6QXbrn7Jas9WMfxd7Jbsn6</td>
    </tr>
    <tr>
      <th>95</th>
      <td>0.124008</td>
      <td>0.562932</td>
      <td>0.454545</td>
      <td>0.694596</td>
      <td>0.0</td>
      <td>0.016485</td>
      <td>0.007149</td>
      <td>0.069339</td>
      <td>0.621658</td>
      <td>0.533529</td>
      <td>0.306433</td>
      <td>0.030448</td>
      <td>0.8</td>
      <td>0.191489</td>
      <td>0.430108</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>3JsrzNokPddRU7Y14saf9n</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0.422889</td>
      <td>0.429912</td>
      <td>0.181818</td>
      <td>0.688710</td>
      <td>1.0</td>
      <td>0.021029</td>
      <td>0.776104</td>
      <td>0.000242</td>
      <td>0.083763</td>
      <td>0.415410</td>
      <td>0.606519</td>
      <td>0.042059</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>4aJUbjTjKgv1L0Roph6ntJ</td>
    </tr>
    <tr>
      <th>97</th>
      <td>0.564178</td>
      <td>0.754962</td>
      <td>1.000000</td>
      <td>0.746041</td>
      <td>0.0</td>
      <td>0.017965</td>
      <td>0.000290</td>
      <td>0.000000</td>
      <td>0.374163</td>
      <td>0.684682</td>
      <td>0.523421</td>
      <td>0.061911</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>5bAksw6xAIvki1AdTr2b3M</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0.539181</td>
      <td>0.644945</td>
      <td>0.454545</td>
      <td>0.720160</td>
      <td>1.0</td>
      <td>0.011519</td>
      <td>0.000505</td>
      <td>0.548096</td>
      <td>0.096239</td>
      <td>0.962963</td>
      <td>0.464297</td>
      <td>0.025757</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>4zU9UoOqteCrnFpjGRselX</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0.489186</td>
      <td>0.578935</td>
      <td>0.818182</td>
      <td>0.761172</td>
      <td>1.0</td>
      <td>0.010356</td>
      <td>0.495984</td>
      <td>0.000000</td>
      <td>0.101311</td>
      <td>0.331325</td>
      <td>0.495324</td>
      <td>0.074579</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Athens Indie</td>
      <td>0H00TGxLXWoZ6dJohP2MwF</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 18 columns</p>
</div>



#### 2. Select a random song from the given genre



```python
# Counting number of available genres
print("Number of genres available: {}".format(len(data['genre'].unique())))
data['genre'].unique()

# Checking the most popular genres
data['genre'].value_counts()
```


    Number of genres available: 1096





    Accordion                  195
    Footwork                   100
    Chanson Quebecois          100
    Fidget House               100
    KC Indie                   100
    Deep Gothic Post-Punk      100
    Skiffle                    100
    Nu Disco                   100
    LDS                        100
    Danish Indie               100
    Ambient Dub Techno         100
    Deep German Punk           100
    Deep Discofox              100
    Traditional Rockabilly     100
    Irish Indie                100
    Melodic Hard Rock          100
    Breaks                     100
    Deep Deep Tech House       100
    French Folk Pop            100
    Belgian Indie              100
    Ostrock                    100
    Hard Stoner Rock           100
    Barbershop                 100
    Deep German Indie          100
    Samba-Enredo               100
    Witch House                100
    Rio De La Plata            100
    UK Dub                     100
    Solipsynthm                100
    New Age Piano              100
                              ... 
    Azonto                      42
    Gothic Symphonic Metal      41
    Filmi                       41
    Crossover Thrash            40
    Electro-Industrial          40
    Byzantine                   40
    Voidgaze                    39
    Old-Time                    39
    Acid House                  39
    Indian Pop                  38
    Desi Hip Hop                38
    Straight Edge               38
    Symphonic Black Metal       37
    Mande Pop                   37
    Doom Metal                  34
    Post-Doom Metal             32
    Deathgrind                  31
    Jungle                      31
    Drone Metal                 30
    Avantgarde Metal            28
    Acoustic Blues              26
    Classical Performance       26
    Early Music                 25
    Louisiana Blues             24
    Pagan Black Metal           22
    Grindcore                   21
    Brutal Death Metal          19
    Pop                         18
    Atmospheric Black Metal     13
    Prank                       12
    Name: genre, Length: 1096, dtype: int64



From the list above, we see that the taxonomy of styles can be too granular (we have Hebrew pop, Domenican pop, etc.). From all that, we would ideally get only "Pop". Here, we are going to focus on the following 5 styles:
- Pop
- Rock
- Jazz
- House
- Hip hop
- Opera

In the following cell, we are going to create these categories



```python
# Collapsing relevant song categories

category = []

for s in data['genre']:
    if (s.find('Pop')>0):
        value = 'Pop'
    elif (s.find('Rock')>0):
        value = 'Rock'
    elif (s.find('Jazz')>0):
        value = 'Jazz'
    elif (s.find('Hip Hop')>0):
        value = 'Hip Hop'
    elif (s.find('House')>0):
        value = 'House'
    elif (s.find('Opera')>0):
        value = 'Opera'
    else:
        value = s
    
    category.append(value)
        
    
data['category'] = category

data['category'].value_counts()
```





    Pop                         7646
    Rock                        7118
    Jazz                        2825
    House                       2554
    Hip Hop                     2218
    Opera                        268
    Accordion                    195
    Lithumania                   100
    Neue Deutsche Welle          100
    Ambient IDM                  100
    Sega                         100
    Corrosion                    100
    Traditional British Folk     100
    Dubstep Product              100
    Native American              100
    Deep Freestyle               100
    Deep Thrash Metal            100
    Minimal Dubstep              100
    Vocaloid                     100
    Chill-Out Trance             100
    Neo-Synthpop                 100
    Dark Electro-Industrial      100
    Thrash-Groove Metal          100
    Northern Irish Indie         100
    Hyphy                        100
    Ambient Psychill             100
    Perth Indie                  100
    Cyber Metal                  100
    Melodipop                    100
    Tribute                      100
                                ... 
    Post-Metal                    44
    Detroit Techno                43
    Futurepop                     43
    Breakbeat                     43
    Coupe Decale                  42
    Azonto                        42
    Gothic Symphonic Metal        41
    Filmi                         41
    Byzantine                     40
    Electro-Industrial            40
    Crossover Thrash              40
    Old-Time                      39
    Voidgaze                      39
    Straight Edge                 38
    Symphonic Black Metal         37
    Doom Metal                    34
    Post-Doom Metal               32
    Jungle                        31
    Deathgrind                    31
    Drone Metal                   30
    Avantgarde Metal              28
    Classical Performance         26
    Acoustic Blues                26
    Early Music                   25
    Louisiana Blues               24
    Pagan Black Metal             22
    Grindcore                     21
    Brutal Death Metal            19
    Atmospheric Black Metal       13
    Prank                         12
    Name: category, Length: 854, dtype: int64



Now we get a much more reasonable result, at least with respect to the categories that we are interested in: Pop, Rock, Jazz, Hip Hop, House and Opera



```python
# Select a genre to generate playlist
selected_genre = 'Opera'
N = 100

genre_data = data[data.category==selected_genre]


# Sorting
genre_data = genre_data.sort_values(by=['popularity','artist_popularity'], ascending=False)
genre_data.head(10)
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
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>duration_ms</th>
      <th>time_signature</th>
      <th>popularity</th>
      <th>artist_popularity</th>
      <th>explicit</th>
      <th>genre</th>
      <th>id</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>72433</th>
      <td>0.108793</td>
      <td>0.095660</td>
      <td>0.454545</td>
      <td>0.554277</td>
      <td>1.0</td>
      <td>0.018599</td>
      <td>0.879518</td>
      <td>0.897796</td>
      <td>0.385321</td>
      <td>0.037228</td>
      <td>0.493511</td>
      <td>0.041012</td>
      <td>0.6</td>
      <td>0.500000</td>
      <td>0.677419</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>0BqbWiOxccb0qQshm3Whjt</td>
      <td>Opera</td>
    </tr>
    <tr>
      <th>72444</th>
      <td>0.187045</td>
      <td>0.321895</td>
      <td>0.181818</td>
      <td>0.687875</td>
      <td>0.0</td>
      <td>0.009722</td>
      <td>0.678715</td>
      <td>0.834669</td>
      <td>0.114497</td>
      <td>0.253246</td>
      <td>0.272107</td>
      <td>0.033346</td>
      <td>0.8</td>
      <td>0.468085</td>
      <td>0.548387</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>4WYZuhsa8idA4kPDPsNY6B</td>
      <td>Opera</td>
    </tr>
    <tr>
      <th>72435</th>
      <td>0.143571</td>
      <td>0.301892</td>
      <td>0.454545</td>
      <td>0.598426</td>
      <td>1.0</td>
      <td>0.050724</td>
      <td>0.995984</td>
      <td>0.019439</td>
      <td>0.677446</td>
      <td>0.127118</td>
      <td>0.206239</td>
      <td>0.063333</td>
      <td>0.8</td>
      <td>0.404255</td>
      <td>0.623656</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>3Vb0EDj3nLxxK3IOcp27j3</td>
      <td>Opera</td>
    </tr>
    <tr>
      <th>72482</th>
      <td>0.097924</td>
      <td>0.393906</td>
      <td>0.363636</td>
      <td>0.710636</td>
      <td>1.0</td>
      <td>0.013104</td>
      <td>0.917671</td>
      <td>0.301603</td>
      <td>0.096239</td>
      <td>0.168160</td>
      <td>0.269246</td>
      <td>0.058589</td>
      <td>0.8</td>
      <td>0.404255</td>
      <td>0.483871</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>1zZAo2svewOprAfphYjGL5</td>
      <td>Opera</td>
    </tr>
    <tr>
      <th>72440</th>
      <td>0.110966</td>
      <td>0.071656</td>
      <td>0.090909</td>
      <td>0.452351</td>
      <td>0.0</td>
      <td>0.028321</td>
      <td>0.974900</td>
      <td>0.866733</td>
      <td>0.106382</td>
      <td>0.039930</td>
      <td>0.265849</td>
      <td>0.062893</td>
      <td>0.8</td>
      <td>0.393617</td>
      <td>0.500000</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>3KpKYUwzIEMYDv77Qygun4</td>
      <td>Opera</td>
    </tr>
    <tr>
      <th>72450</th>
      <td>0.385936</td>
      <td>0.168871</td>
      <td>0.090909</td>
      <td>0.625791</td>
      <td>0.0</td>
      <td>0.028849</td>
      <td>0.995984</td>
      <td>0.000391</td>
      <td>0.073214</td>
      <td>0.311304</td>
      <td>0.281935</td>
      <td>0.071341</td>
      <td>0.8</td>
      <td>0.382979</td>
      <td>0.516129</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>6YJiRaVLOrCTe9f6t0P2nx</td>
      <td>Opera</td>
    </tr>
    <tr>
      <th>72411</th>
      <td>0.085969</td>
      <td>0.099860</td>
      <td>0.454545</td>
      <td>0.496742</td>
      <td>1.0</td>
      <td>0.026102</td>
      <td>0.992972</td>
      <td>0.019138</td>
      <td>0.175356</td>
      <td>0.065056</td>
      <td>0.670060</td>
      <td>0.084589</td>
      <td>1.0</td>
      <td>0.372340</td>
      <td>0.430108</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>2wGCbQEmMAbt5X30s7pnSc</td>
      <td>Opera</td>
    </tr>
    <tr>
      <th>72429</th>
      <td>0.093577</td>
      <td>0.133866</td>
      <td>0.636364</td>
      <td>0.575311</td>
      <td>0.0</td>
      <td>0.022720</td>
      <td>0.963855</td>
      <td>0.000000</td>
      <td>0.087617</td>
      <td>0.102093</td>
      <td>0.186111</td>
      <td>0.045105</td>
      <td>0.8</td>
      <td>0.372340</td>
      <td>0.370968</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>0YRppTZQqA5h3zTat2EpvC</td>
      <td>Opera</td>
    </tr>
    <tr>
      <th>72412</th>
      <td>0.290295</td>
      <td>0.152869</td>
      <td>0.909091</td>
      <td>0.533910</td>
      <td>1.0</td>
      <td>0.036035</td>
      <td>0.948795</td>
      <td>0.000011</td>
      <td>0.734248</td>
      <td>0.278271</td>
      <td>0.211123</td>
      <td>0.034943</td>
      <td>0.8</td>
      <td>0.361702</td>
      <td>0.634409</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>3iVh3cgWwbYNFGZz5W8g12</td>
      <td>Opera</td>
    </tr>
    <tr>
      <th>72438</th>
      <td>0.244647</td>
      <td>0.085058</td>
      <td>0.454545</td>
      <td>0.516737</td>
      <td>1.0</td>
      <td>0.018599</td>
      <td>0.969880</td>
      <td>0.023246</td>
      <td>0.309247</td>
      <td>0.121112</td>
      <td>0.378998</td>
      <td>0.057035</td>
      <td>0.6</td>
      <td>0.361702</td>
      <td>0.434409</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>7LOytcUG9JFhM8AEmN15Xr</td>
      <td>Opera</td>
    </tr>
  </tbody>
</table>
</div>





```python
seed = genre_data.iloc[0,:]
seed
```





    danceability                       0.108793
    energy                            0.0956598
    key                                0.454545
    loudness                           0.554277
    mode                                      1
    speechiness                       0.0185988
    acousticness                       0.879518
    instrumentalness                   0.897796
    liveness                           0.385321
    valence                           0.0372276
    tempo                              0.493511
    duration_ms                        0.041012
    time_signature                          0.6
    popularity                              0.5
    artist_popularity                  0.677419
    explicit                                  0
    genre                                 Opera
    id                   0BqbWiOxccb0qQshm3Whjt
    category                              Opera
    Name: 72433, dtype: object





```python
# Getting feature values for our seed song
popularity = seed.popularity
artist_popularity = seed.artist_popularity
speechiness = seed.speechiness
```


#### 3. Get the closest N songs on those features (by euclidean distance)



```python
# Calculating euclidean distance for every song with respect to the seed song
distance = []

# We will be putting different weights on these three features
for i in genre_data.index:
#     print(i)
    d = np.sqrt((3*(genre_data.loc[i,'popularity']-popularity))**2 + (2.5*(genre_data.loc[i,'artist_popularity']-artist_popularity))**2 + (genre_data.loc[i,'speechiness']-speechiness)**2)  
    distance.append(d)
    
distance
```





    [0.0,
     0.33660676159120606,
     0.31874926007259402,
     0.56272969052333299,
     0.54652148531033817,
     0.53473538809447296,
     0.72732291502883872,
     0.8565298487986347,
     0.42895543984815759,
     0.73568038267847502,
     0.44718136557292221,
     0.50813131753513918,
     0.64880110853548922,
     0.71492094693789832,
     0.51189466928967831,
     0.51349368005216411,
     0.78815311031334889,
     0.80257050362250115,
     0.88413257449735461,
     0.68264183779771137,
     0.90755919732156776,
     0.6707517268333768,
     0.74382137412843208,
     0.82663060209749595,
     0.9491190676740543,
     0.85271213309028759,
     0.8018065401096045,
     0.8647431756740156,
     0.89429509048294564,
     0.89443039042735484,
     0.95976874161342762,
     0.95151204810076939,
     1.1531176636383069,
     1.4716632406536172,
     1.5400369868256241,
     0.83856391523929996,
     1.0316059798885198,
     1.0348416125086994,
     1.1024320077031091,
     1.2987911528147176,
     1.1097488626464891,
     1.126663163696785,
     1.3159817211616767,
     0.9209722360413265,
     1.6765277546250881,
     0.99265296983776008,
     1.1408754537833223,
     1.3938742504794672,
     1.0269221158507038,
     1.0884157739352163,
     1.7529593367547818,
     1.4167317817219631,
     1.6715581728345881,
     1.7288393929389974,
     1.0857380386027404,
     1.3209948410110965,
     1.7274598734379936,
     1.1393378171370552,
     1.217323989834018,
     1.2899918883961619,
     1.4098766654328585,
     1.8637894024938004,
     1.8962083291857816,
     1.1667378921935148,
     1.2922187776300349,
     1.3005102158444199,
     1.3177090376605243,
     1.4038672152404212,
     1.5284980478899048,
     1.2522089783767409,
     1.2761448794207049,
     1.3288012097664128,
     1.3860836446039708,
     1.5497068287090059,
     1.6338140191268073,
     1.8094122017953873,
     1.8504966555918769,
     1.913275173965566,
     1.9558915963009016,
     1.22043762663814,
     1.386524254429532,
     1.386534563587724,
     1.4868788694958062,
     1.5352325682866041,
     1.770800594753567,
     1.2609074412202592,
     1.3198616773610865,
     1.37481729209598,
     1.912175438780743,
     1.9326819100462791,
     1.9741277364409295,
     1.418813790622077,
     1.704533889334926,
     1.7587161124701249,
     1.8554415248783733,
     1.893097962466016,
     1.9030338550506258,
     1.9738274274594028,
     1.9737929638166016,
     1.994375363669759,
     2.015159593309936,
     2.0359718954132031,
     2.0781236242147028,
     1.35614371813199,
     1.4286659337951939,
     1.4360314398368841,
     1.496352908318312,
     1.5051047216435476,
     1.5098072328959566,
     1.5259020174621021,
     1.9344428355998129,
     1.9543267435551483,
     2.0149651513095135,
     2.0793440567376318,
     2.0979114258920499,
     1.4191132461596723,
     1.4256374365151288,
     1.4443271193180311,
     1.5434883745753263,
     1.6638281028290642,
     1.7017284568872086,
     1.8420489129652233,
     2.0765894139437937,
     2.0972329740036479,
     1.4230469296063608,
     1.5527103218738347,
     1.5783061552150996,
     1.8295202499124061,
     2.0370311435356498,
     2.0370747940578071,
     2.0770757972679554,
     2.1383514994178303,
     2.1383470473880157,
     2.1382847232198063,
     2.1382743931293864,
     1.4134943726432398,
     1.5240284901056802,
     1.6682029350005119,
     1.9070626567992102,
     1.9253667449658312,
     1.9719847109856654,
     2.098306160503455,
     2.1385485330808112,
     2.1385628278084625,
     2.1385767699713329,
     2.1385510395605132,
     2.1691277926394679,
     2.1809293227827986,
     1.4407205464830559,
     1.5366077656309891,
     1.5433158014008626,
     1.7052272011566674,
     1.7174659211531536,
     1.7419200728200257,
     1.7469592158491085,
     1.7736011460789578,
     1.778133368232105,
     1.8942368581858218,
     1.9670677757863604,
     2.0042425717214072,
     2.0420107314530727,
     2.1001273244297889,
     2.1002470176248664,
     2.1099222576198327,
     2.1596473591846106,
     2.2000851844507081,
     2.2205159982880178,
     1.5272997555958907,
     1.5728802828536201,
     1.7152900545588052,
     1.7583524271289876,
     1.7689945195024017,
     1.8197476131857273,
     1.8768789702140001,
     1.9460977402234376,
     1.9723951438079028,
     1.9995511373993815,
     2.0088022839652124,
     2.0835617637438704,
     2.083563586022021,
     2.0835797935081564,
     2.1027229912527305,
     2.1223092885441237,
     2.1611974405795551,
     2.200947318015102,
     2.2009807439039841,
     2.2009626281296732,
     2.2009663953909167,
     2.2009496316631076,
     2.2214317235294128,
     2.2210479891190729,
     2.221068241302838,
     2.2311510913954891,
     2.2412916275829429,
     2.2413009945909779,
     2.241311358043613,
     2.2413086675364768,
     2.2413386391053298,
     2.2412922055484019,
     1.5227269064580295,
     1.532435464859365,
     1.6822983039262429,
     1.7292058585646295,
     1.7358953155450394,
     1.7427020344748982,
     1.7718939230329165,
     1.8014540758782598,
     1.8147943792066639,
     1.8337268478623912,
     1.8458498633004403,
     1.8613916122432324,
     1.9439992863210926,
     1.9506709799180715,
     2.0503620966807943,
     2.0595513612849263,
     2.0626213468503076,
     2.1062822349715704,
     2.1571296796451316,
     2.158718078809609,
     2.1635869542151864,
     2.1830451735156799,
     2.2034113662750467,
     2.2026825998706867,
     2.2306522312117614,
     2.2223823084458472,
     2.2224196930538676,
     2.2223636234433517,
     2.2223672715158824,
     2.2224807130308615,
     2.2223797156190672,
     2.2223645203873597,
     2.2224081963488991,
     2.2323000936185475,
     2.242313407449041,
     2.2422708762138588,
     2.2422832272879916,
     2.2422782071821956,
     2.2522923459495354,
     2.2522946068538547,
     2.2623255448914725,
     2.2800729397769315,
     2.268530664396939,
     2.2657396838813759,
     2.2623263716942987,
     2.262323331032865,
     2.262344980848122,
     2.2647751292960345,
     2.2630669299389727,
     2.2638513037498806,
     2.262335967521115,
     2.2623236568182139,
     2.2628823096754744,
     2.2843838500422171,
     2.2630374287183264,
     2.2629229558010784,
     2.2624340953479987,
     2.2623688220602451,
     2.2623503833973984,
     2.2623345854086505,
     2.2623257003797881,
     2.2623336006530042,
     2.2623236568182139,
     2.2623478536534551,
     2.2623617363597757,
     2.2624340953479987,
     2.2623332822732674,
     2.2623519604753626,
     2.2623245946848662]



We use different weights on these three features (3, 2.5 and 1 respectively) because according to our model, song popularity is the most important feature, followed by artist popularity and only then followed by speechiness.



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
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>duration_ms</th>
      <th>time_signature</th>
      <th>popularity</th>
      <th>artist_popularity</th>
      <th>explicit</th>
      <th>genre</th>
      <th>id</th>
      <th>category</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.108793</td>
      <td>0.095660</td>
      <td>0.454545</td>
      <td>0.554277</td>
      <td>1.0</td>
      <td>0.018599</td>
      <td>0.879518</td>
      <td>0.897796</td>
      <td>0.385321</td>
      <td>0.037228</td>
      <td>0.493511</td>
      <td>0.041012</td>
      <td>0.6</td>
      <td>0.500000</td>
      <td>0.677419</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>0BqbWiOxccb0qQshm3Whjt</td>
      <td>Opera</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.187045</td>
      <td>0.321895</td>
      <td>0.181818</td>
      <td>0.687875</td>
      <td>0.0</td>
      <td>0.009722</td>
      <td>0.678715</td>
      <td>0.834669</td>
      <td>0.114497</td>
      <td>0.253246</td>
      <td>0.272107</td>
      <td>0.033346</td>
      <td>0.8</td>
      <td>0.468085</td>
      <td>0.548387</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>4WYZuhsa8idA4kPDPsNY6B</td>
      <td>Opera</td>
      <td>0.336607</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.143571</td>
      <td>0.301892</td>
      <td>0.454545</td>
      <td>0.598426</td>
      <td>1.0</td>
      <td>0.050724</td>
      <td>0.995984</td>
      <td>0.019439</td>
      <td>0.677446</td>
      <td>0.127118</td>
      <td>0.206239</td>
      <td>0.063333</td>
      <td>0.8</td>
      <td>0.404255</td>
      <td>0.623656</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>3Vb0EDj3nLxxK3IOcp27j3</td>
      <td>Opera</td>
      <td>0.318749</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.097924</td>
      <td>0.393906</td>
      <td>0.363636</td>
      <td>0.710636</td>
      <td>1.0</td>
      <td>0.013104</td>
      <td>0.917671</td>
      <td>0.301603</td>
      <td>0.096239</td>
      <td>0.168160</td>
      <td>0.269246</td>
      <td>0.058589</td>
      <td>0.8</td>
      <td>0.404255</td>
      <td>0.483871</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>1zZAo2svewOprAfphYjGL5</td>
      <td>Opera</td>
      <td>0.562730</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.110966</td>
      <td>0.071656</td>
      <td>0.090909</td>
      <td>0.452351</td>
      <td>0.0</td>
      <td>0.028321</td>
      <td>0.974900</td>
      <td>0.866733</td>
      <td>0.106382</td>
      <td>0.039930</td>
      <td>0.265849</td>
      <td>0.062893</td>
      <td>0.8</td>
      <td>0.393617</td>
      <td>0.500000</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>3KpKYUwzIEMYDv77Qygun4</td>
      <td>Opera</td>
      <td>0.546521</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.385936</td>
      <td>0.168871</td>
      <td>0.090909</td>
      <td>0.625791</td>
      <td>0.0</td>
      <td>0.028849</td>
      <td>0.995984</td>
      <td>0.000391</td>
      <td>0.073214</td>
      <td>0.311304</td>
      <td>0.281935</td>
      <td>0.071341</td>
      <td>0.8</td>
      <td>0.382979</td>
      <td>0.516129</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>6YJiRaVLOrCTe9f6t0P2nx</td>
      <td>Opera</td>
      <td>0.534735</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.085969</td>
      <td>0.099860</td>
      <td>0.454545</td>
      <td>0.496742</td>
      <td>1.0</td>
      <td>0.026102</td>
      <td>0.992972</td>
      <td>0.019138</td>
      <td>0.175356</td>
      <td>0.065056</td>
      <td>0.670060</td>
      <td>0.084589</td>
      <td>1.0</td>
      <td>0.372340</td>
      <td>0.430108</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>2wGCbQEmMAbt5X30s7pnSc</td>
      <td>Opera</td>
      <td>0.727323</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.093577</td>
      <td>0.133866</td>
      <td>0.636364</td>
      <td>0.575311</td>
      <td>0.0</td>
      <td>0.022720</td>
      <td>0.963855</td>
      <td>0.000000</td>
      <td>0.087617</td>
      <td>0.102093</td>
      <td>0.186111</td>
      <td>0.045105</td>
      <td>0.8</td>
      <td>0.372340</td>
      <td>0.370968</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>0YRppTZQqA5h3zTat2EpvC</td>
      <td>Opera</td>
      <td>0.856530</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.290295</td>
      <td>0.152869</td>
      <td>0.909091</td>
      <td>0.533910</td>
      <td>1.0</td>
      <td>0.036035</td>
      <td>0.948795</td>
      <td>0.000011</td>
      <td>0.734248</td>
      <td>0.278271</td>
      <td>0.211123</td>
      <td>0.034943</td>
      <td>0.8</td>
      <td>0.361702</td>
      <td>0.634409</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>3iVh3cgWwbYNFGZz5W8g12</td>
      <td>Opera</td>
      <td>0.428955</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.244647</td>
      <td>0.085058</td>
      <td>0.454545</td>
      <td>0.516737</td>
      <td>1.0</td>
      <td>0.018599</td>
      <td>0.969880</td>
      <td>0.023246</td>
      <td>0.309247</td>
      <td>0.121112</td>
      <td>0.378998</td>
      <td>0.057035</td>
      <td>0.6</td>
      <td>0.361702</td>
      <td>0.434409</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>7LOytcUG9JFhM8AEmN15Xr</td>
      <td>Opera</td>
      <td>0.735680</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.171829</td>
      <td>0.017848</td>
      <td>1.000000</td>
      <td>0.252697</td>
      <td>1.0</td>
      <td>0.022086</td>
      <td>0.873494</td>
      <td>0.140281</td>
      <td>0.183471</td>
      <td>0.038929</td>
      <td>0.424914</td>
      <td>0.043075</td>
      <td>0.8</td>
      <td>0.351064</td>
      <td>0.684588</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>1tf5l5Cur6hdwoStk1f1WY</td>
      <td>Opera</td>
      <td>0.447181</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.269645</td>
      <td>0.156869</td>
      <td>0.090909</td>
      <td>0.611552</td>
      <td>1.0</td>
      <td>0.023777</td>
      <td>0.964859</td>
      <td>0.003357</td>
      <td>0.092182</td>
      <td>0.242235</td>
      <td>0.485370</td>
      <td>0.059259</td>
      <td>0.2</td>
      <td>0.351064</td>
      <td>0.580645</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>5M9QQBxwH8ybk3xOZEJei4</td>
      <td>Opera</td>
      <td>0.508131</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.210955</td>
      <td>0.158870</td>
      <td>0.090909</td>
      <td>0.661642</td>
      <td>1.0</td>
      <td>0.019338</td>
      <td>0.985944</td>
      <td>0.000000</td>
      <td>0.101311</td>
      <td>0.065957</td>
      <td>0.281379</td>
      <td>0.055366</td>
      <td>0.8</td>
      <td>0.351064</td>
      <td>0.489247</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>3asyS9IMGkZb5HrbEnrwpH</td>
      <td>Opera</td>
      <td>0.648801</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.089229</td>
      <td>0.031950</td>
      <td>0.181818</td>
      <td>0.338674</td>
      <td>0.0</td>
      <td>0.026736</td>
      <td>0.766064</td>
      <td>0.569138</td>
      <td>0.062665</td>
      <td>0.037127</td>
      <td>0.254829</td>
      <td>0.084353</td>
      <td>0.6</td>
      <td>0.340426</td>
      <td>0.465054</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>4E0bxKybeThhGkYoie27Nf</td>
      <td>Opera</td>
      <td>0.714921</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.081622</td>
      <td>0.039751</td>
      <td>0.363636</td>
      <td>0.493196</td>
      <td>1.0</td>
      <td>0.018493</td>
      <td>0.992972</td>
      <td>0.952906</td>
      <td>0.095225</td>
      <td>0.121112</td>
      <td>0.196486</td>
      <td>0.049150</td>
      <td>0.8</td>
      <td>0.329787</td>
      <td>0.691756</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>1KeasaJ8MshakeNvRqbDzx</td>
      <td>Opera</td>
      <td>0.511895</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.092490</td>
      <td>0.244883</td>
      <td>0.272727</td>
      <td>0.648646</td>
      <td>1.0</td>
      <td>0.012787</td>
      <td>0.984940</td>
      <td>0.484970</td>
      <td>0.099282</td>
      <td>0.149141</td>
      <td>0.202996</td>
      <td>0.044947</td>
      <td>0.8</td>
      <td>0.329787</td>
      <td>0.655914</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>1wgUTAbPQkblN0nykDueMI</td>
      <td>Opera</td>
      <td>0.513494</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.200087</td>
      <td>0.194875</td>
      <td>0.727273</td>
      <td>0.631361</td>
      <td>1.0</td>
      <td>0.016697</td>
      <td>0.977912</td>
      <td>0.000117</td>
      <td>0.096239</td>
      <td>0.146138</td>
      <td>0.275799</td>
      <td>0.032806</td>
      <td>1.0</td>
      <td>0.329787</td>
      <td>0.437276</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>23XpALLxFdVMMYe5z5BGJc</td>
      <td>Opera</td>
      <td>0.788153</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.202261</td>
      <td>0.243883</td>
      <td>0.090909</td>
      <td>0.664983</td>
      <td>1.0</td>
      <td>0.016591</td>
      <td>0.994980</td>
      <td>0.003036</td>
      <td>0.061853</td>
      <td>0.082774</td>
      <td>0.261946</td>
      <td>0.035542</td>
      <td>0.6</td>
      <td>0.319149</td>
      <td>0.440860</td>
      <td>0.0</td>
      <td>Deep Opera</td>
      <td>0vFIcgSTRg0fTlESbPPCsJ</td>
      <td>Opera</td>
      <td>0.802571</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.351157</td>
      <td>0.353900</td>
      <td>0.363636</td>
      <td>0.737816</td>
      <td>0.0</td>
      <td>0.012681</td>
      <td>0.917671</td>
      <td>0.000000</td>
      <td>0.135797</td>
      <td>0.277270</td>
      <td>0.396887</td>
      <td>0.040148</td>
      <td>0.8</td>
      <td>0.308511</td>
      <td>0.408602</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>6h1EIRHC2cjricHyk9UVQ7</td>
      <td>Opera</td>
      <td>0.884133</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.264210</td>
      <td>0.416910</td>
      <td>0.090909</td>
      <td>0.718137</td>
      <td>1.0</td>
      <td>0.012047</td>
      <td>0.815261</td>
      <td>0.000001</td>
      <td>0.085893</td>
      <td>0.155147</td>
      <td>0.216838</td>
      <td>0.071128</td>
      <td>0.8</td>
      <td>0.287234</td>
      <td>0.580645</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>2joixSFXKk1uYN7czBQzBY</td>
      <td>Opera</td>
      <td>0.682642</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.133790</td>
      <td>0.256885</td>
      <td>0.454545</td>
      <td>0.713625</td>
      <td>0.0</td>
      <td>0.021135</td>
      <td>0.986948</td>
      <td>0.000012</td>
      <td>0.650059</td>
      <td>0.103094</td>
      <td>0.135105</td>
      <td>0.061711</td>
      <td>0.6</td>
      <td>0.287234</td>
      <td>0.419355</td>
      <td>0.0</td>
      <td>Deep Opera</td>
      <td>3GJdkrB0Vy42ZQRrR2nY2o</td>
      <td>Opera</td>
      <td>0.907559</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.230518</td>
      <td>0.156869</td>
      <td>0.818182</td>
      <td>0.602974</td>
      <td>1.0</td>
      <td>0.018916</td>
      <td>0.964859</td>
      <td>0.024048</td>
      <td>0.105368</td>
      <td>0.070461</td>
      <td>0.468471</td>
      <td>0.051822</td>
      <td>0.8</td>
      <td>0.276596</td>
      <td>0.666667</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>1opcgxc7LjcutXrx4hGBN2</td>
      <td>Opera</td>
      <td>0.670752</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.380502</td>
      <td>0.498922</td>
      <td>0.636364</td>
      <td>0.734753</td>
      <td>0.0</td>
      <td>0.013421</td>
      <td>0.789157</td>
      <td>0.000005</td>
      <td>0.090660</td>
      <td>0.376370</td>
      <td>0.364192</td>
      <td>0.049682</td>
      <td>0.8</td>
      <td>0.276596</td>
      <td>0.548387</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>2IzAde95HP2wDt6QS6H7Sw</td>
      <td>Opera</td>
      <td>0.743821</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.214216</td>
      <td>0.187874</td>
      <td>0.000000</td>
      <td>0.603921</td>
      <td>1.0</td>
      <td>0.019973</td>
      <td>0.992972</td>
      <td>0.010621</td>
      <td>0.072707</td>
      <td>0.046237</td>
      <td>0.261189</td>
      <td>0.045066</td>
      <td>0.8</td>
      <td>0.276596</td>
      <td>0.483871</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>7id0eYuQAxnwgd03jCQzr1</td>
      <td>Opera</td>
      <td>0.826631</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.183784</td>
      <td>0.354900</td>
      <td>0.636364</td>
      <td>0.683920</td>
      <td>0.0</td>
      <td>0.018387</td>
      <td>0.968876</td>
      <td>0.245491</td>
      <td>0.378220</td>
      <td>0.362356</td>
      <td>0.636981</td>
      <td>0.033949</td>
      <td>1.0</td>
      <td>0.276596</td>
      <td>0.408602</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>66iMbl3gTZYWEeWYK4U4dO</td>
      <td>Opera</td>
      <td>0.949119</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.266384</td>
      <td>0.216879</td>
      <td>0.363636</td>
      <td>0.612907</td>
      <td>1.0</td>
      <td>0.016697</td>
      <td>0.954819</td>
      <td>0.000000</td>
      <td>0.274760</td>
      <td>0.398392</td>
      <td>0.253777</td>
      <td>0.048151</td>
      <td>0.8</td>
      <td>0.265957</td>
      <td>0.483871</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>3pTVlPsOH9Gf8JgpMXT8BG</td>
      <td>Opera</td>
      <td>0.852712</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.495707</td>
      <td>0.012347</td>
      <td>1.000000</td>
      <td>0.543026</td>
      <td>0.0</td>
      <td>0.014477</td>
      <td>0.996988</td>
      <td>0.188377</td>
      <td>0.055057</td>
      <td>0.077168</td>
      <td>0.262152</td>
      <td>0.036145</td>
      <td>0.8</td>
      <td>0.255319</td>
      <td>0.548387</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>36JNf7QIvLWb6XI8JePDxL</td>
      <td>Opera</td>
      <td>0.801807</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.277252</td>
      <td>0.217879</td>
      <td>0.000000</td>
      <td>0.607337</td>
      <td>0.0</td>
      <td>0.029695</td>
      <td>0.998996</td>
      <td>0.229459</td>
      <td>0.097253</td>
      <td>0.036627</td>
      <td>0.256245</td>
      <td>0.054719</td>
      <td>0.8</td>
      <td>0.255319</td>
      <td>0.494624</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>5e5Umo93FhQ4BwperG6TjN</td>
      <td>Opera</td>
      <td>0.864743</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.260950</td>
      <td>0.166871</td>
      <td>0.636364</td>
      <td>0.526224</td>
      <td>1.0</td>
      <td>0.027370</td>
      <td>0.990964</td>
      <td>0.003327</td>
      <td>0.087617</td>
      <td>0.064655</td>
      <td>0.206491</td>
      <td>0.036145</td>
      <td>1.0</td>
      <td>0.255319</td>
      <td>0.473118</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>2ZDqxewFaSQxhFGBNTtAxa</td>
      <td>Opera</td>
      <td>0.894295</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.292468</td>
      <td>0.068956</td>
      <td>0.818182</td>
      <td>0.512133</td>
      <td>0.0</td>
      <td>0.036458</td>
      <td>0.993976</td>
      <td>0.872745</td>
      <td>0.316347</td>
      <td>0.083174</td>
      <td>0.179147</td>
      <td>0.034351</td>
      <td>0.6</td>
      <td>0.255319</td>
      <td>0.473118</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>4jWsGe1IZrtrWJvu7KE0UT</td>
      <td>Opera</td>
      <td>0.894430</td>
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
    </tr>
    <tr>
      <th>238</th>
      <td>0.359852</td>
      <td>0.355900</td>
      <td>0.090909</td>
      <td>0.609008</td>
      <td>0.0</td>
      <td>0.010250</td>
      <td>0.952811</td>
      <td>0.026754</td>
      <td>0.058709</td>
      <td>0.606603</td>
      <td>0.258895</td>
      <td>0.019783</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.005376</td>
      <td>0.0</td>
      <td>Chinese Opera</td>
      <td>3KqxuQIWxfqGI71WxTeOuk</td>
      <td>Opera</td>
      <td>2.252295</td>
    </tr>
    <tr>
      <th>239</th>
      <td>0.194653</td>
      <td>0.195875</td>
      <td>0.909091</td>
      <td>0.611811</td>
      <td>1.0</td>
      <td>0.015323</td>
      <td>0.997992</td>
      <td>0.000467</td>
      <td>0.095225</td>
      <td>0.161153</td>
      <td>0.211282</td>
      <td>0.040953</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Deep Opera</td>
      <td>6m9C9MjmU5nfuRu9UadQI5</td>
      <td>Opera</td>
      <td>2.262326</td>
    </tr>
    <tr>
      <th>240</th>
      <td>0.245734</td>
      <td>0.340898</td>
      <td>0.727273</td>
      <td>0.587732</td>
      <td>1.0</td>
      <td>0.302547</td>
      <td>0.981928</td>
      <td>0.000000</td>
      <td>0.946241</td>
      <td>0.289282</td>
      <td>0.262447</td>
      <td>0.021762</td>
      <td>0.6</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Deep Opera</td>
      <td>0APOXWShlZgJH8k6NvptwA</td>
      <td>Opera</td>
      <td>2.280073</td>
    </tr>
    <tr>
      <th>241</th>
      <td>0.415281</td>
      <td>0.296891</td>
      <td>0.272727</td>
      <td>0.655032</td>
      <td>1.0</td>
      <td>0.186305</td>
      <td>0.989960</td>
      <td>0.000000</td>
      <td>0.336633</td>
      <td>0.293286</td>
      <td>0.134408</td>
      <td>0.073822</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Chinese Opera</td>
      <td>5xr7nU3r3adyhjyeyzjW58</td>
      <td>Opera</td>
      <td>2.268531</td>
    </tr>
    <tr>
      <th>242</th>
      <td>0.369634</td>
      <td>0.073756</td>
      <td>0.000000</td>
      <td>0.469914</td>
      <td>0.0</td>
      <td>0.142978</td>
      <td>0.997992</td>
      <td>0.134269</td>
      <td>0.097253</td>
      <td>0.553549</td>
      <td>0.256259</td>
      <td>0.071347</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Chinese Opera</td>
      <td>0TCLrYcImZqpaFWSSkrLp2</td>
      <td>Opera</td>
      <td>2.265740</td>
    </tr>
    <tr>
      <th>243</th>
      <td>0.371807</td>
      <td>0.545930</td>
      <td>0.909091</td>
      <td>0.732952</td>
      <td>1.0</td>
      <td>0.022403</td>
      <td>0.577309</td>
      <td>0.000195</td>
      <td>0.152027</td>
      <td>0.606603</td>
      <td>0.538292</td>
      <td>0.050701</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Chinese Opera</td>
      <td>04YbcNwNzfBkIuYhOvBBG0</td>
      <td>Opera</td>
      <td>2.262326</td>
    </tr>
    <tr>
      <th>244</th>
      <td>0.403326</td>
      <td>0.175872</td>
      <td>0.000000</td>
      <td>0.604459</td>
      <td>1.0</td>
      <td>0.019444</td>
      <td>0.898594</td>
      <td>0.000005</td>
      <td>0.083966</td>
      <td>0.594591</td>
      <td>0.432163</td>
      <td>0.060304</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Chinese Opera</td>
      <td>0dyyO2ypcPpN8OtAjgUPb7</td>
      <td>Opera</td>
      <td>2.262323</td>
    </tr>
    <tr>
      <th>245</th>
      <td>0.453320</td>
      <td>0.068156</td>
      <td>0.727273</td>
      <td>0.564283</td>
      <td>0.0</td>
      <td>0.028532</td>
      <td>0.996988</td>
      <td>0.000025</td>
      <td>0.132754</td>
      <td>0.589585</td>
      <td>0.467456</td>
      <td>0.034033</td>
      <td>0.6</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Chinese Opera</td>
      <td>6JXFifl6WW8GTx4xnk4SQs</td>
      <td>Opera</td>
      <td>2.262345</td>
    </tr>
    <tr>
      <th>246</th>
      <td>0.319639</td>
      <td>0.078057</td>
      <td>0.727273</td>
      <td>0.584297</td>
      <td>1.0</td>
      <td>0.123956</td>
      <td>0.997992</td>
      <td>0.001493</td>
      <td>0.162170</td>
      <td>0.072763</td>
      <td>0.225330</td>
      <td>0.081250</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Chinese Opera</td>
      <td>5XVQcwAEuzDKnJwAq8RZnh</td>
      <td>Opera</td>
      <td>2.264775</td>
    </tr>
    <tr>
      <th>247</th>
      <td>0.420715</td>
      <td>0.128865</td>
      <td>0.545455</td>
      <td>0.554982</td>
      <td>1.0</td>
      <td>0.076614</td>
      <td>0.952811</td>
      <td>0.000012</td>
      <td>0.096239</td>
      <td>0.751749</td>
      <td>0.233280</td>
      <td>0.069945</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Chinese Opera</td>
      <td>73m7Dcli05K1D62MT411Jb</td>
      <td>Opera</td>
      <td>2.263067</td>
    </tr>
    <tr>
      <th>248</th>
      <td>0.308771</td>
      <td>0.139867</td>
      <td>1.000000</td>
      <td>0.537159</td>
      <td>1.0</td>
      <td>0.101765</td>
      <td>0.997992</td>
      <td>0.002435</td>
      <td>0.167241</td>
      <td>0.572568</td>
      <td>0.157088</td>
      <td>0.075527</td>
      <td>0.6</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Chinese Opera</td>
      <td>12skRc73GOSAWClslHvIgY</td>
      <td>Opera</td>
      <td>2.263851</td>
    </tr>
    <tr>
      <th>249</th>
      <td>0.366373</td>
      <td>0.186874</td>
      <td>1.000000</td>
      <td>0.587453</td>
      <td>1.0</td>
      <td>0.010990</td>
      <td>0.932731</td>
      <td>0.001283</td>
      <td>0.252445</td>
      <td>0.481476</td>
      <td>0.486529</td>
      <td>0.063259</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Chinese Opera</td>
      <td>1Qs89WyC1T5jeoJ4w8QXF6</td>
      <td>Opera</td>
      <td>2.262336</td>
    </tr>
    <tr>
      <th>250</th>
      <td>0.454407</td>
      <td>0.196876</td>
      <td>0.090909</td>
      <td>0.625346</td>
      <td>0.0</td>
      <td>0.020078</td>
      <td>0.865462</td>
      <td>0.000000</td>
      <td>0.031627</td>
      <td>0.731729</td>
      <td>0.380521</td>
      <td>0.040319</td>
      <td>0.6</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Chinese Opera</td>
      <td>2hjUf7jpetmnY4jJbpIUmm</td>
      <td>Opera</td>
      <td>2.262324</td>
    </tr>
    <tr>
      <th>251</th>
      <td>0.348984</td>
      <td>0.268887</td>
      <td>0.727273</td>
      <td>0.548670</td>
      <td>0.0</td>
      <td>0.068900</td>
      <td>0.984940</td>
      <td>0.000006</td>
      <td>0.223029</td>
      <td>0.364358</td>
      <td>0.377596</td>
      <td>0.060734</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Chinese Opera</td>
      <td>1VYHe8hOLSf1qOhG8gETrr</td>
      <td>Opera</td>
      <td>2.262882</td>
    </tr>
    <tr>
      <th>252</th>
      <td>0.420715</td>
      <td>0.239882</td>
      <td>0.454545</td>
      <td>0.578208</td>
      <td>0.0</td>
      <td>0.335306</td>
      <td>0.987952</td>
      <td>0.000003</td>
      <td>0.338662</td>
      <td>0.236229</td>
      <td>0.222376</td>
      <td>0.118649</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Chinese Opera</td>
      <td>4twPZLYy1JZZJqUqBPRjvl</td>
      <td>Opera</td>
      <td>2.284384</td>
    </tr>
    <tr>
      <th>253</th>
      <td>0.360939</td>
      <td>0.130865</td>
      <td>0.818182</td>
      <td>0.486289</td>
      <td>1.0</td>
      <td>0.075452</td>
      <td>0.990964</td>
      <td>0.536072</td>
      <td>0.215929</td>
      <td>0.197189</td>
      <td>0.159177</td>
      <td>0.092477</td>
      <td>0.6</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Chinese Opera</td>
      <td>1I04rLV5Vj5O4QlWQaAzjV</td>
      <td>Opera</td>
      <td>2.263037</td>
    </tr>
    <tr>
      <th>254</th>
      <td>0.365286</td>
      <td>0.232881</td>
      <td>0.818182</td>
      <td>0.644078</td>
      <td>1.0</td>
      <td>0.070696</td>
      <td>0.976908</td>
      <td>0.027856</td>
      <td>0.463424</td>
      <td>0.392386</td>
      <td>0.159098</td>
      <td>0.088060</td>
      <td>0.6</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Chinese Opera</td>
      <td>0k2F3AnVN23bJ83yK1z8pC</td>
      <td>Opera</td>
      <td>2.262923</td>
    </tr>
    <tr>
      <th>255</th>
      <td>0.267471</td>
      <td>0.285889</td>
      <td>0.636364</td>
      <td>0.604478</td>
      <td>0.0</td>
      <td>0.041002</td>
      <td>0.945783</td>
      <td>0.008547</td>
      <td>0.191585</td>
      <td>0.410405</td>
      <td>0.184124</td>
      <td>0.106136</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Chinese Opera</td>
      <td>3Q2HtQBZ13UofqnWgZqRMV</td>
      <td>Opera</td>
      <td>2.262434</td>
    </tr>
    <tr>
      <th>256</th>
      <td>0.418541</td>
      <td>0.610940</td>
      <td>0.818182</td>
      <td>0.721237</td>
      <td>1.0</td>
      <td>0.032971</td>
      <td>0.046687</td>
      <td>0.000002</td>
      <td>0.783949</td>
      <td>0.520516</td>
      <td>0.271097</td>
      <td>0.047594</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Chinese Opera</td>
      <td>5iTaxyknRVgSf20tpimL9f</td>
      <td>Opera</td>
      <td>2.262369</td>
    </tr>
    <tr>
      <th>257</th>
      <td>0.480491</td>
      <td>0.409909</td>
      <td>0.545455</td>
      <td>0.680374</td>
      <td>1.0</td>
      <td>0.029695</td>
      <td>0.921687</td>
      <td>0.288577</td>
      <td>0.067838</td>
      <td>0.480475</td>
      <td>0.747877</td>
      <td>0.055335</td>
      <td>0.6</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Chinese Opera</td>
      <td>2AcAQUb5OzmfNBwSVNtyQC</td>
      <td>Opera</td>
      <td>2.262350</td>
    </tr>
    <tr>
      <th>258</th>
      <td>0.322900</td>
      <td>0.125865</td>
      <td>0.000000</td>
      <td>0.549542</td>
      <td>1.0</td>
      <td>0.025785</td>
      <td>0.863454</td>
      <td>0.000000</td>
      <td>0.256502</td>
      <td>0.696694</td>
      <td>0.169436</td>
      <td>0.046149</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Chinese Opera</td>
      <td>6Kv6UbtdDGZKbyiJXczXmk</td>
      <td>Opera</td>
      <td>2.262335</td>
    </tr>
    <tr>
      <th>259</th>
      <td>0.575046</td>
      <td>0.629943</td>
      <td>0.818182</td>
      <td>0.753319</td>
      <td>0.0</td>
      <td>0.015217</td>
      <td>0.173695</td>
      <td>0.000002</td>
      <td>0.054956</td>
      <td>0.476471</td>
      <td>0.261806</td>
      <td>0.039371</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Chinese Opera</td>
      <td>13sPHoaDXjd3DkHllyXPp3</td>
      <td>Opera</td>
      <td>2.262326</td>
    </tr>
    <tr>
      <th>260</th>
      <td>0.710901</td>
      <td>0.146868</td>
      <td>0.454545</td>
      <td>0.594081</td>
      <td>1.0</td>
      <td>0.011730</td>
      <td>0.617470</td>
      <td>0.003758</td>
      <td>0.442123</td>
      <td>0.276269</td>
      <td>0.378792</td>
      <td>0.085755</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Chinese Opera</td>
      <td>1WCK43jXobM68LNFVS5OMQ</td>
      <td>Opera</td>
      <td>2.262334</td>
    </tr>
    <tr>
      <th>261</th>
      <td>0.402239</td>
      <td>0.415909</td>
      <td>0.818182</td>
      <td>0.637581</td>
      <td>1.0</td>
      <td>0.017119</td>
      <td>0.224900</td>
      <td>0.646293</td>
      <td>0.205786</td>
      <td>0.336330</td>
      <td>0.250950</td>
      <td>0.072166</td>
      <td>0.6</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Chinese Opera</td>
      <td>1z3NH3L9hwrsynANxGZMKk</td>
      <td>Opera</td>
      <td>2.262324</td>
    </tr>
    <tr>
      <th>262</th>
      <td>0.451147</td>
      <td>0.379904</td>
      <td>0.909091</td>
      <td>0.749513</td>
      <td>1.0</td>
      <td>0.008031</td>
      <td>0.824297</td>
      <td>0.000000</td>
      <td>0.127683</td>
      <td>0.263256</td>
      <td>0.514705</td>
      <td>0.047236</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Chinese Opera</td>
      <td>1rLGimq91qSlZf6Udy5ULz</td>
      <td>Opera</td>
      <td>2.262348</td>
    </tr>
    <tr>
      <th>263</th>
      <td>0.584828</td>
      <td>0.387905</td>
      <td>0.000000</td>
      <td>0.734623</td>
      <td>1.0</td>
      <td>0.005389</td>
      <td>0.687751</td>
      <td>0.000003</td>
      <td>0.043798</td>
      <td>0.158150</td>
      <td>0.444463</td>
      <td>0.046389</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Chinese Opera</td>
      <td>1mf4MpBNZmv7DHoYEWUWiF</td>
      <td>Opera</td>
      <td>2.262362</td>
    </tr>
    <tr>
      <th>264</th>
      <td>0.478318</td>
      <td>0.256885</td>
      <td>0.818182</td>
      <td>0.492230</td>
      <td>0.0</td>
      <td>0.041002</td>
      <td>0.001596</td>
      <td>0.222445</td>
      <td>0.185499</td>
      <td>0.039630</td>
      <td>0.388620</td>
      <td>0.046062</td>
      <td>0.6</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Chinese Opera</td>
      <td>3gkZMABXO4w4hDkDOU01nb</td>
      <td>Opera</td>
      <td>2.262434</td>
    </tr>
    <tr>
      <th>265</th>
      <td>0.466362</td>
      <td>0.495922</td>
      <td>1.000000</td>
      <td>0.779143</td>
      <td>0.0</td>
      <td>0.011836</td>
      <td>0.798193</td>
      <td>0.005661</td>
      <td>0.300118</td>
      <td>0.684682</td>
      <td>0.393826</td>
      <td>0.027052</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Chinese Opera</td>
      <td>4jKD3GwpcphT6e4ktm9qJ6</td>
      <td>Opera</td>
      <td>2.262333</td>
    </tr>
    <tr>
      <th>266</th>
      <td>0.270731</td>
      <td>0.338898</td>
      <td>0.545455</td>
      <td>0.721813</td>
      <td>0.0</td>
      <td>0.007186</td>
      <td>0.769076</td>
      <td>0.000006</td>
      <td>0.107396</td>
      <td>0.215207</td>
      <td>0.513817</td>
      <td>0.046818</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Chinese Opera</td>
      <td>4QUmmOQQtkUs8mHFz2ofY6</td>
      <td>Opera</td>
      <td>2.262352</td>
    </tr>
    <tr>
      <th>267</th>
      <td>0.545702</td>
      <td>0.710955</td>
      <td>0.181818</td>
      <td>0.790673</td>
      <td>0.0</td>
      <td>0.016063</td>
      <td>0.120482</td>
      <td>0.000000</td>
      <td>0.290989</td>
      <td>0.497492</td>
      <td>0.475504</td>
      <td>0.045080</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>Chinese Opera</td>
      <td>0IJmZ5KZvXn6wFlTaZwMIJ</td>
      <td>Opera</td>
      <td>2.262325</td>
    </tr>
  </tbody>
</table>
<p>268 rows × 20 columns</p>
</div>





```python
genre_data = genre_data.sort_values(by=['distance'], ascending=True)
```




```python
playlist = genre_data.iloc[:N,:]
print(playlist.shape)
playlist
```


    (100, 20)





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
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>duration_ms</th>
      <th>time_signature</th>
      <th>popularity</th>
      <th>artist_popularity</th>
      <th>explicit</th>
      <th>genre</th>
      <th>id</th>
      <th>category</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.108793</td>
      <td>0.095660</td>
      <td>0.454545</td>
      <td>0.554277</td>
      <td>1.0</td>
      <td>0.018599</td>
      <td>0.879518</td>
      <td>0.897796</td>
      <td>0.385321</td>
      <td>0.037228</td>
      <td>0.493511</td>
      <td>0.041012</td>
      <td>0.6</td>
      <td>0.500000</td>
      <td>0.677419</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>0BqbWiOxccb0qQshm3Whjt</td>
      <td>Opera</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.143571</td>
      <td>0.301892</td>
      <td>0.454545</td>
      <td>0.598426</td>
      <td>1.0</td>
      <td>0.050724</td>
      <td>0.995984</td>
      <td>0.019439</td>
      <td>0.677446</td>
      <td>0.127118</td>
      <td>0.206239</td>
      <td>0.063333</td>
      <td>0.8</td>
      <td>0.404255</td>
      <td>0.623656</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>3Vb0EDj3nLxxK3IOcp27j3</td>
      <td>Opera</td>
      <td>0.318749</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.187045</td>
      <td>0.321895</td>
      <td>0.181818</td>
      <td>0.687875</td>
      <td>0.0</td>
      <td>0.009722</td>
      <td>0.678715</td>
      <td>0.834669</td>
      <td>0.114497</td>
      <td>0.253246</td>
      <td>0.272107</td>
      <td>0.033346</td>
      <td>0.8</td>
      <td>0.468085</td>
      <td>0.548387</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>4WYZuhsa8idA4kPDPsNY6B</td>
      <td>Opera</td>
      <td>0.336607</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.290295</td>
      <td>0.152869</td>
      <td>0.909091</td>
      <td>0.533910</td>
      <td>1.0</td>
      <td>0.036035</td>
      <td>0.948795</td>
      <td>0.000011</td>
      <td>0.734248</td>
      <td>0.278271</td>
      <td>0.211123</td>
      <td>0.034943</td>
      <td>0.8</td>
      <td>0.361702</td>
      <td>0.634409</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>3iVh3cgWwbYNFGZz5W8g12</td>
      <td>Opera</td>
      <td>0.428955</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.171829</td>
      <td>0.017848</td>
      <td>1.000000</td>
      <td>0.252697</td>
      <td>1.0</td>
      <td>0.022086</td>
      <td>0.873494</td>
      <td>0.140281</td>
      <td>0.183471</td>
      <td>0.038929</td>
      <td>0.424914</td>
      <td>0.043075</td>
      <td>0.8</td>
      <td>0.351064</td>
      <td>0.684588</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>1tf5l5Cur6hdwoStk1f1WY</td>
      <td>Opera</td>
      <td>0.447181</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.269645</td>
      <td>0.156869</td>
      <td>0.090909</td>
      <td>0.611552</td>
      <td>1.0</td>
      <td>0.023777</td>
      <td>0.964859</td>
      <td>0.003357</td>
      <td>0.092182</td>
      <td>0.242235</td>
      <td>0.485370</td>
      <td>0.059259</td>
      <td>0.2</td>
      <td>0.351064</td>
      <td>0.580645</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>5M9QQBxwH8ybk3xOZEJei4</td>
      <td>Opera</td>
      <td>0.508131</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.081622</td>
      <td>0.039751</td>
      <td>0.363636</td>
      <td>0.493196</td>
      <td>1.0</td>
      <td>0.018493</td>
      <td>0.992972</td>
      <td>0.952906</td>
      <td>0.095225</td>
      <td>0.121112</td>
      <td>0.196486</td>
      <td>0.049150</td>
      <td>0.8</td>
      <td>0.329787</td>
      <td>0.691756</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>1KeasaJ8MshakeNvRqbDzx</td>
      <td>Opera</td>
      <td>0.511895</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.092490</td>
      <td>0.244883</td>
      <td>0.272727</td>
      <td>0.648646</td>
      <td>1.0</td>
      <td>0.012787</td>
      <td>0.984940</td>
      <td>0.484970</td>
      <td>0.099282</td>
      <td>0.149141</td>
      <td>0.202996</td>
      <td>0.044947</td>
      <td>0.8</td>
      <td>0.329787</td>
      <td>0.655914</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>1wgUTAbPQkblN0nykDueMI</td>
      <td>Opera</td>
      <td>0.513494</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.385936</td>
      <td>0.168871</td>
      <td>0.090909</td>
      <td>0.625791</td>
      <td>0.0</td>
      <td>0.028849</td>
      <td>0.995984</td>
      <td>0.000391</td>
      <td>0.073214</td>
      <td>0.311304</td>
      <td>0.281935</td>
      <td>0.071341</td>
      <td>0.8</td>
      <td>0.382979</td>
      <td>0.516129</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>6YJiRaVLOrCTe9f6t0P2nx</td>
      <td>Opera</td>
      <td>0.534735</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.110966</td>
      <td>0.071656</td>
      <td>0.090909</td>
      <td>0.452351</td>
      <td>0.0</td>
      <td>0.028321</td>
      <td>0.974900</td>
      <td>0.866733</td>
      <td>0.106382</td>
      <td>0.039930</td>
      <td>0.265849</td>
      <td>0.062893</td>
      <td>0.8</td>
      <td>0.393617</td>
      <td>0.500000</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>3KpKYUwzIEMYDv77Qygun4</td>
      <td>Opera</td>
      <td>0.546521</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.097924</td>
      <td>0.393906</td>
      <td>0.363636</td>
      <td>0.710636</td>
      <td>1.0</td>
      <td>0.013104</td>
      <td>0.917671</td>
      <td>0.301603</td>
      <td>0.096239</td>
      <td>0.168160</td>
      <td>0.269246</td>
      <td>0.058589</td>
      <td>0.8</td>
      <td>0.404255</td>
      <td>0.483871</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>1zZAo2svewOprAfphYjGL5</td>
      <td>Opera</td>
      <td>0.562730</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.210955</td>
      <td>0.158870</td>
      <td>0.090909</td>
      <td>0.661642</td>
      <td>1.0</td>
      <td>0.019338</td>
      <td>0.985944</td>
      <td>0.000000</td>
      <td>0.101311</td>
      <td>0.065957</td>
      <td>0.281379</td>
      <td>0.055366</td>
      <td>0.8</td>
      <td>0.351064</td>
      <td>0.489247</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>3asyS9IMGkZb5HrbEnrwpH</td>
      <td>Opera</td>
      <td>0.648801</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.230518</td>
      <td>0.156869</td>
      <td>0.818182</td>
      <td>0.602974</td>
      <td>1.0</td>
      <td>0.018916</td>
      <td>0.964859</td>
      <td>0.024048</td>
      <td>0.105368</td>
      <td>0.070461</td>
      <td>0.468471</td>
      <td>0.051822</td>
      <td>0.8</td>
      <td>0.276596</td>
      <td>0.666667</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>1opcgxc7LjcutXrx4hGBN2</td>
      <td>Opera</td>
      <td>0.670752</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.264210</td>
      <td>0.416910</td>
      <td>0.090909</td>
      <td>0.718137</td>
      <td>1.0</td>
      <td>0.012047</td>
      <td>0.815261</td>
      <td>0.000001</td>
      <td>0.085893</td>
      <td>0.155147</td>
      <td>0.216838</td>
      <td>0.071128</td>
      <td>0.8</td>
      <td>0.287234</td>
      <td>0.580645</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>2joixSFXKk1uYN7czBQzBY</td>
      <td>Opera</td>
      <td>0.682642</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.089229</td>
      <td>0.031950</td>
      <td>0.181818</td>
      <td>0.338674</td>
      <td>0.0</td>
      <td>0.026736</td>
      <td>0.766064</td>
      <td>0.569138</td>
      <td>0.062665</td>
      <td>0.037127</td>
      <td>0.254829</td>
      <td>0.084353</td>
      <td>0.6</td>
      <td>0.340426</td>
      <td>0.465054</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>4E0bxKybeThhGkYoie27Nf</td>
      <td>Opera</td>
      <td>0.714921</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.085969</td>
      <td>0.099860</td>
      <td>0.454545</td>
      <td>0.496742</td>
      <td>1.0</td>
      <td>0.026102</td>
      <td>0.992972</td>
      <td>0.019138</td>
      <td>0.175356</td>
      <td>0.065056</td>
      <td>0.670060</td>
      <td>0.084589</td>
      <td>1.0</td>
      <td>0.372340</td>
      <td>0.430108</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>2wGCbQEmMAbt5X30s7pnSc</td>
      <td>Opera</td>
      <td>0.727323</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.244647</td>
      <td>0.085058</td>
      <td>0.454545</td>
      <td>0.516737</td>
      <td>1.0</td>
      <td>0.018599</td>
      <td>0.969880</td>
      <td>0.023246</td>
      <td>0.309247</td>
      <td>0.121112</td>
      <td>0.378998</td>
      <td>0.057035</td>
      <td>0.6</td>
      <td>0.361702</td>
      <td>0.434409</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>7LOytcUG9JFhM8AEmN15Xr</td>
      <td>Opera</td>
      <td>0.735680</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.380502</td>
      <td>0.498922</td>
      <td>0.636364</td>
      <td>0.734753</td>
      <td>0.0</td>
      <td>0.013421</td>
      <td>0.789157</td>
      <td>0.000005</td>
      <td>0.090660</td>
      <td>0.376370</td>
      <td>0.364192</td>
      <td>0.049682</td>
      <td>0.8</td>
      <td>0.276596</td>
      <td>0.548387</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>2IzAde95HP2wDt6QS6H7Sw</td>
      <td>Opera</td>
      <td>0.743821</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.200087</td>
      <td>0.194875</td>
      <td>0.727273</td>
      <td>0.631361</td>
      <td>1.0</td>
      <td>0.016697</td>
      <td>0.977912</td>
      <td>0.000117</td>
      <td>0.096239</td>
      <td>0.146138</td>
      <td>0.275799</td>
      <td>0.032806</td>
      <td>1.0</td>
      <td>0.329787</td>
      <td>0.437276</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>23XpALLxFdVMMYe5z5BGJc</td>
      <td>Opera</td>
      <td>0.788153</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.495707</td>
      <td>0.012347</td>
      <td>1.000000</td>
      <td>0.543026</td>
      <td>0.0</td>
      <td>0.014477</td>
      <td>0.996988</td>
      <td>0.188377</td>
      <td>0.055057</td>
      <td>0.077168</td>
      <td>0.262152</td>
      <td>0.036145</td>
      <td>0.8</td>
      <td>0.255319</td>
      <td>0.548387</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>36JNf7QIvLWb6XI8JePDxL</td>
      <td>Opera</td>
      <td>0.801807</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.202261</td>
      <td>0.243883</td>
      <td>0.090909</td>
      <td>0.664983</td>
      <td>1.0</td>
      <td>0.016591</td>
      <td>0.994980</td>
      <td>0.003036</td>
      <td>0.061853</td>
      <td>0.082774</td>
      <td>0.261946</td>
      <td>0.035542</td>
      <td>0.6</td>
      <td>0.319149</td>
      <td>0.440860</td>
      <td>0.0</td>
      <td>Deep Opera</td>
      <td>0vFIcgSTRg0fTlESbPPCsJ</td>
      <td>Opera</td>
      <td>0.802571</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.214216</td>
      <td>0.187874</td>
      <td>0.000000</td>
      <td>0.603921</td>
      <td>1.0</td>
      <td>0.019973</td>
      <td>0.992972</td>
      <td>0.010621</td>
      <td>0.072707</td>
      <td>0.046237</td>
      <td>0.261189</td>
      <td>0.045066</td>
      <td>0.8</td>
      <td>0.276596</td>
      <td>0.483871</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>7id0eYuQAxnwgd03jCQzr1</td>
      <td>Opera</td>
      <td>0.826631</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0.128356</td>
      <td>0.201876</td>
      <td>0.363636</td>
      <td>0.559419</td>
      <td>0.0</td>
      <td>0.021663</td>
      <td>0.912651</td>
      <td>0.918838</td>
      <td>0.084270</td>
      <td>0.036827</td>
      <td>0.215109</td>
      <td>0.034779</td>
      <td>0.8</td>
      <td>0.223404</td>
      <td>0.629032</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>0Tztnfa3n1OPE1KCkXqeJP</td>
      <td>Opera</td>
      <td>0.838564</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.266384</td>
      <td>0.216879</td>
      <td>0.363636</td>
      <td>0.612907</td>
      <td>1.0</td>
      <td>0.016697</td>
      <td>0.954819</td>
      <td>0.000000</td>
      <td>0.274760</td>
      <td>0.398392</td>
      <td>0.253777</td>
      <td>0.048151</td>
      <td>0.8</td>
      <td>0.265957</td>
      <td>0.483871</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>3pTVlPsOH9Gf8JgpMXT8BG</td>
      <td>Opera</td>
      <td>0.852712</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.093577</td>
      <td>0.133866</td>
      <td>0.636364</td>
      <td>0.575311</td>
      <td>0.0</td>
      <td>0.022720</td>
      <td>0.963855</td>
      <td>0.000000</td>
      <td>0.087617</td>
      <td>0.102093</td>
      <td>0.186111</td>
      <td>0.045105</td>
      <td>0.8</td>
      <td>0.372340</td>
      <td>0.370968</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>0YRppTZQqA5h3zTat2EpvC</td>
      <td>Opera</td>
      <td>0.856530</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.277252</td>
      <td>0.217879</td>
      <td>0.000000</td>
      <td>0.607337</td>
      <td>0.0</td>
      <td>0.029695</td>
      <td>0.998996</td>
      <td>0.229459</td>
      <td>0.097253</td>
      <td>0.036627</td>
      <td>0.256245</td>
      <td>0.054719</td>
      <td>0.8</td>
      <td>0.255319</td>
      <td>0.494624</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>5e5Umo93FhQ4BwperG6TjN</td>
      <td>Opera</td>
      <td>0.864743</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.351157</td>
      <td>0.353900</td>
      <td>0.363636</td>
      <td>0.737816</td>
      <td>0.0</td>
      <td>0.012681</td>
      <td>0.917671</td>
      <td>0.000000</td>
      <td>0.135797</td>
      <td>0.277270</td>
      <td>0.396887</td>
      <td>0.040148</td>
      <td>0.8</td>
      <td>0.308511</td>
      <td>0.408602</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>6h1EIRHC2cjricHyk9UVQ7</td>
      <td>Opera</td>
      <td>0.884133</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.260950</td>
      <td>0.166871</td>
      <td>0.636364</td>
      <td>0.526224</td>
      <td>1.0</td>
      <td>0.027370</td>
      <td>0.990964</td>
      <td>0.003327</td>
      <td>0.087617</td>
      <td>0.064655</td>
      <td>0.206491</td>
      <td>0.036145</td>
      <td>1.0</td>
      <td>0.255319</td>
      <td>0.473118</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>2ZDqxewFaSQxhFGBNTtAxa</td>
      <td>Opera</td>
      <td>0.894295</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.292468</td>
      <td>0.068956</td>
      <td>0.818182</td>
      <td>0.512133</td>
      <td>0.0</td>
      <td>0.036458</td>
      <td>0.993976</td>
      <td>0.872745</td>
      <td>0.316347</td>
      <td>0.083174</td>
      <td>0.179147</td>
      <td>0.034351</td>
      <td>0.6</td>
      <td>0.255319</td>
      <td>0.473118</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>4jWsGe1IZrtrWJvu7KE0UT</td>
      <td>Opera</td>
      <td>0.894430</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.133790</td>
      <td>0.256885</td>
      <td>0.454545</td>
      <td>0.713625</td>
      <td>0.0</td>
      <td>0.021135</td>
      <td>0.986948</td>
      <td>0.000012</td>
      <td>0.650059</td>
      <td>0.103094</td>
      <td>0.135105</td>
      <td>0.061711</td>
      <td>0.6</td>
      <td>0.287234</td>
      <td>0.419355</td>
      <td>0.0</td>
      <td>Deep Opera</td>
      <td>3GJdkrB0Vy42ZQRrR2nY2o</td>
      <td>Opera</td>
      <td>0.907559</td>
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
    </tr>
    <tr>
      <th>51</th>
      <td>0.023150</td>
      <td>0.139867</td>
      <td>0.454545</td>
      <td>0.540260</td>
      <td>1.0</td>
      <td>0.015851</td>
      <td>0.966867</td>
      <td>0.298597</td>
      <td>0.094210</td>
      <td>0.036927</td>
      <td>0.216483</td>
      <td>0.065547</td>
      <td>0.8</td>
      <td>0.148936</td>
      <td>0.298387</td>
      <td>0.0</td>
      <td>Deep Opera</td>
      <td>4WFQRyuHi37pQzop1z4eiG</td>
      <td>Opera</td>
      <td>1.416732</td>
    </tr>
    <tr>
      <th>91</th>
      <td>0.353331</td>
      <td>0.352900</td>
      <td>0.000000</td>
      <td>0.645972</td>
      <td>1.0</td>
      <td>0.051252</td>
      <td>0.969880</td>
      <td>0.250501</td>
      <td>0.127683</td>
      <td>0.601598</td>
      <td>0.163837</td>
      <td>0.054499</td>
      <td>0.8</td>
      <td>0.074468</td>
      <td>0.430108</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>7G2NbcqAdQyyr7lfIaQBKT</td>
      <td>Opera</td>
      <td>1.418814</td>
    </tr>
    <tr>
      <th>115</th>
      <td>0.240300</td>
      <td>0.144867</td>
      <td>0.454545</td>
      <td>0.567755</td>
      <td>1.0</td>
      <td>0.024305</td>
      <td>0.980924</td>
      <td>0.012725</td>
      <td>0.118554</td>
      <td>0.037828</td>
      <td>0.235584</td>
      <td>0.078486</td>
      <td>0.8</td>
      <td>0.053191</td>
      <td>0.491039</td>
      <td>0.0</td>
      <td>Deep Opera</td>
      <td>4IPJCOyRlKGRaNTtjbGJTU</td>
      <td>Opera</td>
      <td>1.419113</td>
    </tr>
    <tr>
      <th>124</th>
      <td>0.220737</td>
      <td>0.167871</td>
      <td>0.090909</td>
      <td>0.555279</td>
      <td>1.0</td>
      <td>0.029061</td>
      <td>0.992972</td>
      <td>0.184369</td>
      <td>0.143912</td>
      <td>0.068159</td>
      <td>0.255581</td>
      <td>0.059293</td>
      <td>0.8</td>
      <td>0.042553</td>
      <td>0.526882</td>
      <td>0.0</td>
      <td>Deep Opera</td>
      <td>2iyMzM6aWkodlXy0elbviq</td>
      <td>Opera</td>
      <td>1.423047</td>
    </tr>
    <tr>
      <th>116</th>
      <td>0.321813</td>
      <td>0.215878</td>
      <td>0.909091</td>
      <td>0.594248</td>
      <td>1.0</td>
      <td>0.058227</td>
      <td>0.990964</td>
      <td>0.000496</td>
      <td>0.379235</td>
      <td>0.173165</td>
      <td>0.268260</td>
      <td>0.034385</td>
      <td>0.8</td>
      <td>0.053191</td>
      <td>0.483871</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>236FbV55sF5WJ0IIv8VYgW</td>
      <td>Opera</td>
      <td>1.425637</td>
    </tr>
    <tr>
      <th>104</th>
      <td>0.137050</td>
      <td>0.138867</td>
      <td>0.909091</td>
      <td>0.608117</td>
      <td>0.0</td>
      <td>0.021875</td>
      <td>0.980924</td>
      <td>0.001613</td>
      <td>0.095225</td>
      <td>0.039930</td>
      <td>0.193719</td>
      <td>0.060042</td>
      <td>0.6</td>
      <td>0.063830</td>
      <td>0.448029</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>2Bt437f3OT85JMKL2kzsJo</td>
      <td>Opera</td>
      <td>1.428666</td>
    </tr>
    <tr>
      <th>105</th>
      <td>0.081622</td>
      <td>0.007236</td>
      <td>0.818182</td>
      <td>0.326792</td>
      <td>1.0</td>
      <td>0.033922</td>
      <td>0.921687</td>
      <td>0.000004</td>
      <td>0.060230</td>
      <td>0.085677</td>
      <td>0.174273</td>
      <td>0.038676</td>
      <td>1.0</td>
      <td>0.063830</td>
      <td>0.440860</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>5L3MLkgbqawuWAX6sz7Cbz</td>
      <td>Opera</td>
      <td>1.436031</td>
    </tr>
    <tr>
      <th>148</th>
      <td>0.366373</td>
      <td>0.474919</td>
      <td>0.000000</td>
      <td>0.656313</td>
      <td>1.0</td>
      <td>0.057698</td>
      <td>0.981928</td>
      <td>0.714429</td>
      <td>0.168256</td>
      <td>0.439434</td>
      <td>0.512046</td>
      <td>0.052047</td>
      <td>0.8</td>
      <td>0.021277</td>
      <td>0.634409</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>1sg449pv2XzXN0zb161Z0r</td>
      <td>Opera</td>
      <td>1.440721</td>
    </tr>
    <tr>
      <th>117</th>
      <td>0.262037</td>
      <td>0.149868</td>
      <td>0.000000</td>
      <td>0.562631</td>
      <td>1.0</td>
      <td>0.035612</td>
      <td>0.977912</td>
      <td>0.000522</td>
      <td>0.588185</td>
      <td>0.152144</td>
      <td>0.221563</td>
      <td>0.047109</td>
      <td>0.8</td>
      <td>0.053191</td>
      <td>0.462366</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>7tsjctC6tYsFQTShw8CE0N</td>
      <td>Opera</td>
      <td>1.444327</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0.604391</td>
      <td>0.678950</td>
      <td>0.636364</td>
      <td>0.728849</td>
      <td>1.0</td>
      <td>0.007926</td>
      <td>0.324297</td>
      <td>0.000000</td>
      <td>0.086907</td>
      <td>0.717715</td>
      <td>0.289697</td>
      <td>0.051107</td>
      <td>0.8</td>
      <td>0.234043</td>
      <td>0.182796</td>
      <td>0.0</td>
      <td>Chinese Opera</td>
      <td>0918h8ImlCAXrZBethhzm7</td>
      <td>Opera</td>
      <td>1.471663</td>
    </tr>
    <tr>
      <th>82</th>
      <td>0.262037</td>
      <td>0.263886</td>
      <td>0.000000</td>
      <td>0.699200</td>
      <td>0.0</td>
      <td>0.012470</td>
      <td>0.972892</td>
      <td>0.005641</td>
      <td>0.259545</td>
      <td>0.216208</td>
      <td>0.215002</td>
      <td>0.039174</td>
      <td>1.0</td>
      <td>0.095745</td>
      <td>0.333333</td>
      <td>0.0</td>
      <td>Deep Opera</td>
      <td>7dOMWPCU2HL8tFmUpu7td3</td>
      <td>Opera</td>
      <td>1.486879</td>
    </tr>
    <tr>
      <th>106</th>
      <td>0.185958</td>
      <td>0.134866</td>
      <td>1.000000</td>
      <td>0.551083</td>
      <td>0.0</td>
      <td>0.027370</td>
      <td>0.951807</td>
      <td>0.071242</td>
      <td>0.092080</td>
      <td>0.193185</td>
      <td>0.213291</td>
      <td>0.042411</td>
      <td>0.6</td>
      <td>0.063830</td>
      <td>0.387097</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>5F78kXGaBlklxzmfLTpPtG</td>
      <td>Opera</td>
      <td>1.496353</td>
    </tr>
    <tr>
      <th>107</th>
      <td>0.244647</td>
      <td>0.136866</td>
      <td>0.454545</td>
      <td>0.503462</td>
      <td>1.0</td>
      <td>0.021663</td>
      <td>0.973896</td>
      <td>0.336673</td>
      <td>0.113482</td>
      <td>0.146138</td>
      <td>0.443323</td>
      <td>0.062164</td>
      <td>0.8</td>
      <td>0.063830</td>
      <td>0.379928</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>3pgUIjVhnGyztB090mMWvy</td>
      <td>Opera</td>
      <td>1.505105</td>
    </tr>
    <tr>
      <th>108</th>
      <td>0.266384</td>
      <td>0.137866</td>
      <td>0.090909</td>
      <td>0.418915</td>
      <td>1.0</td>
      <td>0.046497</td>
      <td>0.994980</td>
      <td>0.001062</td>
      <td>0.916826</td>
      <td>0.187179</td>
      <td>0.361968</td>
      <td>0.040249</td>
      <td>0.8</td>
      <td>0.063830</td>
      <td>0.376344</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>6VqFDmefF4X44d40kZAIWO</td>
      <td>Opera</td>
      <td>1.509807</td>
    </tr>
    <tr>
      <th>199</th>
      <td>0.131616</td>
      <td>0.224880</td>
      <td>0.636364</td>
      <td>0.642612</td>
      <td>1.0</td>
      <td>0.020184</td>
      <td>0.987952</td>
      <td>0.044389</td>
      <td>0.118554</td>
      <td>0.041732</td>
      <td>0.204753</td>
      <td>0.043790</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.572581</td>
      <td>0.0</td>
      <td>Deep Opera</td>
      <td>0uTPw6kSHdIxKOqyVjD4Up</td>
      <td>Opera</td>
      <td>1.522727</td>
    </tr>
    <tr>
      <th>136</th>
      <td>0.263124</td>
      <td>0.062255</td>
      <td>0.818182</td>
      <td>0.440692</td>
      <td>0.0</td>
      <td>0.049878</td>
      <td>0.993976</td>
      <td>0.140281</td>
      <td>0.087921</td>
      <td>0.037127</td>
      <td>0.218292</td>
      <td>0.048905</td>
      <td>0.6</td>
      <td>0.031915</td>
      <td>0.440860</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>6ufuSLtSMOGQSUTZPEfPnl</td>
      <td>Opera</td>
      <td>1.524028</td>
    </tr>
    <tr>
      <th>109</th>
      <td>0.308771</td>
      <td>0.062855</td>
      <td>0.000000</td>
      <td>0.512578</td>
      <td>1.0</td>
      <td>0.024622</td>
      <td>0.988956</td>
      <td>0.000075</td>
      <td>0.097253</td>
      <td>0.124115</td>
      <td>0.244150</td>
      <td>0.029011</td>
      <td>0.8</td>
      <td>0.063830</td>
      <td>0.363441</td>
      <td>0.0</td>
      <td>Deep Opera</td>
      <td>266k3hu9JO219RWpmzypBX</td>
      <td>Opera</td>
      <td>1.525902</td>
    </tr>
    <tr>
      <th>167</th>
      <td>0.463102</td>
      <td>0.323895</td>
      <td>0.545455</td>
      <td>0.620742</td>
      <td>0.0</td>
      <td>0.016168</td>
      <td>0.969880</td>
      <td>0.297595</td>
      <td>0.112468</td>
      <td>0.549545</td>
      <td>0.268363</td>
      <td>0.033704</td>
      <td>0.8</td>
      <td>0.010638</td>
      <td>0.508961</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>75G76BDnVlsr11Zs3KE87C</td>
      <td>Opera</td>
      <td>1.527300</td>
    </tr>
    <tr>
      <th>68</th>
      <td>0.350071</td>
      <td>0.355900</td>
      <td>0.545455</td>
      <td>0.739005</td>
      <td>0.0</td>
      <td>0.010990</td>
      <td>0.786145</td>
      <td>0.000000</td>
      <td>0.090457</td>
      <td>0.217209</td>
      <td>0.442332</td>
      <td>0.052297</td>
      <td>0.8</td>
      <td>0.117021</td>
      <td>0.274194</td>
      <td>0.0</td>
      <td>Chinese Opera</td>
      <td>52M2U3oDinpmibqLLWld9W</td>
      <td>Opera</td>
      <td>1.528498</td>
    </tr>
    <tr>
      <th>200</th>
      <td>0.279426</td>
      <td>0.135866</td>
      <td>0.181818</td>
      <td>0.532388</td>
      <td>1.0</td>
      <td>0.017648</td>
      <td>0.955823</td>
      <td>0.174349</td>
      <td>0.235201</td>
      <td>0.208200</td>
      <td>0.318215</td>
      <td>0.047793</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.551971</td>
      <td>0.0</td>
      <td>Deep Opera</td>
      <td>0ysq1Fq8Cc9vvOseEwrOYL</td>
      <td>Opera</td>
      <td>1.532435</td>
    </tr>
    <tr>
      <th>83</th>
      <td>0.421802</td>
      <td>0.320895</td>
      <td>0.909091</td>
      <td>0.741697</td>
      <td>1.0</td>
      <td>0.048927</td>
      <td>0.982932</td>
      <td>0.000002</td>
      <td>0.137826</td>
      <td>0.628625</td>
      <td>0.219334</td>
      <td>0.028243</td>
      <td>0.6</td>
      <td>0.095745</td>
      <td>0.301075</td>
      <td>0.0</td>
      <td>Deep Opera</td>
      <td>70kf2XVz63ObwSm4wr3JlP</td>
      <td>Opera</td>
      <td>1.535233</td>
    </tr>
    <tr>
      <th>149</th>
      <td>0.417455</td>
      <td>0.260885</td>
      <td>0.454545</td>
      <td>0.605611</td>
      <td>1.0</td>
      <td>0.079573</td>
      <td>0.972892</td>
      <td>0.000096</td>
      <td>0.382278</td>
      <td>0.291284</td>
      <td>0.301507</td>
      <td>0.067654</td>
      <td>0.8</td>
      <td>0.021277</td>
      <td>0.460215</td>
      <td>0.0</td>
      <td>Deep Opera</td>
      <td>41YrHMqmlAcREjZachVwUx</td>
      <td>Opera</td>
      <td>1.536608</td>
    </tr>
    <tr>
      <th>34</th>
      <td>0.628301</td>
      <td>0.538929</td>
      <td>0.454545</td>
      <td>0.745967</td>
      <td>1.0</td>
      <td>0.009299</td>
      <td>0.071988</td>
      <td>0.000000</td>
      <td>0.106382</td>
      <td>0.219211</td>
      <td>0.401865</td>
      <td>0.046769</td>
      <td>0.8</td>
      <td>0.234043</td>
      <td>0.150538</td>
      <td>0.0</td>
      <td>Chinese Opera</td>
      <td>05f0Y0tmUp0MImonwjOFc7</td>
      <td>Opera</td>
      <td>1.540037</td>
    </tr>
    <tr>
      <th>150</th>
      <td>0.147919</td>
      <td>0.286889</td>
      <td>0.727273</td>
      <td>0.636429</td>
      <td>0.0</td>
      <td>0.042270</td>
      <td>0.970884</td>
      <td>0.122244</td>
      <td>0.395464</td>
      <td>0.114105</td>
      <td>0.136338</td>
      <td>0.040336</td>
      <td>0.6</td>
      <td>0.021277</td>
      <td>0.451613</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>47ie2HlKXq3y9z1JFKzBRG</td>
      <td>Opera</td>
      <td>1.543316</td>
    </tr>
    <tr>
      <th>118</th>
      <td>0.398978</td>
      <td>0.112862</td>
      <td>0.454545</td>
      <td>0.426991</td>
      <td>1.0</td>
      <td>0.156716</td>
      <td>0.986948</td>
      <td>0.000062</td>
      <td>0.676431</td>
      <td>0.234227</td>
      <td>0.333773</td>
      <td>0.044846</td>
      <td>0.8</td>
      <td>0.053191</td>
      <td>0.376344</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>6R1y237aJvJxFXrtAokXFe</td>
      <td>Opera</td>
      <td>1.543488</td>
    </tr>
    <tr>
      <th>73</th>
      <td>0.291381</td>
      <td>0.267887</td>
      <td>0.818182</td>
      <td>0.635148</td>
      <td>1.0</td>
      <td>0.017753</td>
      <td>0.959839</td>
      <td>0.001453</td>
      <td>0.146955</td>
      <td>0.494489</td>
      <td>0.345073</td>
      <td>0.034816</td>
      <td>0.8</td>
      <td>0.106383</td>
      <td>0.275986</td>
      <td>0.0</td>
      <td>Deep Opera</td>
      <td>1XawHjwheJBL5Pi0sSSYsr</td>
      <td>Opera</td>
      <td>1.549707</td>
    </tr>
    <tr>
      <th>125</th>
      <td>0.440278</td>
      <td>0.401907</td>
      <td>0.000000</td>
      <td>0.722221</td>
      <td>1.0</td>
      <td>0.046814</td>
      <td>0.979920</td>
      <td>0.013727</td>
      <td>0.114497</td>
      <td>0.377371</td>
      <td>0.484800</td>
      <td>0.056061</td>
      <td>0.8</td>
      <td>0.042553</td>
      <td>0.387097</td>
      <td>0.0</td>
      <td>Deep Opera</td>
      <td>2sXZ9a5laKwvKFhOgcHsIu</td>
      <td>Opera</td>
      <td>1.552710</td>
    </tr>
    <tr>
      <th>168</th>
      <td>0.500054</td>
      <td>0.225880</td>
      <td>0.454545</td>
      <td>0.529714</td>
      <td>1.0</td>
      <td>0.018387</td>
      <td>0.981928</td>
      <td>0.000277</td>
      <td>0.168256</td>
      <td>0.535531</td>
      <td>0.331614</td>
      <td>0.035247</td>
      <td>0.8</td>
      <td>0.010638</td>
      <td>0.451613</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>0QxFF8atLRkQaetxW4PKq3</td>
      <td>Opera</td>
      <td>1.572880</td>
    </tr>
    <tr>
      <th>126</th>
      <td>0.206608</td>
      <td>0.060654</td>
      <td>0.090909</td>
      <td>0.424113</td>
      <td>0.0</td>
      <td>0.016908</td>
      <td>0.972892</td>
      <td>0.931864</td>
      <td>0.156084</td>
      <td>0.102093</td>
      <td>0.283253</td>
      <td>0.037124</td>
      <td>0.8</td>
      <td>0.042553</td>
      <td>0.365591</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>2AFYShhEHrqugU3vguwlGz</td>
      <td>Opera</td>
      <td>1.578306</td>
    </tr>
    <tr>
      <th>74</th>
      <td>0.366373</td>
      <td>0.040151</td>
      <td>0.727273</td>
      <td>0.473776</td>
      <td>1.0</td>
      <td>0.033605</td>
      <td>0.991968</td>
      <td>0.000009</td>
      <td>0.089747</td>
      <td>0.150142</td>
      <td>0.399575</td>
      <td>0.039419</td>
      <td>0.8</td>
      <td>0.106383</td>
      <td>0.225806</td>
      <td>0.0</td>
      <td>Opera</td>
      <td>14wbIkgXXTHK9ZN7GrPDWc</td>
      <td>Opera</td>
      <td>1.633814</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 20 columns</p>
</div>


