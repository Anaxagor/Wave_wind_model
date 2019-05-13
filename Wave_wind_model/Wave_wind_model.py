import csv
import plotly.plotly as py
import plotly as plotly
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib.mlab import PCA
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn import linear_model,svm, neighbors, ensemble,model_selection
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
import scipy.stats as st
from sklearn.cross_validation import KFold
from sklearn.naive_bayes import GaussianNB
from scipy.cluster import hierarchy
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.ar_model import AR





def dir(x):
    if 0<= x < 45:
        return(1)
    elif 45<= x < 90:
         return(3)
    elif 90<= x < 135:
         return(4)
    elif 135<= x < 180:
         return(7)
    elif 180<= x < 225:
         return(8)
    elif 225<= x < 270:
         return(5)
    elif 270<= x < 315:
         return(6)
    elif 315<= x <=360:
         return(2)
   




points = ['145_73_0_72_99', '174_67_75_72_91','175_68_95_73_03','176_72_67_73_35','212_67_31_73_26','213_68_54_73_38','214_69_78_73_5','215_71_05_73_61','216_72_33_73_71','252_66_85_73_61','253_68_1_73_74','254_69_37_73_86','255_70_67_73_97','256_71_98_74_08','290_66_37_73_96','291_67_65_74_09','292_68_95_74_22','293_70_26_74_33','294_71_6_74_44']

hist = ['wind_speed','pick_period','Avg_wave_period']
coef = []
point = []
for i in range(236664):
    point.append(i)
for j in points:
    
    file = 'point_{0}_1989-01-01_00_2015-12-31_23'.format(j)
    
    data = pd.read_fwf(file)
    columns = data.columns.tolist()
    #Take wind_speed picks_period avg_period after lasso
    need_cols_X = [columns[2],columns[14],columns[15]]
    need_cols_y = [columns[13]]
    data_x = data[need_cols_X]
    data_y = data[need_cols_y]
    data_x.columns = ['wind_speed','pick_period','Avg_wave_period']
    data_y.columns = ['hs']
    train_data_y = data_y[1:len(data_y)-12]
    y_final = []
    #test_data = data_y[data_y[len(data_y)-12:]]
    model_auto = AR(train_data_y)
    model_fitted = model_auto.fit()
    print(len(model_fitted.params))
   # data_x['wind_dir'] = data_x['wind_dir'].apply(dir)
   # data_x['wave_dir'] = data_x['wave_dir'].apply(dir)
    #clf = linear_model.Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000, normalize=False, positive=False, precompute=False, random_state=None, selection='cyclic', tol=0.0001, warm_start=False)
    data_x = data_x.dropna()
    train_data_x = data_x[1:len(data_x)-12]
    
    model_norm = LinearRegression()
    Xtrn, Xtest, Ytrn, Ytest = train_test_split(train_data_x, train_data_y, test_size=0.33)
    #for i in range (100):
     #   n_split = round(random.random(),1)
      #  while n_split < 0.2 or n_split==1:
       #    n_split = random.random()
        #Xtrn, Xtest, Ytrn, Ytest = train_test_split(data_x, data_y, test_size=n_split)
        
    model_norm.fit(Xtrn, Ytrn)
        #print(model_norm.coef_)
  
    y_1 = model_fitted.predict(start=len(train_data_y))
    y_2 = model_norm.predict(data_x )
    coef.append(model_norm.coef_[0])
    n_clusters = 3
    km = KMeans(n_clusters=n_clusters)
    coef_frame = pd.DataFrame(coef)
# fit & predict clusters
    coef_frame['cluster'] = km.fit_predict(coef_frame)
    #km.fit(coef)
    y_kmeans = km.predict(coef)
    label = km.labels_
    #data_x.to_csv('clustering.csv')
    scatter = dict(
    mode = "markers",
    name = "y",
    type = "scatter3d",    
    x = coef_frame[0], y = coef_frame[1], z = coef_frame[2],
    marker = dict( size=2, color="rgb(23, 190, 207)" )
)
    clusters = dict(
    alphahull = 7,
    name = "y",
    opacity = 0.1,
    type = "mesh3d",    
    x = coef_frame[0], y = coef_frame[1], z = coef_frame[2]
)
    layout = dict(
    title = '3d point clustering',
    
    
    scene = dict(
        xaxis = dict(title='wind_speed'),
        yaxis = dict(title='picks_period'),
        zaxis = dict(title='avg_wave_period'),
    )
)
    fig = dict( data=[scatter, clusters], layout=layout )
    plotly.offline.plot(fig, filename='point_{0}_1989-01-01_00_2015-12-31_23.html'.format(j))
    


    
    #clf.fit(data_x,data_y)
    #coef.append(clf.coef_)
    #print(clf.score(data_x,data_y))
#print(coef)

#df = pd.DataFrame({'col':coef})
#df.to_csv('coef_lasso.csv')
#data = pd.read_csv('coef_lasso.csv')


#for i in points:
 #   file = 'point_{0}_1989-01-01_00_2015-12-31_23'.format(i)
  #  data = pd.read_csv(file)


