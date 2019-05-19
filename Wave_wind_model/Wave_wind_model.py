import csv
import plotly.plotly as py
import plotly as plotly
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import PCA
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn import linear_model,svm, neighbors, ensemble,model_selection
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import scipy.stats as st
from sklearn.cross_validation import KFold
from sklearn.naive_bayes import GaussianNB
from scipy.cluster import hierarchy
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
#from statsmodels.tsa.ar_model import AR
from datetime import datetime
import math
from sklearn.manifold import TSNE
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn import metrics






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
   

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))





points = ['1'] #['1','2','3','4','5','6','7','8','9','10','11','12','13',#
          


size_of_batch = 864
number_of_batches = math.ceil(236664/size_of_batch)

point = []
chunks = []

y_pred = []
for i in range(236663):
    point.append(i)
for i in range(int(number_of_batches)):
    chunks.append(i)



for j in points:
    y_pred = []
    coef = []
   
    
   
    scaler = MinMaxScaler()
    data = pd.read_fwf(j)
    columns1 = data.columns.tolist()

    mse = []
    #Take wind_speed picks_period avg_period after lasso
    need_cols = [columns1[13],columns1[2],columns1[14],columns1[15]]
   
    data = data[need_cols]
    
    #data = pd.DataFrame(scaler.fit_transform(data))
    
    columns1 = data.columns.tolist()
    need_cols_X = [columns1[1],columns1[2],columns1[3]]
    need_cols_y = [columns1[0]]
    data_x = data[need_cols_X]
    
    data_y = data[need_cols_y]
    #data.columns = ['wind_speed','pick_period','Avg_wave_period','hs']
    #data.to_csv('data in point'+j+'.csv')
    for i in points:
        if i != j:
            
            data_other = pd.read_fwf(i)
            columns2 = data_other.columns.tolist()
            need_cols = [columns2[13]]
            df = data_other[need_cols]
            data_x = pd.concat([data_x, df], axis=1)
            data = pd.concat([data,df], axis = 1)
   
            
    
    
    
    
   # data.columns = ['hs','ws','picks','period','ws1','hs1','ws2','hs2','ws3','hs3','ws4','hs4','ws5','hs5','ws6','hs6','ws7','hs7','ws8','hs8','ws9','hs9','ws10','hs10','ws11','hs11','ws12','hs12','ws13','hs13','ws14','hs14','ws15','hs15','ws16','hs16','ws17','hs17']
    data.to_csv('data in point'+j+'.csv')
    data_norm = pd.DataFrame(scaler.fit_transform(data))
    
    
    data_x_norm = pd.DataFrame(scaler.fit_transform(data_x))
    data_y_norm = pd.DataFrame(scaler.fit_transform(data_y))
   

    #Autoregression
    #train_data_y = data_y[1:len(data_y)-47333]
    #y_final = []
    #test_data_y = data_y[len(data_y)-47333:]
    #model_auto = AR(train_data_y)
    #model_fitted = model_auto.fit()
    #print(len(model_fitted.params))
   # data_x['wind_dir'] = data_x['wind_dir'].apply(dir)
   # data_x['wave_dir'] = data_x['wave_dir'].apply(dir)
   
    data_x_norm = data_x.dropna()
    data_y_norm = data_y.dropna()
   
    #pca = PCA(n_components=4)
    #pca.fit_transform(data_x)
    #scaler = StandardScaler().fit(data_x)
    #scaler.transform(data_x)
    
    y_mean = pd.DataFrame()
    data_x_split = np.array_split(data_x, number_of_batches)
    data_y_split = np.array_split(data_y, number_of_batches)
    data_x_norm_split = np.array_split(data_x_norm, number_of_batches)
    data_y_norm_split = np.array_split(data_y_norm, number_of_batches)
    coef_frame = pd.DataFrame()
    for batch in range(int(number_of_batches)):
        


        Xtrn = data_x_norm_split[batch]
        
        Ytrn = data_y_norm_split[batch]
        Xtest = data_x_split[batch]
        Ytest = data_y_split[batch]
        model_norm = LinearRegression()
        df = pd.DataFrame(np.mean(Ytest))
        y_mean = pd.concat([y_mean,df], axis = 1)
       
   
        model_norm.fit(Xtrn,Ytrn)
        
        

        
        
        mse.append(mean_squared_error(Ytest,model_norm.predict(Xtest)))
        y_pred.append(model_norm.predict(Xtest))
        coef.append(model_norm.coef_.tolist())
        
        coef_frame = pd.concat([coef_frame,pd.DataFrame(np.transpose(model_norm.coef_))], axis = 1)
   
    y_mean = y_mean.transpose() 
    batches = []
    for batch in range(int(number_of_batches)):
        batches.append(batch)
    
        
    
    
    coef_frame = coef_frame.transpose()
    #X = scaler.fit_transform(coef_frame)
  
    #coef_frame = pd.DataFrame(X)
    coef_frame.to_csv('coef in point'+j+'.csv')
    #pca = PCA(n_components=2)
    #X_pca = pca.fit_transform(coef_frame)
    #coef_frame = pd.DataFrame(X_pca)
    X_norm = scaler.fit_transform(coef_frame)
    coef_frame = pd.DataFrame(X_norm)
    n_clusters = 3
    
    #coef_frame.columns = ['wind_speed coef','pick_period coef','Avg_wave_period coef']
    #coef_frame.columns = ['wind_speed_coef','pick_period in_coef','Avg_wave_period_coef']
    #coef_frame.to_csv('PCA coef in point'+j+'.csv')
    #coef_frame.columns = ['one','two']
    
    km = KMeans(n_clusters=n_clusters)
    #coef_frame['cluster'] = km.fit_predict(X_pca_norm)
    #km.fit(X_pca_norm)
    #y_kmeans = km.predict(X_pca_norm)
    
    fig, axs = plt.subplots(4, 1)
    axs[0].plot(chunks,mse)
    axs[0].set_xlabel('chunk')
    axs[0].set_ylabel('MSE in point ' + j)
    axs[0].grid(True)
   
    
    axs[1].plot(chunks,X_norm[:,0])
    axs[1].set_xlabel('chunks')
    axs[1].set_ylabel('wind_speed')
    axs[1].grid(True)

    axs[2].plot(chunks,X_norm[:,1])
    axs[2].set_xlabel('chunks')
    axs[2].set_ylabel('picks')
    axs[2].grid(True)
   
    axs[3].plot(chunks,X_norm[:,2])
    
    axs[3].set_xlabel('chunks')
    axs[3].set_ylabel('avg_period')
    axs[3].grid(True)

    plt.savefig('point '+j+'.png', format='png', dpi=100)
    plt.clf()
    #fig.tight_layout()
    #plt.show()
    print('point  '+j)
    #ax = plt.subplot(111)
    #ax.plot(butches,y_pred[0], label='hs predicted')
    #ax.plot(butches, data_y_split[0],label='hs real')
    #ax.set_xlabel('time')
    #ax.set_ylabel('hs ' + j)
    #ax.grid(True)
    #ax.legend()
    #plt.show()
  
    #y_1 = model_fitted.predict(start=len(train_data_y), end = len(train_data_y)+len(test_data_y) - 1,dynamic=False)
    #y_1 = np.array(y_1)
    #print(y_1)
    #y_2 = np.array(model_norm.predict(Xtest))
    
    #coef.append(model_norm.coef_[0])
    #n_clusters = 3
    #km = KMeans(n_clusters=n_clusters)
    #data_other_points_x = np.zeros((236663,3))
    #n = 0
    #for i in points:
     #   if i != j:
      #      file = 'point_{0}_1989-01-01_00_2015-12-31_23'.format(i)
       #     data_other = pd.read_fwf(file)
        #    columns = data_other.columns.tolist()
    #Take wind_speed picks_period avg_period after lasso
         #   need_cols_X = [columns[2],columns[14],columns[15]]
         #   data_other_points_x += data_other[need_cols_X].values
         #   n+=1
    #data_other_points_x /= n
   
    #model_norm_other = LinearRegression()
 
    #Xtrn, Xtest = train_test_split(data_other_points_x, shuffle=False, test_size=0.2)
    #model_norm_other.fit(Xtrn, Ytrn)

    #y_3 = np.array(model_norm_other.predict(Xtest))
    
    #y_ensemble = pd.DataFrame([y_1,model_norm.predict(Xtest),model_norm_other.predict(Xtest)])
    #y_ensemble = y_ensemble.transpose()
    #y_ensemble.columns = ['a','b','r']
    #print(y_ensemble.columns)
    #y_ensemble['r'] = y_ensemble['r'].apply(lambda x: x[0])
    #y_ensemble['b'] = y_ensemble['b'].apply(lambda x: x[0])
   
   # y_ensemble = pd.DataFrame([y_1,y2_ensemble, y3_ensemble])
   # y_ensemble = y_ensemble['c'].apply(lambda x: x[0])
    #y_ensemble.to_csv('ensemble.csv')
    #ensemble_coef = []

    #print(y_ensemble)
    #for i in range (100):
     #   model_norm_other = LinearRegression()
      #  n_split = round(random.random(),1)
       # while n_split < 0.2 or n_split==1:
        #   n_split = random.random()
        #Xtrn, Xtest, Ytrn, Ytest = train_test_split(y_ensemble, test_data_y, shuffle=False, test_size=n_split)
        
       # model_norm_other.fit(Xtrn, Ytrn)
       # ensemble_coef.append(model_norm_other.coef_[0])



    #print(y_ensemble)
    #model_norm_other.fit(y_ensemble,Ytest)
    #print(model_norm_other.coef_)
    #y_pred_ensemble = model_norm_other.predict(Xtest)
    #plt.scatter(point,Ytest)
    #plt.scatter(point, y_pred_ensemble)
    #plt.show()





    #coef_frame = pd.DataFrame(ensemble_coef)
# fit & predict clusters
    coef_frame['cluster'] = km.fit_predict(coef_frame)
    #km.fit(coef)
    #y_kmeans = km.predict(ensemble_coef)
    label = km.labels_
    print(metrics.adjusted_rand_score(y_mean.values[:,0], label))
    #data_x.to_csv('clustering.csv')
    scatter = dict(
    mode = "markers",
    name = "y",
    type = "scatter3d",    
    x = X_norm[:,0], y = X_norm[:,1], z = X_norm[:,2],
    marker = dict( size=2, color="rgb(23, 190, 207)" )
)
    clusters = dict(
    alphahull = 7,
    name = "y",
    opacity = 0.1,
    type = "mesh3d",    
    x = X_norm[:,0], y = X_norm[:,1], z = X_norm[:,2],
)
    layout = dict(
    title = '3d point clustering',
    
    
    scene = dict(
        xaxis = dict(title='wind_speed'),
        yaxis = dict(title='picks'),
        zaxis = dict(title='avg_period'),
    )
)
    fig = dict( data=[scatter, clusters], layout=layout )
    plotly.offline.plot(fig, filename='point_{0}_1989-01-01_00_2015-12-31_23.html'.format(j))
    


    #clf = linear_model.Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000, normalize=False, positive=False, precompute=False, random_state=None, selection='cyclic', tol=0.0001, warm_start=False)
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


