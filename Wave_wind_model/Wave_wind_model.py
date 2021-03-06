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
from sklearn.metrics import classification_report
from sklearn.cluster import DBSCAN
from saxpy.znorm import znorm
from saxpy.paa import paa
from saxpy.sax import ts_to_string
from saxpy.alphabet import cuts_for_asize
from saxpy.sax import sax_via_window
from saxpy.hotsax import find_discords_hotsax
from numpy import genfromtxt
import seaborn as sns
import plotly.plotly as py
import plotly.tools as tls
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from sklearn.ensemble import RandomForestClassifier






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

def euclid_dist(t1,t2):
    return sqrt(sum((t1-t2)**2))

def DTWDistance(s1, s2,w):
    DTW={}
    
    w = max(w, abs(len(s1)-len(s2)))
    
    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0
  
    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
		
    return math.sqrt(DTW[len(s1)-1, len(s2)-1])

def LB_Keogh(s1,s2,r):
    LB_sum=0
    for ind,i in enumerate(s1):
        
        lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        
        if i>upper_bound:
            LB_sum=LB_sum+(i-upper_bound)**2
        elif i<lower_bound:
            LB_sum=LB_sum+(i-lower_bound)**2
    
    return sqrt(LB_sum)

def aic(y, y_pred, k):
   resid = np.array(y) - np.array(y_pred)
   sse = sum(resid ** 2)
   AIC = 2*k - 2*np.log(sse)
   return AIC






points = ['1'] #['1','2','3','4','5','6','7','8','9','10','11','12','13',#
          


size_of_batch = [72]

point = []



for i in range(76333):
    point.append(i)








for j in points:
    clustering_metrics = pd.DataFrame()
    for chunk in size_of_batch:
        chunks = []
        new_chunks = []
        for i in range(19716):
            new_chunks.append(i)
        number_of_batches = math.ceil(76333/chunk)
        for i in range(int(number_of_batches)):
            chunks.append(i)
        y_pred = []
        coef = []
            
   
    
   
        scaler = MinMaxScaler()
        norm_scaler_x = StandardScaler()
        norm_scaler_y = StandardScaler()
        norm_scaler1 = StandardScaler()
        norm_scaler2 = StandardScaler()


        data = pd.read_fwf(j)
        
        data_icefree = data.loc[(data[data.columns[17]] == 0.00) & (data[data.columns[13]] != 0)]
        data_ice = data.loc[data[data.columns[17]] != 0.00]
        columns1 = data_icefree.columns.tolist()

        df_mse = pd.DataFrame()
        #Take wind_speed picks_period avg_period after lasso
        need_cols_icefree = [columns1[13],columns1[2],columns1[3],columns1[14], columns1[15]]
        need_cols_ice = [columns1[17],columns1[13],columns1[2],columns1[3],columns1[14], columns1[15]]
   
        data_icefree_new = data_icefree[need_cols_icefree]
        data_icefree_new.columns = ['hs','wind_speed','wind_dir','picks_period','avg_period']
        #data = pd.DataFrame(scaler.fit_transform(data))
        data_icefree_new['wind_dir'] = data_icefree_new['wind_dir'].apply(dir)
        #data_icefree_new['wave_dir'] = data_icefree_new['wave_dir'].apply(dir)
       
        
        need_cols_X = ['wind_speed','wind_dir','picks_period','avg_period']
        need_cols_y = ['hs']
        data_x = data_icefree_new[need_cols_X]
        
        data_y = data_icefree_new[need_cols_y]
        #data.columns = ['wind_speed','pick_period','Avg_wave_period','hs']
        #data.to_csv('data in point'+j+'.csv')
        #for i in points:
         #   if i != j:
            
          #      data_other = pd.read_fwf(i)
           #     columns2 = data_other.columns.tolist()
            #    need_cols = [columns2[13]]
             #   df = data_other[need_cols]
              #  data_x = pd.concat([data_x, df], axis=1)
               # data = pd.concat([data,df], axis = 1)
   
            
    
    
    
    
        
        data_icefree_new.to_csv('data in point'+j+' chunk '+str(chunk)+'.csv')
        data_x_t = data_x.transpose()
        data_y_t = data_y.transpose()
    
       
        data_x_norm = pd.DataFrame(norm_scaler_x.fit_transform(data_x))
        data_y_norm = pd.DataFrame(norm_scaler_y.fit_transform(data_y))
   

        #Autoregression
        #train_data_y = data_y[1:len(data_y)-47333]
        #y_final = []
        #test_data_y = data_y[len(data_y)-47333:]
        #model_auto = AR(train_data_y)
        #model_fitted = model_auto.fit()
        #print(len(model_fitted.params))
        # data_x['wind_dir'] = data_x['wind_dir'].apply(dir)
        # data_x['wave_dir'] = data_x['wave_dir'].apply(dir)
   
   
        #pca = PCA(n_components=4)
        #pca.fit_transform(data_x)
        #scaler = StandardScaler().fit(data_x)
        #scaler.transform(data_x)
        
        
        data_x_split = np.array_split(data_x, number_of_batches)
        data_y_split = np.array_split(data_y, number_of_batches)
        data_x_norm_split = np.array_split(data_x_norm, number_of_batches)
        data_y_norm_split = np.array_split(data_y_norm, number_of_batches)
        coef_frame = pd.DataFrame()
        regression_coef = pd.DataFrame()
        feature_importance = pd.DataFrame()
        aic_df = pd.DataFrame()
        
        
        for batch in range(int(number_of_batches)):
        

            clf = linear_model.Lasso(alpha=0.1)
            random_forest_model = RandomForestRegressor()
            Xtrn = data_x_norm_split[batch]       
            Ytrn = data_y_norm_split[batch]
            Xtest = data_x_split[batch]
            
            Ytest = data_y_split[batch]
            
            #model_norm = LinearRegression()
            clf.fit(Xtest,Ytest)
      
            random_forest_model.fit(Xtrn, Ytrn)
        
        

        
            #if mean_squared_error(Ytrn,clf.predict(Xtrn)) <= 1:
           
            df_mse = pd.concat([df_mse, pd.DataFrame([mean_squared_error(Ytest,clf.predict(Xtest))])],axis = 1)
            y_pred.append(clf.predict(Xtest))
            coef = []
            l1 = clf.coef_.tolist()
            l2 = clf.intercept_.tolist()
            coef = [l1[0],l1[1],l1[2],l1[3],l2[0]]
            
            coef_frame = pd.concat([coef_frame,pd.DataFrame(np.transpose(clf.coef_))], axis = 1)
            regression_coef = pd.concat([regression_coef,pd.DataFrame(np.transpose(coef))], axis=1)
            feature_importance = pd.concat([feature_importance,pd.DataFrame(np.transpose(random_forest_model.feature_importances_))], axis=1)
           
            aic_df = pd.concat([aic_df,pd.DataFrame(np.transpose([np.mean(aic(Ytest,clf.predict(Xtest),4)),np.mean(aic(Ytest,random_forest_model.predict(Xtest),4))]))], axis=1)
   
        batches = []
        for batch in range(int(number_of_batches)):
            batches.append(batch)
      
            
            
        aic_df_T = aic_df.transpose()
        aic_df_T.columns = ['linear_aic', 'forest_aic']
        plt.plot(batches,aic_df_T['linear_aic'],label='Linear regression with Lasso')
        plt.plot(batches,aic_df_T['forest_aic'],label='Random forest regression')
        
        plt.legend()
        #sns.heatmap(feature_importance,xticklabels = 1000,yticklabels=need_cols_X,cmap="YlGnBu")
        plt.show()
       
        mse_points = []
        df_mse_T = df_mse.transpose()
        #df_mse_T = pd.DataFrame(scaler.fit_transform(df_mse_T))
        df_mse_T.columns = ['mse']
        regression_coef_norm = pd.DataFrame(norm_scaler1.fit_transform(regression_coef))
        regression_coef_T = regression_coef.transpose()
        feature_importance_T = feature_importance.transpose()
        coef_frame_norm = pd.DataFrame(norm_scaler2.fit_transform(coef_frame))
        #mse = scaler.fit_transform(df_mse)
        
       
        for batch in range(int(number_of_batches)):
            batches.append(batch)
      
        #coef_frame_T = coef_frame.transpose()
        #coef_frame_T = coef_frame_T.transpose()

        sns.heatmap(coef_frame,xticklabels = 1000,yticklabels=need_cols_X,cmap="YlGnBu")
        
        
        plt.show()
        coef_frame_T = coef_frame.transpose()
        coef_frame_T['chunk'] = batches
        regression_coef_T['chunk'] = batches
        #X = scaler.fit_transform(coef_frame)
  
        #coef_frame = pd.DataFrame(X)
        coef_frame_T.to_csv('coef in point'+j+' chunk '+str(chunk)+'.csv')
        coef_frame_T.columns = ['wind_speed','wind_dir','picks_period','avg_period','chunk']
        regression_coef_T.columns = ['wind_speed','wind_dir','picks_period','avg_period','intercept','chunk']
        #cuts_for_asize(3)
        #pca = PCA(n_components=3)
        #X_pca = pca.fit_transform(coef_frame)
        #coef_frame = pd.DataFrame(X_pca)
        #X_norm = norm_scaler.fit_transform(coef_frame_T)
        #coef_frame_new = pd.DataFrame(X_norm)
        #data_X_norm_split = np.array_split(X_norm, 2)
        #clustering = DBSCAN(eps=0.3, min_samples=10).fit(data_X_norm_split[0])
        #df_lables = pd.DataFrame(clustering.labels_)
        #df_lables.to_csv('labesl.csv')

       
       
        #x = np.array_split(X_norm[:,0],3286)
        #labels = []
        #x1 = np.array_split(X_norm[0:1643,0],22)
        #x2 = np.array_split(X_norm[1643:3287,0],22)
        #for number in range(3286):
         #   dat_znorm = znorm(x[number])
          #  dat_paa = paa(dat_znorm, 3)
           # for i in range(6):
            #    labels.append(ts_to_string(dat_paa, cuts_for_asize(3)))
           
            
            
            
            #print(DTWDistance(pd.Series(x1[number]),pd.Series(x2[number]),2))
        
        
        
        #color_dict = {word: c for c, word in enumerate(set(labels))}
        #colors = [color_dict[word] for word in labels]
        #print('-----------------------------------------------------')
        #print(colors)
        #fig, ax = plt.subplots()
        
        #ax.scatter(new_chunks,X_norm[0:19716,0] , c=colors, s=50, cmap='viridis')
     
    
      
        #plt.show()


        #coef_frame = pd.DataFrame(X_norm)
        n_clusters = [3]
        mse_final = pd.DataFrame()    
        #coef_frame.columns = ['wind_speed coef','pick_period coef','Avg_wave_period coef']
        #coef_frame.columns = ['wind_speed_coef','pick_period in_coef','Avg_wave_period_coef']
        #coef_frame.to_csv('PCA coef in point'+j+'.csv')
        #coef_frame.columns = ['one','two']
        chunk_clust = pd.DataFrame()
        for n in n_clusters:
            
            km = KMeans(n_clusters=n)
            km_coef = KMeans(n_clusters=n)
            coef_frame_T['cluster'] = km.fit_predict(coef_frame_T[['wind_speed','wind_dir','picks_period','avg_period']])
            regression_coef_T['cluster'] = km_coef.fit_predict(regression_coef_T[['wind_speed','wind_dir','picks_period','avg_period','intercept']])
          
            #km.fit(X_norm)
            y_kmeans = km.predict(coef_frame_T[['wind_speed','wind_dir','picks_period','avg_period']])
            mse_new = pd.DataFrame()
            clust_centr =  km_coef .cluster_centers_
            #centroid = pd.DataFrame(clust_centr)
            #centroid.columns = ['wind_speed_coef','wind_dir_coef','picks_period_coef','avg_period_coef','intercept']
            #centroid.to_csv('cluster_coef.csv')
            #print(clust_centr)
            for cluster in range(n):
                model = LinearRegression()
                model.coef_=clust_centr[cluster,0:4]
                model.intercept_ = clust_centr[cluster,4]
                clust = pd.DataFrame()
                clust =  regression_coef_T.loc[regression_coef_T['cluster'] == (cluster)]            
                for ch in clust['chunk']:
                    chunk_clust = pd.concat([chunk_clust,pd.DataFrame(np.transpose([ch,cluster]))], axis=1)
                    array = np.array([ch,mean_squared_error(data_y_split[ch],model.predict(data_x_split[ch]))])
                    mse_new = pd.concat([mse_new,pd.DataFrame(np.transpose(array))], axis=1)
            chunk_clust_T = chunk_clust.transpose()
            chunk_clust_T.columns = ['chunk','clust']
            chunk_clust_T = chunk_clust_T.sort_values(['chunk','clust'], ascending=[True, False])
           
            mse_new_T = mse_new.transpose()
            mse_new_T.columns = ['chunk','mse']
            mse_new_T = mse_new_T.sort_values(['chunk','mse'], ascending=[True, False])
            #print(mse_new_T)
            #mse_new_norm = scaler.fit_transform(mse_new_T[['mse']])
            mse_final = pd.concat([mse_final,mse_new_T[['mse']]], axis=1)



            
       
        
        
        
        mse_final = pd.DataFrame(scaler.fit_transform(mse_final))    
        mse_final.columns = ['k1']
        #clust2 =  coef_frame_T.loc[coef_frame_T['cluster'] == 2]
       # clust3 =  coef_frame_T.loc[coef_frame_T['cluster'] == 3]
        
       # plt.scatter(X_norm[:,0],X_norm[:,1],c=y_kmeans, s=50, cmap='viridis')
       # plt.show()
        #colors_points = pd.DataFrame()
        #for c in chunk_clust_T['clust']:
         #   for i in range(72):
          #      colors_points = pd.concat([colors_points,pd.DataFrame([c])], axis=1)
           #     print(colors_points)
 
        #colors_points_T = colors_points.transpose()
        #colors_points_T.columns = ['color']
        #colors_points_T.to_csv('colors.csv')
        #print('-----------------------colors done------------------------------------')
        fig, axs = plt.subplots(5, 1)
     
        
        


        wind_speed_array = data_icefree_new['wind_speed'].values
        wind_speed_array_split = np.array_split(wind_speed_array, number_of_batches)
        point_split = np.array_split(point, number_of_batches)
        
        for ch in range(number_of_batches):
            color = 'blue'
            if (chunk_clust_T.loc[chunk_clust_T['chunk'] == (ch)]['clust'].values[0])==0:
                color = 'darkred'
            if (chunk_clust_T.loc[chunk_clust_T['chunk'] == (ch)]['clust'].values[0])==1:
                color = 'darkblue'
            if (chunk_clust_T.loc[chunk_clust_T['chunk'] == (ch)]['clust'].values[0])==2:
                color = 'green'
            
            axs[0].scatter(point_split[ch],wind_speed_array_split[ch],s=10,c=color)
        axs[0].set_xlabel('points')
        axs[0].set_ylabel('wind_speed ' + j)
        axs[0].grid(True)

        wind_dir_array = data_icefree_new['wind_dir'].values
        wind_dir_array_split = np.array_split(wind_dir_array, number_of_batches)
        point_split = np.array_split(point, number_of_batches)
        
        for ch in range(number_of_batches):
            color = 'blue'
            if (chunk_clust_T.loc[chunk_clust_T['chunk'] == (ch)]['clust'].values[0])==0:
                color = 'darkred'
            if (chunk_clust_T.loc[chunk_clust_T['chunk'] == (ch)]['clust'].values[0])==1:
                color = 'darkblue'
            if (chunk_clust_T.loc[chunk_clust_T['chunk'] == (ch)]['clust'].values[0])==2:
                color = 'green'
            
            axs[1].scatter(point_split[ch],wind_dir_array_split[ch],s=10,c=color)
        axs[1].set_xlabel('points')
        axs[1].set_ylabel('wind_dir ' + j)
        axs[1].grid(True)

        picks_period_array = data_icefree_new['picks_period'].values
        picks_period_array_split = np.array_split(picks_period_array, number_of_batches)
        point_split = np.array_split(point, number_of_batches)
        
        for ch in range(number_of_batches):
            color = 'blue'
            if (chunk_clust_T.loc[chunk_clust_T['chunk'] == (ch)]['clust'].values[0])==0:
                color = 'darkred'
            if (chunk_clust_T.loc[chunk_clust_T['chunk'] == (ch)]['clust'].values[0])==1:
                color = 'darkblue'
            if (chunk_clust_T.loc[chunk_clust_T['chunk'] == (ch)]['clust'].values[0])==2:
                color = 'green'
            
            axs[2].scatter(point_split[ch],picks_period_array_split[ch],s=10,c=color)
        axs[2].set_xlabel('points')
        axs[2].set_ylabel('picks_period ' + j)
        axs[2].grid(True)


        avg_period_array = data_icefree_new['avg_period'].values
        avg_period_array_split = np.array_split(avg_period_array, number_of_batches)
        point_split = np.array_split(point, number_of_batches)
        
        for ch in range(number_of_batches):
            color = 'blue'
            if (chunk_clust_T.loc[chunk_clust_T['chunk'] == (ch)]['clust'].values[0])==0:
                color = 'darkred'
            if (chunk_clust_T.loc[chunk_clust_T['chunk'] == (ch)]['clust'].values[0])==1:
                color = 'darkblue'
            if (chunk_clust_T.loc[chunk_clust_T['chunk'] == (ch)]['clust'].values[0])==2:
                color = 'green'
            
            axs[3].scatter(point_split[ch],avg_period_array_split[ch],s=10,c=color)
        axs[3].set_xlabel('points')
        axs[3].set_ylabel('avg_period ' + j)
        axs[3].grid(True)

        hs_array = data_icefree_new['hs'].values
        hs_array_split = np.array_split(hs_array, number_of_batches)
        point_split = np.array_split(point, number_of_batches)
        
        for ch in range(number_of_batches):
            color = 'blue'
            if (chunk_clust_T.loc[chunk_clust_T['chunk'] == (ch)]['clust'].values[0])==0:
                color = 'darkred'
            if (chunk_clust_T.loc[chunk_clust_T['chunk'] == (ch)]['clust'].values[0])==1:
                color = 'darkblue'
            if (chunk_clust_T.loc[chunk_clust_T['chunk'] == (ch)]['clust'].values[0])==2:
                color = 'green'
            
            axs[4].scatter(point_split[ch],hs_array_split[ch],s=10,c=color)
        axs[4].set_xlabel('points')
        axs[4].set_ylabel('hs ' + j)
        axs[4].grid(True)




       
       
    

        
   
     
        #axs[1].scatter(batches, coef_frame_T['wind_speed'],s=10,c=regression_coef_T['cluster'])
        #axs[1].set_xlabel('chunks')
        #axs[1].set_ylabel('wind_speed 3 clusters')
        #axs[1].grid(True)
      
        

        #axs[2].scatter(batches, coef_frame_T['wind_dir'],s=10,c=regression_coef_T['cluster'])
        #axs[2].set_xlabel('chunks')
        #axs[2].set_ylabel('wind_dir 3 clusters')
        #axs[2].grid(True)
       
   
        #axs[3].scatter(batches, coef_frame_T['picks_period'],s=10,c=regression_coef_T['cluster'])
        #axs[3].set_xlabel('chunks')
        #axs[3].set_ylabel('picks_period 3 clusters')
        #axs[3].grid(True)

        #axs[4].scatter(batches, coef_frame_T['avg_period'],s=10,c=regression_coef_T['cluster'])
        #axs[4].set_xlabel('chunks')
        #axs[4].set_ylabel('avg_period 3 clusters')
        #axs[4].grid(True)
       

       
        plt.show()
        
        
        coef_space1 = coef_frame_T.loc[(coef_frame_T['wind_speed'] != 0) & (coef_frame_T['wind_dir'] != 0) & (coef_frame_T['picks_period'] != 0)&(coef_frame_T['avg_period'] != 0)]
        #plt.savefig('point '+j+' chunk '+str(chunk)+'.png', format='png', dpi=100)
        #plt.clf()
        #fig.tight_layout()
        plt.show()
        print('point  '+j+' chunk '+str(chunk))

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
        #coef_frame['cluster'] = km.fit_predict(coef_frame)
        #km.fit(coef)
        #y_kmeans = km.predict(ensemble_coef)
       # label = km.labels_
            
#data_x.to_csv('clustering.csv')
    columns = coef_space1.columns.tolist()
    coef_space1 = coef_space1[[columns[0],columns[1],columns[2]]]
    coef_space1.columns = ['wind_speed','wind_dir','picks_period']
    for l in range (2,20,3):

        coef_embedded = TSNE(n_components = 3, perplexity = 5,early_exaggeration=l).fit_transform(coef_frame_T)
        df_coef_embedded = pd.DataFrame(coef_embedded)
        df_coef_embedded['cluster'] = km.fit_predict(df_coef_embedded)
        df_coef_embedded.columns = ['ax1','ax2','ax3','cluster']
        y_kmeans =  df_coef_embedded['cluster']
        scatter = dict(

            mode = "markers",
            name = "y",
            type = "scatter3d",    
            x = df_coef_embedded['ax1'], y = df_coef_embedded['ax2'], z = df_coef_embedded['ax3'],
            marker = dict( size=3, color=y_kmeans )
    )
        clusters = dict(
            alphahull = 7,
            name = "y",
            opacity = 0.1,
            type = "mesh3d",    
            x = df_coef_embedded['ax1'], y = df_coef_embedded['ax2'], z = df_coef_embedded['ax3'],
    )
        layout = dict(
            title = '3d point clustering',
    
    
            scene = dict(
            xaxis = dict(title='ax1'),
            yaxis = dict(title='ax2'),
            zaxis = dict(title='ax3'),
        )
    )
        fig = dict( data=[scatter, clusters], layout=layout )
        plotly.offline.plot(fig, filename='early_exaggeration_{0}_72_TSNE(3 components).html'.format(l))
    


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


