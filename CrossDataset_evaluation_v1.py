#need to load the variables from the spyder file with trained model
#then load the trained .h5 model file
#then test the model using different data sets (not the loaded/trained data set)

import spyder_kernels as sk
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, RepeatVector, TimeDistributed
import plotly.graph_objects as go
import math
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#load parameter from model trained using what data set

os.chdir('C:/Users/aljcl/Desktop/SDS/Final_Project/Data')
trained_model='KKK15'

tstep='00008'
parameters=sk.utils.iofuncs.load_dictionary("C:/Users/aljcl/Desktop/SDS/Final_Project/"+trained_model+"_"+"tstep"+tstep+".spydata")
parameters=parameters[0]
train_mae_loss=parameters.get('train_mae_loss')

#load trained LSTM AE model
save_path = './SDS_'+trained_model+'.h5'
model = keras.models.load_model(save_path)

#%%
#load test dataset
Dataset='bl_test11_41-57'
tstep='00008' #for KKK13, tstep=0.01, KKK14,15,16, tstep=0.008, KKK17, tstep=0.005

os.chdir('C:/Users/aljcl/Desktop/SDS/Final_Project/Data')
names=['energy','std','maxabs','skw','kurt','rms','clear','crest','shape','fe','p2p','spectrms','label']
names_r=['energy_r','std_r','maxabs_r','skw_r','kurt_r','rms_r','clear_r','crest_r','shape_r','fe_r','p2p_r','spectrms_r','label']

data_2= pd.read_excel('LDV_train_classification_signalfeatures_'+Dataset+'_tstep'+tstep+'.xls',header=None,names=names, sheet_name='ldv2')
data_2=data_2.drop(['label'],axis=1)
data_r= pd.read_excel('LDV_train_classification_signalfeatures_'+Dataset+'_tstep'+tstep+'.xls',header=None,names=names_r, sheet_name='ldv_r')
data=pd.concat([data_2,data_r],axis=1)
data_label=pd.DataFrame(data['label'])


float_columns=[x for x in data.columns != 'label']
float_columns=data.columns[float_columns]




data_2=data_2.drop(['clear','p2p','maxabs','crest','spectrms'],axis=1)
data_r=data_r.drop(['kurt_r','skw_r','clear_r','shape_r','energy_r','std_r','p2p_r','crest_r','fe_r','rms_r'],axis=1) #spectrms_r, rms_r
data=pd.concat([data_2,data_r],axis=1)

float_columns=[x for x in data.columns != 'label']
float_columns=data.columns[float_columns]

skew_columns=(data[float_columns].skew().sort_values(ascending=False))
skew_columns = skew_columns.loc[skew_columns>0.75]


for col in skew_columns.index.tolist():
    data[col]=np.log1p(data[col])



for col in float_columns:
    data[col]=data[col].rolling(window=70,center=False,min_periods=1).mean() #300 for KKK13 70 for 14 15 16 , 50for 17
#%%
plt.plot(data['energy'])


#%%

start_index = 2000
end_index = 3000
x_train = data[start_index:end_index]
y_train = data['label'][start_index:end_index]

x_train1, x_val1, y_train1, y_val1 = train_test_split(x_train, y_train, test_size = 0.1, shuffle = False) #train test split


#%%
#x_train1, x_val1, y_train1, y_val1 = train_test_split(data[float_columns], data['label'], test_size = 0.9, shuffle = False) #train test split

x_test_1 = data[float_columns][:start_index]
y_test_1 = data['label'][:start_index]
x_test_2 = data[float_columns][end_index:]
y_test_2 = data['label'][end_index:]
#x_train2, x_test2, y_train2, y_test2 = train_test_split(data[float_columns], data['label'], test_size = 0.2, shuffle = False) #train test split

x_testtest1 = np.concatenate([x_test_1, x_test_2], axis=0)
y_test1 = np.concatenate([y_test_1, y_test_2], axis=0)
x_testtest1=pd.DataFrame(x_testtest1,columns=['energy','std','skw','kurt','rms','shape','fe','maxabs_r','spectrms_r'])


#%%

mms = MinMaxScaler()
ss = StandardScaler()

ssmean=pd.DataFrame()
ssscale= pd.DataFrame()
ssmean.rename(columns=data.columns)
ssscale.rename(columns=data.columns)

for col in float_columns:
    x_train1[col] = ss.fit_transform(x_train1[[col]]).squeeze()
    ssmean[col]=ss.mean_
    ssscale[col]=ss.scale_
    data[col]=ss.transform(data[[col]]).squeeze()



pca_columns=pd.Series(['std','fe','energy','rms','kurt']) 


x_train1=x_train1[pca_columns]
x_val1=x_val1[pca_columns]
data=data[pca_columns]

TIME_STEPS=20
#%%
class SequenceGenerator:
    def __init__(self, time_steps=TIME_STEPS):
        self.time_steps = time_steps
    
    def create_sequences(self, X, y):
        Xs, ys = [], []
        for i in range(len(X)-self.time_steps):
            Xs.append(X.iloc[i:(i+self.time_steps)].values)
            ys.append(y.iloc[i+self.time_steps])
        
        return np.array(Xs), np.array(ys)
    

seq_gen = SequenceGenerator(TIME_STEPS)
data_feature, data_label = seq_gen.create_sequences(data,data_label)
x_train, y_train = seq_gen.create_sequences(x_train1, y_train1)
x_test, y_test = seq_gen.create_sequences(x_val1, y_val1)

#%
# Get train MAE loss.
train_mae_loss = train_mae_loss[50:-1,]
threshold = np.max(train_mae_loss)*1+3*np.std(train_mae_loss)  #1.5 for KKK13, 2.1-KKK15,3.6for KKK17


# Get test MAE loss.
x_test_pred = model.predict(data_feature)
test_mae_loss = np.mean(np.abs(x_test_pred - data_feature), axis=1)


test_mae_loss=np.sum(test_mae_loss, axis=1)
test_mae_loss = test_mae_loss.reshape((-1))

plt.hist(test_mae_loss, bins=50)
plt.xlabel("test MAE loss")
plt.ylabel("No of samples")
plt.show()

# Detect all the samples which are anomalies.
anomalies = test_mae_loss > threshold
print("Number of anomaly samples: ", np.sum(anomalies))
print("Indices of anomaly samples: ", np.where(anomalies))

# data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
anomalous_data_indices_fact = []

for data_idx in range(TIME_STEPS - 1, len(data_feature) - TIME_STEPS + 1):
    if np.all(anomalies[data_idx - TIME_STEPS + 1 : data_idx]):
        anomalous_data_indices_fact.append(data_idx)


predicted_labels = (test_mae_loss > threshold).astype(int)
#%%

velocity_ldv2=pd.read_excel('LDV2_downsample_'+Dataset+'_tstep'+tstep+'.xls',header=None)

from scipy.interpolate import interp1d

def interpolate_b(a, b):
    # Create a function that interpolates b
    f = interp1d(np.arange(len(b)), b, kind='linear')

    # Create a new array that contains the interpolated values of b
    new_b = f(np.linspace(0, len(b)-1, len(a)))

    return new_b

# Example usage
velocity_ldv2 = pd.DataFrame(interpolate_b(np.array(data_label).squeeze() , np.array(velocity_ldv2).squeeze()))

fig1, ax1 = plt.subplots()
ax1.plot(velocity_ldv2)
ax1.plot(velocity_ldv2.iloc[anomalous_data_indices_fact],color='r')
plt.xlabel("data points")
plt.ylabel("velocity(m/s)")

indx_fact=np.array(anomalous_data_indices_fact)
labels_pretrain=np.zeros(len(velocity_ldv2))
labels_pretrain[indx_fact.astype(int)]=1



fig2, ax2 = plt.subplots()
ax2.plot(data['energy'])
ax2.plot(data['energy'].iloc[anomalous_data_indices_fact],color='r')
plt.xlabel("data points")
#%%
ldv2_acc_org= pd.read_csv('LDV2_vel_original_'+Dataset+'_tstep'+tstep+'.txt', sep=" ")
fs=1250000
time=np.linspace(0, len(ldv2_acc_org)/fs, num=len(ldv2_acc_org))
mae_loss=test_mae_loss
time_mae=np.linspace(0,time[-1],num=len(mae_loss))

fig1, ax1 = plt.subplots()
ax1.plot(time_mae,mae_loss,c='k')
plt.xlabel("Time(s)",fontsize=16,fontname='Times New Roman')
plt.ylabel("MAE Loss",fontsize=16,fontname='Times New Roman')
plt.xlim(xmin=min(time), xmax=max(time))

fig1.set_figwidth(12)
fig1.set_figheight(4)
ax1.axhline(threshold, xmin=0.0, xmax=1.0, color='k',linestyle='--',linewidth=3)
timearray=np.append(np.arange(0, time[-1], step=1),round(time[-1],1))
plt.xticks(timearray,fontsize=12) 
#ax1.plot(velocity_ldv2.iloc[indx],color='r')

indx_fact=np.array(anomalous_data_indices_fact)
labels_fact=np.zeros(len(velocity_ldv2))
labels_fact[indx_fact.astype(int)]=1

#%%



anomalous_data_indices=[]
for data_idx in range(TIME_STEPS - 1, len(x_test) - TIME_STEPS + 1):
    if np.all(anomalies[data_idx - TIME_STEPS + 1 : data_idx]):
        anomalous_data_indices.append(data_idx)

indx=anomalous_data_indices+np.ones(len(anomalous_data_indices))*y_train.shape[0]


#%

# read text file into pandas DataFrame
ldv2_acc_org= pd.read_csv('LDV2_vel_original_'+Dataset+'_tstep'+tstep+'.txt', sep=" ")
ldv1_acc_org= pd.read_csv('LDV1_vel_original_'+Dataset+'_tstep'+tstep+'.txt', sep=" ")

fs=1250000
# display DataFrame
print(ldv2_acc_org)

#%
def separate_array(arr):
  sublists = []
  sublist = [arr[0]]
  for i in range(1, len(arr)):
    if arr[i] - arr[i-1] == 1:
      sublist.append(arr[i])
    else:
      sublists.append(sublist)
      sublist = [arr[i]]
  sublists.append(sublist)
  return sublists


indx_sep=separate_array(indx_fact)
#%
anom_start_end=[]
anom_org=[]
for i in range(0,len(indx_sep)):
    indx_org_s=math.floor((indx_sep[i][0]/len(velocity_ldv2))*len(ldv2_acc_org))
    indx_org_e=math.floor((indx_sep[i][-1]/len(velocity_ldv2))*len(ldv2_acc_org))
    anom_start_end.append([indx_org_s,indx_org_e])
    anom_org.append(np.arange(indx_org_s,indx_org_e+1))

# indx_org_s=math.floor((indx[0]/len(velocity_ldv2))*len(ldv2_acc_org))
# indx_org_e=math.floor((indx[-1]/len(velocity_ldv2))*len(ldv2_acc_org))
# indx_org=np.arange(indx_org_s,indx_org_e+1)


#%
shift=2.0
hfont = {'fontname':'Times New Roman'}
time=np.linspace(0, len(ldv2_acc_org)/fs, num=len(ldv2_acc_org))
fig2, ax2 = plt.subplots()
ax2.plot(time,ldv2_acc_org,color='k',linewidth=0.5)
ax2.plot(time,ldv1_acc_org,color='k',linestyle='--',linewidth=0.5)
for i in range(0,len(indx_sep)):
    ax2.plot(time[anom_org[i]],ldv2_acc_org.iloc[anom_org[i]],c='0.45',linewidth=0.5) #,marker="o",markersize=3,markerfacecolor='none'
    ax2.plot(time[anom_org[i]],ldv1_acc_org.iloc[anom_org[i]],c='0.45',linestyle='--',linewidth=0.5) #,marker="o",markersize=3,markerfacecolor='none'

#ax2.plot(time[indx_org],ldv2_acc_org.iloc[indx_org],c='0.45',linewidth=0.1,marker="o",markersize=3,markerfacecolor='none')



plt.xlabel("Time(s)", **hfont,fontsize=16)
plt.ylabel("Velocity(m/s)", **hfont,fontsize=16)
# plt.xlim(xmin=min(time), xmax=max(time))


timearray=np.append(np.arange(0, time[-1], step=1),round(time[-1],1))
plt.xticks(timearray,fontsize=12) 
plt.xlim(xmin=shift, xmax=max(time))
fig2.set_figwidth(12)
fig2.set_figheight(4)
#%%
ldv2_acc_org = ldv2_acc_org.to_numpy()
lim_array=max(ldv2_acc_org)*1.5
lim=round(lim_array[0],2)
#lim=0.06
ax2.set_ylim([-1*lim,lim])
ax2.set_yticks(np.linspace(-1*lim,lim,num=5))
plt.yticks(fontsize=12)
ax2.grid(False)
ax2.legend(["LDV2","LDV1","LDV2_anomaly","LDV1_anomaly"],loc='upper right', fancybox=True, framealpha=0.1,ncol=4)
