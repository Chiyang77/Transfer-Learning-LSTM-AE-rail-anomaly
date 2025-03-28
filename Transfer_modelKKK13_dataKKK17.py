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
trained_model='KKK13'
tstep='0001'
parameters=sk.utils.iofuncs.load_dictionary("C:/Users/aljcl/Desktop/SDS/Final_Project/"+trained_model+"_"+"tstep"+tstep+".spydata")
parameters=parameters[0]
train_mae_loss=parameters.get('train_mae_loss')

#load trained LSTM AE model
save_path = './SDS_'+trained_model+'.h5'
model = keras.models.load_model(save_path)

#%%
#load test dataset
#Dataset='KKK17'
Dataset='KKK17'
tstep='00005' #for KKK13, tstep=0.01, KKK14,15,16, tstep=0.008, KKK17, tstep=0.005

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
    data[col]=data[col].rolling(window=50,center=False,min_periods=1).mean() #300 for KKK13 70 for 14 15 16 , 50for 17


#%%
X_train1, X_test1, y_train1, y_test1 = train_test_split(data[float_columns], data['label'], test_size = 0.3, shuffle = False)
data=data.drop(['label'],1)





mms = MinMaxScaler()
ss = StandardScaler()

ssmean=pd.DataFrame()
ssscale= pd.DataFrame()
ssmean.rename(columns=data.columns)
ssscale.rename(columns=data.columns)

for col in float_columns:
    X_train1[col] = ss.fit_transform(X_train1[[col]]).squeeze()
    ssmean[col]=ss.mean_
    ssscale[col]=ss.scale_
    X_test1[col] = ss.transform(X_test1[[col]]).squeeze()
    data[col]=ss.transform(data[[col]]).squeeze()



# =============================================================================
# ss1 = StandardScaler()
# ss1.mean_ = ssmean
# ss1.scale_ = ssscale
# 
# for col in float_columns:
#     ss1.mean_ = ssmean[col].squeeze()
#     ss1.scale_ = ssscale[col].squeeze()
#     X_train1[col]=ss1.transform(X_train1[[col]]).squeeze()
#     X_test1[col]=ss1.transform(X_test1[[col]]).squeeze()
#     data[col]=ss1.transform(data[[col]]).squeeze()
# 
# =============================================================================


pca_columns=pd.Series(['std','fe','energy','rms','kurt']) 

data=data[pca_columns]
X_train1=X_train1[pca_columns]
X_test1=X_test1[pca_columns]

TIME_STEPS=20

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
X_train, y_train = seq_gen.create_sequences(X_train1, y_train1)
X_test, y_test = seq_gen.create_sequences(X_test1, y_test1)

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
        
#%%

velocity_ldv2=pd.read_excel('LDV2_downsample_'+Dataset+'_tstep'+tstep+'.xls',header=None)
#velocity_ldv2=pd.read_excel('LDV2_downsample_KKK14_tstep00008.xls',header=None)


fig1, ax1 = plt.subplots()
ax1.plot(velocity_ldv2)
ax1.plot(velocity_ldv2.iloc[anomalous_data_indices_fact],color='r')
plt.xlabel("data points")
plt.ylabel("velocity(m/s)")

fig2, ax2 = plt.subplots()
ax2.plot(data['energy'])
ax2.plot(data['energy'].iloc[anomalous_data_indices_fact],color='r')
plt.xlabel("data points")

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
timearray=np.append(np.arange(0, time[-1], step=0.5),round(time[-1],1))
plt.xticks(timearray,fontsize=12) 
#ax1.plot(velocity_ldv2.iloc[indx],color='r')

indx_fact=np.array(anomalous_data_indices_fact)
labels_fact=np.zeros(len(velocity_ldv2))
labels_fact[indx_fact.astype(int)]=1
#%%



anomalous_data_indices=[]
for data_idx in range(TIME_STEPS - 1, len(X_test) - TIME_STEPS + 1):
    if np.all(anomalies[data_idx - TIME_STEPS + 1 : data_idx]):
        anomalous_data_indices.append(data_idx)

indx=anomalous_data_indices+np.ones(len(anomalous_data_indices))*y_train.shape[0]


#%%

# read text file into pandas DataFrame
ldv2_acc_org= pd.read_csv('LDV2_vel_original_'+Dataset+'_tstep'+tstep+'.txt', sep=" ")
ldv1_acc_org= pd.read_csv('LDV1_vel_original_'+Dataset+'_tstep'+tstep+'.txt', sep=" ")

fs=1250000
# display DataFrame
print(ldv2_acc_org)

#%%
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
#%%
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


#%%
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
plt.xlim(xmin=min(time), xmax=max(time))
timearray=np.append(np.arange(0, time[-1], step=0.5),round(time[-1],1))
plt.xticks(timearray,fontsize=12) 
#plt.grid(visible=None)
fig2.set_figwidth(12)
fig2.set_figheight(4)

ldv2_acc_org = ldv2_acc_org.to_numpy()
lim_array=max(ldv2_acc_org)*1.5
lim=round(lim_array[0],2)
#lim=0.06
ax2.set_ylim([-1*lim,lim])
ax2.set_yticks(np.linspace(-1*lim,lim,num=5))
plt.yticks(fontsize=12)
ax2.grid(False)
ax2.legend(["LDV2","LDV1","LDV2_anomaly","LDV1_anomaly"],loc='upper right', fancybox=True, framealpha=0.1,ncol=4)

#%%



os.chdir('C:/Users/aljcl/Desktop/SDS/Final_Project/Data')
trained_model='KKK17'
tstep='00005'
parameters=sk.utils.iofuncs.load_dictionary("C:/Users/aljcl/Desktop/SDS/Final_Project/"+trained_model+"_"+"tstep"+tstep+".spydata")
parameters=parameters[0]
train_mae_loss=parameters.get('train_mae_loss')

#load trained LSTM AE model
save_path = './SDS_'+trained_model+'.h5'
model = keras.models.load_model(save_path)


#%%

# Get train MAE loss.
train_mae_loss = train_mae_loss[50:-1,]

# Get reconstruction loss threshold.
#threshold = np.max(train_mae_loss[:,0]+train_mae_loss[:,1]+train_mae_loss[:,2]+train_mae_loss[:,3]+train_mae_loss[:,4])
#threshold = np.max(train_mae_loss)+30*np.std(np.abs(x_train_pred - x_train))
threshold = np.max(train_mae_loss)*1+3*np.std(train_mae_loss)  #1.5 for KKK13, 2.1-KKK15,3.6for KKK17


# Get test MAE loss.
x_test_pred = model.predict(data_feature)
test_mae_loss = np.mean(np.abs(x_test_pred - data_feature), axis=1)

#test_mae_loss= test_mae_loss[:,1]+test_mae_loss[:,0]+test_mae_loss[:,2]+test_mae_loss[:,3]++test_mae_loss[:,4]
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
anomalous_data_indices_pretrain = []

for data_idx in range(TIME_STEPS - 1, len(data_feature) - TIME_STEPS + 1):
    if np.all(anomalies[data_idx - TIME_STEPS + 1 : data_idx]):
        anomalous_data_indices_pretrain.append(data_idx)

indx_pretrain=np.array(anomalous_data_indices_pretrain)
labels_pretrain=np.zeros(len(velocity_ldv2))
labels_pretrain[indx_pretrain.astype(int)]=1

#

#velocity_ldv2=pd.read_excel('LDV2_downsample_KKK14_tstep00008.xls',header=None)


fig1, ax1 = plt.subplots()
ax1.plot(velocity_ldv2)
ax1.plot(velocity_ldv2.iloc[indx_pretrain],color='r')
plt.xlabel("data points")
plt.ylabel("velocity(m/s)")

fig2, ax2 = plt.subplots()
ax2.plot(data['energy'])
ax2.plot(data['energy'].iloc[indx_pretrain],color='r')
plt.xlabel("data points")
#ldv2_acc_org= pd.read_csv('LDV2_vel_original_'+Dataset+'_tstep'+tstep+'.txt', sep=" ")
#ldv2_acc_org= pd.read_csv('LDV2_vel_original_'+Dataset+'_tstep'+'00008'+'.txt', sep=" ")
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
timearray=np.append(np.arange(0, time[-1], step=0.5),round(time[-1],1))
plt.xticks(timearray,fontsize=12) 



pretrain_cm = confusion_matrix(labels_fact, labels_pretrain)

pretrain_report = classification_report(labels_fact, labels_pretrain)

print(pretrain_report)


#%%
os.chdir('C:/Users/aljcl/Desktop/SDS/Final_Project/Data')
trained_model='KKK13'
tstep='0001'
parameters=sk.utils.iofuncs.load_dictionary("C:/Users/aljcl/Desktop/SDS/Final_Project/"+trained_model+"_"+"tstep"+tstep+".spydata")
parameters=parameters[0]
train_mae_loss=parameters.get('train_mae_loss')

#load trained LSTM AE model
save_path = './SDS_'+trained_model+'.h5'
model = keras.models.load_model(save_path)


Dataset='KKK17'
tstep='00005' #for KKK13, tstep=0.01, KKK14,15,16, tstep=0.008, KKK17, tstep=0.005

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
    data[col]=data[col].rolling(window=50,center=False,min_periods=1).mean() #300 for KKK13 70 for 14 15 16 , 50for 17


X_train1, X_test1, y_train1, y_test1 = train_test_split(data[float_columns], data['label'], test_size = 0.3, shuffle = False)
data=data.drop(['label'],1)





mms = MinMaxScaler()
ss = StandardScaler()

ssmean=pd.DataFrame()
ssscale= pd.DataFrame()
ssmean.rename(columns=data.columns)
ssscale.rename(columns=data.columns)

for col in float_columns:
    X_train1[col] = ss.fit_transform(X_train1[[col]]).squeeze()
    ssmean[col]=ss.mean_
    ssscale[col]=ss.scale_
    X_test1[col] = ss.transform(X_test1[[col]]).squeeze()
    data[col]=ss.transform(data[[col]]).squeeze()



# =============================================================================
# ss1 = StandardScaler()
# ss1.mean_ = ssmean
# ss1.scale_ = ssscale
# 
# for col in float_columns:
#     ss1.mean_ = ssmean[col].squeeze()
#     ss1.scale_ = ssscale[col].squeeze()
#     X_train1[col]=ss1.transform(X_train1[[col]]).squeeze()
#     X_test1[col]=ss1.transform(X_test1[[col]]).squeeze()
#     data[col]=ss1.transform(data[[col]]).squeeze()
# 
# =============================================================================


pca_columns=pd.Series(['std','fe','energy','rms','kurt']) 

data=data[pca_columns]
X_train1=X_train1[pca_columns]
X_test1=X_test1[pca_columns]

TIME_STEPS=20

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
X_train, y_train = seq_gen.create_sequences(X_train1, y_train1)
X_test, y_test = seq_gen.create_sequences(X_test1, y_test1)


print(f'Training shape: {X_train.shape}')
print(f'Testing shape: {X_test.shape}')




#%%
transfer_model = model
for layer in model.layers:

  print(layer, layer.trainable)


# Freezing all layers except the five inner most layers.

for layer in transfer_model.layers[0:5]:
  layer.trainable = False

for layer in transfer_model.layers[-5:]:
  layer.trainable = False


for layer in transfer_model.layers:

  print(layer, layer.trainable)


## Re-Initialization of layers in the center
initializer = keras.initializers.GlorotUniform()

# =============================================================================
# for layer in transfer_model.layers[5:10]:
#   #layer.kernel.initializer.run(session=s)
#   layer.set_weights([initializer(shape=w.shape) for w in layer.get_weights()])
# 
# =============================================================================

frozen_lr = 0.001
optimizer = keras.optimizers.Adam(learning_rate=frozen_lr)

for layer in transfer_model.layers:
    if not layer.trainable:
        layer.set_weights([initializer(shape=w.shape) for w in layer.get_weights()])
        layer._trainable_weights = []
        layer._non_trainable_weights = layer.weights
        layer._optimizer = optimizer

#%%

transfer_es = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
transfer_history=transfer_model.fit(X_train, X_train, epochs=50, batch_size=72, validation_split=0.2,callbacks= transfer_es)
#transfer_history=transfer_model.fit(X_train, X_train, epochs=50, batch_size=72, validation_split=0.1)
#%%
hfont = {'fontname':'Times New Roman'}
plt.grid(visible=None)
plt.ylim(ymin=0,ymax=round(max(max(transfer_history.history["val_loss"]),max(transfer_history.history["loss"])),1)*1.2)
plt.xlim(0,50)
plt.plot(transfer_history.history["val_loss"], label="Validation Loss",color='k',linestyle='-')
plt.plot(transfer_history.history["loss"], label="Training Loss",color='0.45',linestyle='--')
plt.xlabel("Number of Epoch",fontsize=24) #plt.xlabel("Number of Epoch", **hfont,fontsize=24)
plt.ylabel("Loss",fontsize=24)
plt.yticks(np.linspace(0,round(max(max(transfer_history.history["val_loss"]),max(transfer_history.history["loss"])),1)*1.2,num=5),fontsize=18)
plt.xticks(fontsize=18)
plt.legend(fontsize=16)
plt.show()
#

x_train=X_train
x_test= X_test

# Get train MAE loss.
x_train_pred = transfer_model.predict(x_train)
train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)

plt.hist(train_mae_loss, bins=50)
plt.xlabel("Train MAE loss")
plt.ylabel("No. of samples")
plt.show()

train_mae_loss=np.sum(train_mae_loss, axis=1)
train_mae_loss = train_mae_loss.reshape((-1))
train_mae_loss = train_mae_loss[50:-1,]
# Get reconstruction loss threshold.


threshold = np.max(train_mae_loss)*1+3*np.std(train_mae_loss) 


print("Reconstruction error threshold: ", threshold)


training_mean = x_train.mean()
training_std = x_train.std()

df_test_value = (X_test - training_mean) / training_std


print("Test input shape: ", x_test.shape)

# Get test MAE loss.
x_test_pred = model.predict(x_test)
test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)

#test_mae_loss= test_mae_loss[:,1]+test_mae_loss[:,0]+test_mae_loss[:,2]+test_mae_loss[:,3]++test_mae_loss[:,4]
test_mae_loss=np.sum(test_mae_loss, axis=1)
test_mae_loss = test_mae_loss.reshape((-1))

plt.hist(test_mae_loss, bins=50)
plt.xlabel("test MAE loss")
plt.ylabel("No of samples")
plt.show()

# Detect all the samples which are anomalies.
anomalies=[]
anomalies = test_mae_loss > threshold
print("Number of anomaly samples: ", np.sum(anomalies))
print("Indices of anomaly samples: ", np.where(anomalies))
#%
# data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
anomalous_data_indices_posttrain = []

for data_idx in range(TIME_STEPS - 1, len(df_test_value) - TIME_STEPS + 1):
    if np.all(anomalies[data_idx - TIME_STEPS + 1 : data_idx]):
        anomalous_data_indices_posttrain.append(data_idx)

indx_posttrain=anomalous_data_indices_posttrain+np.ones(len(anomalous_data_indices_posttrain))*y_train.shape[0]
# labels_posttrain=np.zeros(len(velocity_ldv2))
# labels_posttrain[indx_posttrain.astype(int)]=1

# posttrain_cm = confusion_matrix(labels_fact, labels_posttrain)

# posttrain_report = classification_report(labels_fact, labels_posttrain)

# print(posttrain_report)

#%%



predicted_labels = (test_mae_loss > threshold).astype(int)

#%
fs=1250000
mae_loss=np.concatenate([train_mae_loss,test_mae_loss],axis=0)
time_mae=np.linspace(0,time[-1],num=len(mae_loss))

fig1, ax1 = plt.subplots()
ax1.plot(time_mae,mae_loss,c='k')
plt.xlabel("Time(s)",fontsize=16,fontname='Times New Roman')
plt.ylabel("MAE Loss",fontsize=16,fontname='Times New Roman')
plt.xlim(xmin=min(time), xmax=max(time))

fig1.set_figwidth(12)
fig1.set_figheight(4)
ax1.axhline(threshold, xmin=0.0, xmax=1.0, color='k',linestyle='--',linewidth=3)
timearray=np.append(np.arange(0, time[-1], step=0.5),round(time[-1],1))
plt.xticks(timearray,fontsize=12) 
#ax1.plot(velocity_ldv2.iloc[indx],color='r')




fig1, ax1 = plt.subplots()
ax1.plot(velocity_ldv2)
ax1.plot(velocity_ldv2.iloc[indx_posttrain],color='r')
plt.xlabel("data points")
plt.ylabel("velocity(m/s)")

fig2, ax2 = plt.subplots()
ax2.plot(data['energy'])
ax2.plot(data['energy'].iloc[indx_posttrain],color='r')
plt.xlabel("data points")
#%%

anomalous_data_indices=[]
for data_idx in range(TIME_STEPS - 1, len(df_test_value) - TIME_STEPS + 1):
    if np.all(anomalies[data_idx - TIME_STEPS + 1 : data_idx]):
        anomalous_data_indices.append(data_idx)

indx=anomalous_data_indices+np.ones(len(anomalous_data_indices))*y_train.shape[0]




# read text file into pandas DataFrame
ldv2_acc_org= pd.read_csv('LDV2_vel_original_'+Dataset+'_tstep'+tstep+'.txt', sep=" ")
ldv1_acc_org= pd.read_csv('LDV1_vel_original_'+Dataset+'_tstep'+tstep+'.txt', sep=" ")

fs=1250000
# display DataFrame
print(ldv2_acc_org)


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


indx_sep=separate_array(indx)
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
plt.xlim(xmin=min(time), xmax=max(time))
timearray=np.append(np.arange(0, time[-1], step=0.5),round(time[-1],1))
plt.xticks(timearray,fontsize=12) 
#plt.grid(visible=None)
fig2.set_figwidth(12)
fig2.set_figheight(4)

ldv2_acc_org = ldv2_acc_org.to_numpy()
lim_array=max(ldv2_acc_org)*1.5
lim=round(lim_array[0],2)
#lim=0.06
ax2.set_ylim([-1*lim,lim])
ax2.set_yticks(np.linspace(-1*lim,lim,num=5))
plt.yticks(fontsize=12)
ax2.grid(False)
ax2.legend(["LDV2","LDV1","LDV2_anomaly","LDV1_anomaly"],loc='upper right', fancybox=True, framealpha=0.1,ncol=4)




#%%

# =============================================================================
# transfer_model.trainable = True
# 
# # It's important to recompile your model after you make any changes
# # to the `trainable` attribute of any inner layer, so that your changes
# # are take into account
# transfer_model.compile(optimizer=keras.optimizers.Adam(1e-5),  # Very low learning rate
#               loss='mae',
#               metrics=[keras.metrics.BinaryAccuracy()])
# 
# # Train end-to-end. Be careful to stop before you overfit!
# transfer_model.fit(X_train, X_train,epochs=10, batch_size=36, validation_split=0.1)
# =============================================================================
#%%

#load test dataset
Dataset='KKK13'
tstep='0001' #for KKK13, tstep=0.01, KKK14,15,16, tstep=0.008, KKK17, tstep=0.005

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
    data[col]=data[col].rolling(window=350,center=False,min_periods=1).mean() #300 for KKK13 70 for 14 15 16 , 50for 17



X_train1, X_test1, y_train1, y_test1 = train_test_split(data[float_columns], data['label'], test_size = 0.4, shuffle = False)
data=data.drop(['label'],1)




# =============================================================================
# mms = MinMaxScaler()
# ss = StandardScaler()
# 
# ssmean=pd.DataFrame()
# ssscale= pd.DataFrame()
# ssmean.rename(columns=data.columns)
# ssscale.rename(columns=data.columns)
# 
# for col in float_columns:
#     X_train1[col] = ss.fit_transform(X_train1[[col]]).squeeze()
#     ssmean[col]=ss.mean_
#     ssscale[col]=ss.scale_
#     X_test1[col] = ss.transform(X_test1[[col]]).squeeze()
#     data[col]=ss.transform(data[[col]]).squeeze()
# =============================================================================


ss = StandardScaler()
ssmean=pd.DataFrame()
ssscale= pd.DataFrame()
ssmean.rename(columns=data.columns)
ssscale.rename(columns=data.columns)

for col in float_columns:
    X_train1[col] = ss.fit_transform(X_train1[[col]]).squeeze()
    ssmean[col]=ss.mean_
    ssscale[col]=ss.scale_
    X_test1[col] = ss.transform(X_test1[[col]]).squeeze()
    data[col]=ss.transform(data[[col]]).squeeze()



pca_columns=pd.Series(['std','fe','energy','rms','kurt']) #for shape for 14,15,16

data=data[pca_columns]
X_train1=X_train1[pca_columns]
X_test1=X_test1[pca_columns]

TIME_STEPS=20

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
X_train, y_train = seq_gen.create_sequences(X_train1, y_train1)
X_test, y_test = seq_gen.create_sequences(X_test1, y_test1)
#%%


# Get train MAE loss.
train_mae_loss = train_mae_loss[50:-1,]

# Get reconstruction loss threshold.
#threshold = np.max(train_mae_loss[:,0]+train_mae_loss[:,1]+train_mae_loss[:,2]+train_mae_loss[:,3]+train_mae_loss[:,4])
#threshold = np.max(train_mae_loss)+30*np.std(np.abs(x_train_pred - x_train))
threshold = np.max(train_mae_loss)*1+3*np.std(train_mae_loss)  #1.5 for KKK13, 2.1-KKK15,3.6for KKK17


# Get test MAE loss.
x_test_pred = transfer_model.predict(data_feature)
test_mae_loss = np.mean(np.abs(x_test_pred - data_feature), axis=1)

#test_mae_loss= test_mae_loss[:,1]+test_mae_loss[:,0]+test_mae_loss[:,2]+test_mae_loss[:,3]++test_mae_loss[:,4]
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
anomalous_data_indices = []

for data_idx in range(TIME_STEPS - 1, len(data_feature) - TIME_STEPS + 1):
    if np.all(anomalies[data_idx - TIME_STEPS + 1 : data_idx]):
        anomalous_data_indices.append(data_idx)
        


velocity_ldv2=pd.read_excel('LDV2_downsample_'+Dataset+'_tstep'+tstep+'.xls',header=None)
#velocity_ldv2=pd.read_excel('LDV2_downsample_KKK14_tstep00008.xls',header=None)


indx=anomalous_data_indices

fig1, ax1 = plt.subplots()
ax1.plot(velocity_ldv2)
ax1.plot(velocity_ldv2.iloc[indx],color='r')
plt.xlabel("data points")
plt.ylabel("velocity(m/s)")

fig2, ax2 = plt.subplots()
ax2.plot(data['energy'])
ax2.plot(data['energy'].iloc[indx],color='r')
plt.xlabel("data points")



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
timearray=np.append(np.arange(0, time[-1], step=0.5),round(time[-1],1))
plt.xticks(timearray,fontsize=12) 
#ax1.plot(velocity_ldv2.iloc[indx],color='r')
