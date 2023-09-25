from sklearn import tree
from sklearn.tree import export_graphviz
from io import StringIO 
from IPython.display import Image  
import math
import numpy as np
import matplotlib.pyplot as plt
import statistics
from sklearn.tree import export_text
import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
import hrvanalysis as hrv
import warnings
warnings.filterwarnings("ignore")

def read_data_from_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            value = float(line.strip())
            data.append(value)
    return data
def SDNN(arr,mean):
    sum=0
    for i in arr:
        sum+=(i-mean)**2 
    return np.sqrt(sum/len(arr))  
def RMSSD(arr):
    arr1=[]
    for i in range(len(arr)-1):
        arr1.append((arr[i+1]-arr[i])**2)
    return math.sqrt(sum(arr1)/len(arr1))  
def pNN50(arr):
    pr=0
    for i in range(len(arr)-1):
        if abs(arr[i+1]-arr[i]) > 50 :
            pr+=1
    return float(pr/len(arr))
def segmentation(arr):
    segmented_rr_intervals = []
    interval_length = 302 # interwał co 302, przecidziały +/- 5 min 
    start_index = 0
    end_index = interval_length
    while end_index <= len(arr):
       segment = arr[start_index:end_index]
       segmented_rr_intervals.append(segment)
       start_index += interval_length
       end_index += interval_length
    for i in range(int(len(arr)/interval_length)):
        segmented_rr_intervals[i]=round(np.std(segmented_rr_intervals[i]),2)
    return np.mean(segmented_rr_intervals)
def above_line(tab,tab1):
    filtered_x=[]
    filtered_y=[]
    for i in range(len(tab)):
        if tab[i]<tab1[i]:
            filtered_x.append(tab[i])
            filtered_y.append(tab1[i])
    return filtered_x,filtered_y
def SD1(tab):
    sum=0
    for i in range(len(tab)-1):
        sum+=((tab[i]-tab[i+1])**2)
    return np.sqrt(sum/(2*(len(tab)-1)))
def SD2(tab,k):
    sum=0
    for i in range(len(tab)-1):
        sum+=((tab[i]+tab[i+1]-2*k)**2)
    return np.sqrt(sum/(2*(len(tab)-1)))
def dist_from_line(tab,tab1):
    sum=0
    for i in range(len(tab)):
        sum+=np.abs(-1*tab[i]+tab1[i])/np.sqrt(2)
    return sum
def max_diff(data,data1):
    arr=[]
    for k in range(len(data)):
        arr.append(abs(data[k]-data1[k]))
    return max(arr)
def heaviside(x):
    return 1 if x >= 0 else 0
def AlphaV(k,var,m,N,data):
    alpha=0
    for i in range(N-m+1):
        for j in range(N-m+1):
            if i != j:
                alpha+=heaviside(k*var-max_diff(data[i:i+m],data[j:j+m]))
    return alpha
def ApEN(m,N,data,var):
    k=0.2 # przyjete 20 %
    return np.log(((N-m-2)/(N-m-1))*AlphaV(k,var,m,N,data)/AlphaV(k,var,m+1,N,data))   
def count_elements_in_bins(data, bins):
    hist, _ = np.histogram(data, bins=np.append(bins, bins[-1]))
    return hist    
data_array=[]
for i in range(2,48,1):
  file_path='./data_zdrowi/' + str(i)+ '.txt'
  data_array.append(read_data_from_file(file_path))

data_array_NN=[]
for i in range(len(data_array)):
  data_array_NN.append(hrv.get_nn_intervals(data_array[i]))

meanRR_list=[]
SDNN_list=[]
RMSSD_list=[]
pNN50_list=[]
SDANN_list=[]
for i in range(46):
  meanRR_list.append(round(sum(data_array_NN[i])/len(data_array_NN[i]),2))
  SDNN_list.append(round(SDNN(data_array_NN[i],meanRR_list[i]),2)) 
  RMSSD_list.append(round(RMSSD(data_array_NN[i]),2))
  pNN50_list.append(round(pNN50(data_array_NN[i])*100,2))
  SDANN_list.append(segmentation(data_array_NN[i]))

data_array_heart=[]
for i in range(2,29,1):
  if i == 26 or i == 25 or i ==18:
      continue
  file_path='./data_zastoinowa_niewydolnosc_serca/rr'+ '.txt.'+str(i)
  data_array_heart.append(read_data_from_file(file_path))

NN_heart =[]
for i in range(len(data_array_heart)):
  NN_heart.append(hrv.get_nn_intervals(data_array_heart[i]))

meanRR_list_heart=[]
SDNN_list_heart=[]
RMSSD_list_heart=[]
pNN50_list_heart=[]
SDANN_list_heart=[]

for i in range(24):
  meanRR_list_heart.append(round(sum(NN_heart[i])/len(NN_heart[i]),2))
  SDNN_list_heart.append(round(SDNN(NN_heart[i],meanRR_list_heart[i]),2)) 
  RMSSD_list_heart.append(round(RMSSD(NN_heart[i]),2))
  pNN50_list_heart.append(round(pNN50(NN_heart[i])*100,2))
  SDANN_list_heart.append(segmentation(NN_heart[i]))


data_array_heart_failure=[]
for i in range(0,23,1):
  if i == 0:
    file_path='./data_nagla_smierc/rr'+ '.txt.'#str(i)
  else:
    file_path='./data_nagla_smierc/rr'+ '.txt.'+str(i)
  data_array_heart_failure.append(read_data_from_file(file_path))

NN_heart_F =[]
for i in range(len(data_array_heart_failure)):
  NN_heart_F.append(hrv.get_nn_intervals(data_array_heart_failure[i]))


meanRR_list_heart_failure=[]
SDNN_list_heart_failure=[]
RMSSD_list_heart_failure=[]
pNN50_list_heart_failure=[]
SDANN_list_heart_failure=[]

for i in range(23):
  meanRR_list_heart_failure.append(round(sum(NN_heart_F[i])/len(NN_heart_F[i]),2))
  SDNN_list_heart_failure.append(round(SDNN(NN_heart_F[i],meanRR_list_heart_failure[i]),2)) 
  RMSSD_list_heart_failure.append(round(RMSSD(NN_heart_F[i]),2))
  pNN50_list_heart_failure.append(round(pNN50(NN_heart_F[i])*100,2))
  SDANN_list_heart_failure.append(segmentation(NN_heart_F[i]))

data_array_heart_AF=[]
for i in range(0,25,1):
  if i == 0:
    file_path='./data_migotanie/rr'+ '.txt.'#str(i)
  else:
    file_path='./data_migotanie/rr'+ '.txt.'+str(i)
  data_array_heart_AF.append(read_data_from_file(file_path))

NN_heart_AF =[]
for i in range(len(data_array_heart_AF)):
  NN_heart_AF.append(hrv.get_nn_intervals(data_array_heart_AF[i]))


meanRR_list_heart_AF=[]
SDNN_list_heart_AF=[]
RMSSD_list_heart_AF=[]
pNN50_list_heart_AF=[]
SDANN_list_heart_AF=[]




for i in range(len(NN_heart_AF)):
  meanRR_list_heart_AF.append(round(sum(NN_heart_AF[i])/len(NN_heart_AF[i]),2))
  SDNN_list_heart_AF.append(round(SDNN(NN_heart_AF[i],meanRR_list_heart_AF[i]),2)) 
  RMSSD_list_heart_AF.append(round(RMSSD(NN_heart_AF[i]),2))
  pNN50_list_heart_AF.append(round(pNN50(NN_heart_AF[i])*100,2))
  SDANN_list_heart_AF.append(segmentation(NN_heart_AF[i]))

# lista mówiąca o tym czy ktos zalicza sie do zdrowych czy do chorych 
label0 = np.ones(len(data_array_heart))
label1=np.zeros(118-len(data_array_heart))
label= np.concatenate((label0, label1))

data = {
   'Średnie RR': meanRR_list+meanRR_list_heart+meanRR_list_heart_failure+meanRR_list_heart_AF,
   'SDNN' : SDNN_list+SDNN_list_heart+SDNN_list_heart_failure+SDNN_list_heart_AF,
   'RMSSD' : RMSSD_list+RMSSD_list_heart+RMSSD_list_heart_failure+RMSSD_list_heart_AF,
   'pNN50' : pNN50_list+pNN50_list_heart+pNN50_list_heart_failure+pNN50_list_heart_AF,
   'zdrowy' : label
}
import pandas as pd
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, mean_squared_error
table = pd.DataFrame(data)
from sklearn.model_selection import GridSearchCV


X = table.drop('zdrowy', axis=1) 
y = data['zdrowy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# model = XGBClassifier()
# model.fit(X_train, y_train)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

parametry = {
     'objective': 'binary:logistic',  
      'n_estimators': 100,
      'max_depth': 2,
      'learning_rate': 0.01,
      'random_state': 42
  }
model = xgb.train(parametry, dtrain)

y_pred = model.predict(dtest)

predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# # Create regression matrices
# dtrain = xgb.DMatrix(X_train, y_train, enable_categorical=True)
# dtest = xgb.DMatrix(X_test, y_test, enable_categorical=True)

# parametry = {
#     'objective': 'binary:logistic',  # Dla klasyfikacji binarnej
#     'n_estimators': 100,
#     'max_depth': 3,
#     'learning_rate': 0.1,
#     'random_state': 42
# }

# 
# model = xgb.train(parametry, dtrain)

# 
# y_pred = model.predict(dtest)

# 
# y_pred_binary = (y_pred >= 0.5).astype(int)
# print(y_test)
# print(y_pred)
# # Ocena modelu
# accuracy = accuracy_score(y_test, y_pred_binary)
# print(f'Accuracy: {accuracy}')
















# RMSSD_list_heart = [x for x in RMSSD_list_heart if not math.isnan(x)]
# SDNN_list_heart = [x for x in SDNN_list_heart if not math.isnan(x)]

# RMSSD_list_heart_failure = [x for x in RMSSD_list_heart_failure if not math.isnan(x)]
# SDNN_list_heart_failure = [x for x in SDNN_list_heart_failure if not math.isnan(x)]

# print("Zdrowi", np.mean(RMSSD_list), np.mean(SDNN_list), np.mean(pNN50_list))
# print("CHF",np.mean(RMSSD_list_heart), np.mean(SDNN_list_heart), np.mean(pNN50_list_heart))
# print("AF" ,np.mean(RMSSD_list_heart_AF), np.mean(SDNN_list_heart_AF), np.mean(pNN50_list_heart_AF))
# print("nagła smierc", np.mean(RMSSD_list_heart_failure), np.mean(SDNN_list_heart_failure), np.mean(pNN50_list_heart_failure))
 