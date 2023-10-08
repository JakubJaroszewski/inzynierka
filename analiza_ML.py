from sklearn import tree
from sklearn.tree import export_graphviz
from io import StringIO 
from IPython.display import Image  
import math
import numpy as np
import matplotlib.pyplot as plt
import statistics
from sklearn.tree import export_text
# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import hrvanalysis as hrv
import warnings
warnings.filterwarnings("ignore")
import random

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
  meanRR = round(sum(NN_heart[i]) / len(NN_heart[i]), 2)
  SDNN_val = round(SDNN(NN_heart[i], meanRR), 2)
  RMSSD_val = round(RMSSD(NN_heart[i]), 2)
  pNN50_val = round(pNN50(NN_heart[i]) * 100, 2)
  SDANN_val = segmentation(NN_heart[i])
  if any(math.isnan(x) for x in [meanRR, SDNN_val, RMSSD_val, pNN50_val, SDANN_val]):
      continue
  meanRR_list_heart.append(meanRR)
  SDNN_list_heart.append(SDNN_val)
  RMSSD_list_heart.append(RMSSD_val)
  pNN50_list_heart.append(pNN50_val)
  SDANN_list_heart.append(SDANN_val)



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
  meanRR = round(sum(NN_heart_F[i]) / len(NN_heart_F[i]), 2)
  SDNN_val = round(SDNN(NN_heart_F[i], meanRR), 2)
  RMSSD_val = round(RMSSD(NN_heart_F[i]), 2)
  pNN50_val = round(pNN50(NN_heart_F[i]) * 100, 2)
  SDANN_val = segmentation(NN_heart_F[i])
  if any(math.isnan(x) for x in [meanRR, SDNN_val, RMSSD_val, pNN50_val, SDANN_val]):
    continue
  meanRR_list_heart_failure.append(meanRR)
  SDNN_list_heart_failure.append(SDNN_val)
  RMSSD_list_heart_failure.append(RMSSD_val)
  pNN50_list_heart_failure.append(pNN50_val)
  SDANN_list_heart_failure.append(SDANN_val)


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
label1=np.zeros(len(pNN50_list+pNN50_list_heart+pNN50_list_heart_failure+pNN50_list_heart_AF)-len(data_array_heart))
label= np.concatenate((label0, label1))


data = {
   'Średnie RR': meanRR_list+meanRR_list_heart+meanRR_list_heart_failure+meanRR_list_heart_AF,
   'SDNN' : SDNN_list+SDNN_list_heart+SDNN_list_heart_failure+SDNN_list_heart_AF,
   'RMSSD' : RMSSD_list+RMSSD_list_heart+RMSSD_list_heart_failure+RMSSD_list_heart_AF,
   'pNN50' : pNN50_list+pNN50_list_heart+pNN50_list_heart_failure+pNN50_list_heart_AF,
   'zdrowy' : label
}
import pandas as pd
table = pd.DataFrame(data)

counterSVM = 0
counterLR = 0
counterDT = 0
counterRF = 0
counterXGB = 0

AccXGB_arr=[]
AccDT_arr=[]
AccLR_arr=[]
AccRF_arr=[]
AccSVM_arr=[]

for i in range(100):
#SVM
  from sklearn.model_selection import train_test_split
  from sklearn import svm
  from sklearn import metrics
  X1 = table.drop('zdrowy', axis=1)
  y1 = table['zdrowy']
  X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.3)
  clf = svm.LinearSVC(loss='hinge',C=1)  
  clf.fit(X_train1, y_train1)
  y_pred = clf.predict(X_test1)
  AccSVM = round(metrics.accuracy_score(y_test1, y_pred)*100,2)
  AccSVM_arr.append(AccSVM)
  print("SVM: Accuracy:",AccSVM,'%')


  #Logistic Regresion
  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import accuracy_score, mean_squared_error
  feature_cols = ['Średnie RR', 'SDNN', 'RMSSD', 'pNN50']
  X = table[feature_cols]
  y = table.zdrowy 
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
  logreg = LogisticRegression()
  logreg.fit(X_train, y_train)
  y_pred = logreg.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  AccLR = round(accuracy*100,2)
  AccLR_arr.append(AccLR)
  print("Logistic Regresion: Accuracy:",AccLR ,"%")

  # Wykres
  # y_pred_proba = logreg.predict_proba(X_test)[::,1]
  # fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
  # auc = metrics.roc_auc_score(y_test, y_pred_proba)
  # plt.plot(fpr,tpr,label="data 1, auc="+str(round(auc,3)))
  # plt.title("Receiver Operating Characteristic curve")
  # plt.xlabel("False positvie")
  # plt.ylabel("True positive")
  # plt.legend(loc=4)
  # plt.savefig("ROC_curve_LR.png")


  #Decision Tree
  feature_cols = ['Średnie RR', 'SDNN', 'RMSSD', 'pNN50']
  X = table[feature_cols] 
  y = table.zdrowy 
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
  clf = DecisionTreeClassifier(max_depth= random.randint(2, 16))
  clf = clf.fit(X_train,y_train)
  y_pred = clf.predict(X_test)
  AccDT = round(accuracy_score(y_test, y_pred)*100,2)
  AccDT_arr.append(AccDT)
  print("Decition Tree: Accuracy:",AccDT, "%")
  #tree_rules = export_text(clf, feature_names=feature_cols, show_weights=True)
  #fig, ax = plt.subplots(figsize=(12, 12))
  #tree.plot_tree(clf, feature_names=feature_cols, class_names=['chorzy', 'zdrowi'], filled=True, rounded=True, ax=ax)
  #plt.savefig('DecisionTreeView2.pdf', format='pdf')
  #plt.show()


  #Random Forest
  from sklearn.ensemble import RandomForestClassifier
  X = table.drop('zdrowy', axis=1)
  y = table['zdrowy']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
  rf = RandomForestClassifier(n_estimators=87,max_depth= 2)
  rf.fit(X_train, y_train)
  y_pred = rf.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  AccRF = round(accuracy,4)*100
  AccRF_arr.append(AccRF)
  print("Random Forest: Accuracy:",AccRF , "%")


  # XGBOOST
  from xgboost import XGBClassifier
  import xgboost as xgb
  X = table.drop('zdrowy', axis=1) 
  y = data['zdrowy']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
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
  AccXGB = round(accuracy * 100.0, 2)
  AccXGB_arr.append(AccXGB)
  print("XGBoost: Accuracy:" ,AccXGB, "%")

  max_value = max(AccXGB,AccDT,AccLR,AccRF,AccSVM)
  print(max_value)

  
  if max_value == AccLR:
    counterLR +=1
    continue
  if max_value == AccRF:
    counterRF +=1
    continue
  if max_value == AccSVM:
    counterXGB +=1
    continue
  if max_value == AccXGB:
    counterXGB +=1
    continue
  if max_value == AccDT:
    counterDT+=1
    continue
print(counterSVM)
print(counterLR)
print(counterDT)
print(counterRF)
print(counterXGB)
labels = ["SVM", "LR", "DT", "RF", "XGB"]
values = [counterSVM, counterLR, counterDT, counterRF, counterXGB]
colors = ['blue', 'c', 'green', 'purple', 'orange']
bars = plt.bar(labels, values, color = colors,edgecolor='black')
plt.xlabel("Algorytm")
plt.ylabel("Liczba najlepszych dokładności")
plt.title("Histogram najlepszych algorytmów")

labels = ["SVM - Support Vector Machine", "LR - Logistic Regresion" ,  " DT - Decision Tree"," RF - Random Forest ",  "XGB - XGboost" ]
handles = [bar for bar in bars] 
plt.legend(handles, labels, loc='upper left')
# plt.savefig("Historgam_najlepszych_algorytmów1.png")
plt.show()


fig, ax = plt.subplots(figsize=(10, 7))
ax.boxplot([AccXGB_arr, AccDT_arr, AccRF_arr, AccLR_arr, AccSVM_arr], labels=['XGB', 'DT', 'RF', 'LR', 'SVM'],showfliers=False)
ax.set_title("Box Plot'y dla danych z ML")
plt.xlabel("Algorytm")
plt.ylabel("Dokładność [%]")
# plt.savefig("Wykres_Boxplot.png")
plt.show()








