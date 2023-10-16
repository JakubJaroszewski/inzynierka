from sklearn import tree
from sklearn.tree import export_graphviz
from io import StringIO 
from IPython.display import Image  
import pydotplus
import math
import numpy as np
import matplotlib.pyplot as plt
import statistics
import graphviz
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics 

import warnings
warnings.filterwarnings("ignore")

def read_data_from_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            value = float(line.strip())
            data.append(value)
    return data
def SDNN(arr):
    sum=0
    mean=np.mean(arr)
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
import hrvanalysis as hrv
for i in range(len(data_array)):
  a=[]
  a=hrv.get_nn_intervals(data_array[i])
  if np.isnan(a).any():
     continue
  data_array_NN.append(a)

zdrowy=1
chory=0
meanRR_list=[]
SDNN_list=[]
RMSSD_list=[]
pNN50_list=[]
SDANN_list=[]
zdrowy_list=[]
for i in range(len(data_array_NN)):
  if np.isnan(data_array_NN[i]).any():
     continue
  meanRR_list.append(round(sum(data_array_NN[i])/len(data_array_NN[i]),2))
  SDNN_list.append(round(SDNN(data_array_NN[i]),2)) 
  RMSSD_list.append(round(RMSSD(data_array_NN[i]),2))
  pNN50_list.append(round(pNN50(data_array_NN[i])*100,2))
  SDANN_list.append(segmentation(data_array_NN[i]))
  zdrowy_list.append(zdrowy)

data_array_heart=[]
for i in range(2,29,1):
  if i == 26 or i == 25 or i ==18:
      continue
  file_path='./data_zastoinowa_niewydolnosc_serca/rr'+ '.txt.'+str(i)
  data_array_heart.append(read_data_from_file(file_path))

NN_heart =[]
for i in range(len(data_array_heart)):
  NN_heart.append(hrv.get_nn_intervals(data_array_heart[i]))

for i in range(24):
  if np.isnan(NN_heart[i]).any():
     continue
  meanRR_list.append(round(sum(NN_heart[i])/len(NN_heart[i]),2))
  SDNN_list.append(round(SDNN(NN_heart[i]),2)) 
  RMSSD_list.append(round(RMSSD(NN_heart[i]),2))
  pNN50_list.append(round(pNN50(NN_heart[i])*100,2))
  SDANN_list.append(segmentation(NN_heart[i]))
  zdrowy_list.append(chory)


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

for i in range(23):
  if np.isnan(NN_heart_F[i]).any():
     continue
  meanRR_list.append(round(sum(NN_heart_F[i])/len(NN_heart_F[i]),2))
  SDNN_list.append(round(SDNN(NN_heart_F[i]),2)) 
  RMSSD_list.append(round(RMSSD(NN_heart_F[i]),2))
  pNN50_list.append(round(pNN50(NN_heart_F[i])*100,2))
  SDANN_list.append(segmentation(NN_heart_F[i]))
  zdrowy_list.append(chory)

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

for i in range(len(NN_heart_AF)):
  if np.isnan(NN_heart_AF[i]).any():
     continue
  meanRR_list.append(round(sum(NN_heart_AF[i])/len(NN_heart_AF[i]),2))
  SDNN_list.append(round(SDNN(NN_heart_AF[i]),2)) 
  RMSSD_list.append(round(RMSSD(NN_heart_AF[i]),2))
  pNN50_list.append(round(pNN50(NN_heart_AF[i])*100,2))
  SDANN_list.append(segmentation(NN_heart_AF[i]))
  zdrowy_list.append(chory)
data = {
   'Średnie RR': meanRR_list,
   'SDNN' : SDNN_list,
   'RMSSD' : RMSSD_list,
   'pNN50' : pNN50_list,
   'zdrowy' : zdrowy_list
}
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
table = pd.DataFrame(data)


X = table.drop('zdrowy', axis=1)
y = table['zdrowy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y) 
clf = svm.LinearSVC(loss='hinge',C=1)  
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
TN=cm[0][0]
FP=cm[0][1]
FN=cm[1][0]
TP=cm[1][1]
print("Precyzja:" , TP/(TP+FP))
print("NPV:" , TN/(FN+TN))
print("czułość:" , TP/(TP+FN))
print("swoistość:", TN/(FP+TN))
print("Dokładność:", (TP+TN)/(TN+TP+FN+FP))
plt.xlabel("Klasa predykcji")
plt.ylabel("Klasa rzeczywista")
plt.title("Tablica pomyłek SVM")
plt.savefig('./MacierzePomyłek/confusion_matrix_plot_SVM.png')
plt.show()

data = {
    "": [
        "Precyzja [%]",
        "NPV [%]" ,
        "Czułość [%]",
        "Swoistość [%]",
        "Dokładność [%]"
    ],
    "Wyniki": [
        round(TP/(TP+FP)*100,2),
        round(TN/(FN+TN)*100,2),
        round(TP/(TP+FN)*100,2),
        round(TN/(FP+TN)*100,2),
        round((TP+TN)/(TN+TP+FN+FP)*100,2)
    ]
}
import pandas as pd
df1 = pd.DataFrame(data)
from matplotlib.backends.backend_pdf import PdfPages   
fig, ax =plt.subplots(figsize=(8,4))
ax.axis('tight')
ax.axis('off')
kolorowa_macierz = [['lightgray', 'lightgray'],
['white', 'white' ],
['lightgray', 'lightgray' ],
['white', 'white' ],
['lightgray', 'lightgray' ]]
the_table = ax.table(cellText=df1.values,colLabels=df1.columns,loc='center',cellColours=kolorowa_macierz, cellLoc= 'left' )
plt.title('SVM wartości macierzy pomyłek', pad=-30)
pp = PdfPages("./MacierzePomyłek/SVM_values.pdf")
pp.savefig(fig, bbox_inches='tight')
pp.close()