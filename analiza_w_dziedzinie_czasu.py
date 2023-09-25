import math
import numpy as np
import matplotlib.pyplot as plt
import statistics
from matplotlib.patches import Ellipse
# from hrvanalysis import get_frequency_domain_features
# from hrvanalysis import get_nn_intervals
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
def SampEN(m,N,data,var):
    k=0.2 # przyjete 20 %
    return np.log(((N-m-2)/(N-m-1))*AlphaV(k,var,m,N,data)/AlphaV(k,var,m+1,N,data))   
def count_elements_in_bins(data, bins):
    hist, _ = np.histogram(data, bins=np.append(bins, bins[-1]))
    return hist    
data_array=[]
for i in range(2,48,1):
  file_path='./data_zdrowi/' + str(i)+ '.txt'
  data_array.append(read_data_from_file(file_path))
meanRR_list=[]
SDNN_list=[]
RMSSD_list=[]
pNN50_list=[]
SDANN_list=[]
for i in range(46):
  meanRR_list.append(round(sum(data_array[i])/len(data_array[i]),2))
  SDNN_list.append(round(SDNN(data_array[i],meanRR_list[i]),2)) 
  RMSSD_list.append(round(RMSSD(data_array[i]),2))
  pNN50_list.append(round(pNN50(data_array[i])*100,2))
  SDANN_list.append(segmentation(data_array[i]))

# gg=get_nn_intervals(read_data_from_file('chf201.txt_N.csv'))
# print(gg)
# print(get_frequency_domain_features(gg, method='welch'))


# LICZENIE UDARU
# data_array_ill=[]
# for i in range(2,43,1):
#   file_path='./data_udar_niedok_mozgu/' + str(i)+ '.txt'
#   data_array_ill.append(read_data_from_file(file_path))
# meanRR_list_ill=[]
# SDNN_list_ill=[]
# RMSSD_list_ill=[]
# pNN50_list_ill=[]
# SDANN_list_ill=[]
# for i in range(41):
#   meanRR_list_ill.append(round(sum(data_array_ill[i])/len(data_array_ill[i]),2))
#   SDNN_list_ill.append(round(SDNN(data_array_ill[i],meanRR_list_ill[i]),2)) 
#   RMSSD_list_ill.append(round(RMSSD(data_array_ill[i]),2))
#   pNN50_list_ill.append(round(pNN50(data_array_ill[i])*100,2))
#   SDANN_list_ill.append(segmentation(data_array_ill[i]))
# KONIEC LICZNEIA


data_array_heart=[]
for i in range(2,29,1):
  if i == 26 or i == 25 or i ==18:
      continue
  file_path='./data_zastoinowa_niewydolnosc_serca/rr'+ '.txt.'+str(i)
  data_array_heart.append(read_data_from_file(file_path))
meanRR_list_heart=[]
SDNN_list_heart=[]
RMSSD_list_heart=[]
pNN50_list_heart=[]
SDANN_list_heart=[]

for i in range(24):
  meanRR_list_heart.append(round(sum(data_array_heart[i])/len(data_array_heart[i]),2))
  SDNN_list_heart.append(round(SDNN(data_array_heart[i],meanRR_list_heart[i]),2)) 
  RMSSD_list_heart.append(round(RMSSD(data_array_heart[i]),2))
  pNN50_list_heart.append(round(pNN50(data_array_heart[i])*100,2))
  SDANN_list_heart.append(segmentation(data_array_heart[i]))


data_array_heart_failure=[]
for i in range(0,23,1):
  if i == 0:
    file_path='./data_nagla_smierc/rr'+ '.txt.'#str(i)
  else:
    file_path='./data_nagla_smierc/rr'+ '.txt.'+str(i)
  data_array_heart_failure.append(read_data_from_file(file_path))
meanRR_list_heart_failure=[]
SDNN_list_heart_failure=[]
RMSSD_list_heart_failure=[]
pNN50_list_heart_failure=[]
SDANN_list_heart_failure=[]

for i in range(23):
  meanRR_list_heart_failure.append(round(sum(data_array_heart_failure[i])/len(data_array_heart_failure[i]),2))
  SDNN_list_heart_failure.append(round(SDNN(data_array_heart_failure[i],meanRR_list_heart_failure[i]),2)) 
  RMSSD_list_heart_failure.append(round(RMSSD(data_array_heart_failure[i]),2))
  pNN50_list_heart_failure.append(round(pNN50(data_array_heart_failure[i])*100,2))
  SDANN_list_heart_failure.append(segmentation(data_array_heart_failure[i]))

data_array_heart_AF=[]
for i in range(0,25,1):
  if i == 0:
    file_path='./data_migotanie/rr'+ '.txt.'#str(i)
  else:
    file_path='./data_migotanie/rr'+ '.txt.'+str(i)
  data_array_heart_AF.append(read_data_from_file(file_path))
meanRR_list_heart_AF=[]
SDNN_list_heart_AF=[]
RMSSD_list_heart_AF=[]
pNN50_list_heart_AF=[]
SDANN_list_heart_AF=[]

for i in range(25):
  meanRR_list_heart_AF.append(round(sum(data_array_heart_AF[i])/len(data_array_heart_AF[i]),2))
  SDNN_list_heart_AF.append(round(SDNN(data_array_heart_AF[i],meanRR_list_heart_AF[i]),2)) 
  RMSSD_list_heart_AF.append(round(RMSSD(data_array_heart_AF[i]),2))
  pNN50_list_heart_AF.append(round(pNN50(data_array_heart_AF[i])*100,2))
  SDANN_list_heart_AF.append(segmentation(data_array_heart_AF[i]))

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
table = pd.DataFrame(data)
print(table)

# Liczenie Entropi
# entropy=[]
# entropy_ill=[]
# entropy_heart=[]
# entropy_heart_fauliure=[]
# entropy_heart_AF=[]
# for i in range(len(data_array)):
#    entropy.append(SampEn(2,len(data_array[i]),data_array[i],RMSSD_list[i]))
# #for i in range(len(data_array_ill)):
# #   entropy_ill.append(SampEn(2,len(data_array_ill[i]),data_array_ill[i],RMSSD_list_ill[i]))
# for i in range(len(data_array_heart)):
#    entropy_heart.append(SampEn(2,len(data_array_heart[i]),data_array_heart[i],RMSSD_list_heart[i]))
# for i in range(len(data_array_heart_failure)):
#    entropy_heart_fauliure.append(SampEn(2,len(data_array_heart_failure[i]),data_array_heart_failure[i],RMSSD_list_heart_failure[i]))
# for i in range(len(data_array_heart_AF)):
#    entropy_heart_AF.append(SampEn(2,len(data_array_heart_AF[i]),data_array_heart_AF[i],RMSSD_list_heart_AF[i]))

# plt.figure(figsize=(12, 12))
# Tworzenie histogramu
# bins=[450,500,550,600,650,700,750,800,850,900,950,1000,1050,1100]
# plt.subplot(2, 2, 1)
# plt.hist(meanRR_list, bins=bins,label='Zdrowi')
# plt.hist(meanRR_list_ill,bins=bins, color='red', label='Udar niedokorwienny mózgu') nie używam danych z udaru
# plt.hist(meanRR_list_heart,bins=bins,color='green', label = 'niewydolność serca')
# plt.hist(meanRR_list_heart_failure,bins=bins,color='c', label = 'nagła śmierć')
# plt.hist(meanRR_list_heart_AF,bins=bins,color='magenta', label = 'migotanie przedsionków')
# plt.ylabel('Liczba')
# plt.legend()
# plt.savefig('Histogram_interwaly_RR1.png')
# plt.title('Histogram średniej interwałów RR w grupie 51 +/- 12 lat')

#bins1=np.linspace(10,100,18)
# plt.hist(SDNN_list, bins=bins1,label='Zdrowi')
# plt.plot((50, 50), (0, 10), color='k', label='średnia')
# plt.hist(SDNN_list_ill,bins=bins1,color='red', label='Udar niedokorwienny mózgu')
# plt.hist(SDNN_list_heart,bins=bins1,color='green', label='niewydolność serca')
# plt.hist(SDNN_list_heart_failure,bins=bins1,color='c', label='nagła śmierć')
# plt.hist(SDNN_list_heart_AF,bins=bins1,color='magenta', label = 'migotanie przedsionków')
# plt.ylabel('Liczba')
# plt.legend()
# plt.title('Histogram SDNN w grupie 51 +/- 12 lat')

# plt.hist(RMSSD_list, bins=bins2,label='Zdrowi')
# plt.plot((22, 22), (0, 10), color='k', label='średnia')
# plt.hist(RMSSD_list_ill,bins=bins2, color='red', label='Udar niedokorwienny mózgu')
# plt.hist(RMSSD_list_heart,bins=bins2, color='green', label='niewydolność serca')
# plt.hist(RMSSD_list_heart_failure,bins=bins2, color='c', label='nagła smierć')
# plt.hist(RMSSD_list_heart_AF,bins=bins2,color='magenta', label = 'migotanie przedsionków')
# plt.ylabel('Liczba')
# plt.legend()
# plt.title('Histogram RMSSD w grupie 51 +/- 12 lat')

# plt.hist(pNN50_list, bins=bins3,label='Zdrowi')
# plt.plot((10, 10), (0, 35), color='k', label='średnia')
# plt.hist(pNN50_list_ill,bins=bins3, color='red', label='Udar niedokorwienny mózgu')
# plt.hist(pNN50_list_heart,bins=bins3, color='green', label='niewydolność serca')
# plt.hist(pNN50_list_heart,bins=bins3, color='c', label='nagła smierć')
# plt.hist(pNN50_list_heart_AF,bins=bins3, color='magenta', label='migotanie przedsionków')
# plt.ylabel('Liczba')
# plt.legend()
# plt.title('Histogram pNN50 w grupie 51 +/- 12 lat')
# plt.show()


# HISTOGRAM SDNN 248-272
# plt.figure(figsize=(16, 9))# Tworzenie histogramu
# bins1=np.linspace(10,100,18)
# bins1=np.round(bins1,2)
# 
# SDNN_list1=count_elements_in_bins(SDNN_list, bins1)
# SDNN_list_heart1=count_elements_in_bins(SDNN_list_heart, bins1)
# SDNN_list_heart_failure1=count_elements_in_bins(SDNN_list_heart_failure, bins1)
# SDNN_list_heart_AF1=count_elements_in_bins(SDNN_list_heart_AF, bins1)
# SDNN_list_ill1=count_elements_in_bins(SDNN_list_ill,bins1)
# x_axis = np.arange(len(bins1))
# space=0.25                  
# 
# plt.bar(x_axis+0.25,SDNN_list1, width=space, label='Zdrowi',edgecolor='black')
##plt.bar(x_axis+0.2*2, SDNN_list_ill1 , width=space, color='red', label='Udar niedokrwienny mózgu', edgecolor='black')
# plt.bar(x_axis+0.25*2,SDNN_list_heart_AF1, width=space, color='m', label='migotanie przedsionów', edgecolor='black')
# plt.bar(x_axis + 0.25*3, SDNN_list_heart1 , width=space,  color='green', label='Niewydolność serca', edgecolor='black')
# plt.bar(x_axis + 0.25*4,SDNN_list_heart_failure1, width=space, color='c', label='Nagła śmierć', edgecolor='black')
# plt.ylabel('Liczba')
# plt.xticks(x_axis+0.1,bins1)
# plt.legend()
# plt.title('Histogram SDNN w grupie 51 +/- 12 lat')
# plt.tight_layout()
# plt.savefig('Histogram_SDNN1.png')
# plt.show()

# HISTOGRAM RMSSD 272-296
# plt.figure(figsize=(16, 9))# Tworzenie histogramu
# bins2=np.linspace(0,100,18)
# bins2=np.round(bins2,2)
# 
# RMSSD_list1=count_elements_in_bins(RMSSD_list,bins2)
# RMSSD_list_ill1=count_elements_in_bins(RMSSD_list_ill,bins2)
# RMSSD_list_heart1=count_elements_in_bins(RMSSD_list_heart,bins2)
# RMSSD_list_heart_failure1=count_elements_in_bins(RMSSD_list_heart_failure,bins2)
# RMSSD_list_heart_AF1=count_elements_in_bins(RMSSD_list_heart_AF,bins2)
# x_axis = np.arange(len(bins2))
# space=0.25    
# 
# plt.bar(x_axis+0.25,RMSSD_list1, width=space, label='Zdrowi',edgecolor='black')
##plt.bar(x_axis+0.2*2, RMSSD_list_ill1 , width=space, color='red', label='Udar niedokrwienny mózgu', edgecolor='black')
# plt.bar(x_axis+0.25*2,RMSSD_list_heart_AF1, width=space, color='m', label='migotanie przedsionów', edgecolor='black')
# plt.bar(x_axis + 0.25*3, RMSSD_list_heart1 , width=space,  color='green', label='Niewydolność serca', edgecolor='black')
# plt.bar(x_axis + 0.25*4,RMSSD_list_heart_failure1, width=space, color='c', label='Nagła śmierć', edgecolor='black')
# plt.ylabel('Liczba')
# plt.xticks(x_axis+0.1,bins2)
# plt.legend()
# plt.title('Histogram RMSSD w grupie 51 +/- 12 lat')
# plt.tight_layout()
# plt.savefig('Histogram_RMSSD1.png')
# plt.show()

# HISTOGRAM pNN50 299-325
# plt.figure(figsize=(16, 9))# Tworzenie histogramu
# bins3=np.linspace(0,40,8)
# bins3=np.round(bins3,2)
# 
# pNN50_list1=count_elements_in_bins(pNN50_list,bins3)
# pNN50_list_ill1=count_elements_in_bins(pNN50_list_ill,bins3)
# pNN50_list_heart1=count_elements_in_bins(pNN50_list_heart,bins3)
# pNN50_list_heart_failure1=count_elements_in_bins(pNN50_list_heart_failure,bins3)
# pNN50_list_heart_AF1=count_elements_in_bins(pNN50_list_heart_AF,bins3)
# x_axis = np.arange(len(bins3))
# space=0.25   
# 
# 
# plt.bar(x_axis+0.25,pNN50_list1, width=space, label='Zdrowi',edgecolor='black')
#plt.bar(x_axis+0.2*2, pNN50_list_ill1 , width=space, color='red', label='Udar niedokrwienny mózgu', edgecolor='black')
# plt.bar(x_axis+0.25*2,pNN50_list_heart_AF1, width=space, color='m', label='migotanie przedsionów', edgecolor='black')
# plt.bar(x_axis + 0.25*3, pNN50_list_heart1 , width=space,  color='green', label='Niewydolność serca', edgecolor='black')
# plt.bar(x_axis + 0.25*4,pNN50_list_heart_failure1, width=space, color='c', label='Nagła śmierć', edgecolor='black')
# 
# plt.ylabel('Liczba')
# plt.xticks(x_axis+0.1,bins3)
# plt.legend()
# plt.title('Histogram pNN50 w grupie 51 +/- 12 lat')
# plt.tight_layout()
# plt.savefig('Histogram_pNN501.png')
# plt.show()

# HISTOGRAM INTERWAŁY RR 327-351
# plt.figure(figsize=(16, 9))# Tworzenie histogramu
# bins = [450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050,1100]
# # Utwórz przedziały dla słupków

# meanRR_list1=count_elements_in_bins(meanRR_list, bins)
# meanRR_list_heart1=count_elements_in_bins(meanRR_list_heart, bins)
# meanRR_list_heart_failure1=count_elements_in_bins( meanRR_list_heart_failure, bins)
# meanRR_list_ill1=count_elements_in_bins( meanRR_list_ill, bins)
# meanRR_list_heart_AF1=count_elements_in_bins(meanRR_list_heart_AF,bins)


# x_axis = np.arange(len(bins))# Rysuj słupki dla różnych kategorii
# plt.bar(x_axis+0.25,meanRR_list1, width=0.25, label='Zdrowi',edgecolor='black')
# #plt.bar(x_axis+0.2*2, meanRR_list_ill1, width=0.2, color='red', label='Udar niedokrwienny mózgu', edgecolor='black')
# plt.bar(x_axis+0.25*2, meanRR_list_heart_AF1, width=0.25, color='m', label='migotanie przedsionów', edgecolor='black')
# plt.bar(x_axis + 0.25*3, meanRR_list_heart1, width=0.25,  color='green', label='Niewydolność serca', edgecolor='black')
# plt.bar(x_axis + 0.25*4, meanRR_list_heart_failure1, width=0.25, color='c', label='Nagła śmierć', edgecolor='black')
# plt.ylabel('Liczba')
# plt.xlabel('Zakres interwałów RR [ms]')
# plt.xticks(x_axis+0.1,bins)
# plt.legend()
# plt.title('Histogram średniej interwałów RR w grupie 51 +/- 12 lat')
# plt.savefig('Histogram_interwaly_RR1.png')
# plt.show()


# # bins4=np.linspace(10,100,18)
# # plt.subplot(2, 3, 5)
# # plt.hist(SDANN_list, bins=bins4, label='Zdrowi')
# # plt.plot((90, 90), (0, 10), color='k', label='srednia')
# # plt.hist(SDANN_list_ill,bins=bins4, color='red', label='Udar niedokorwienny mózgu')
# # plt.ylabel('Liczba')
# # plt.title('Histogram SDANN w grupie 51 +/- 12 lat')
# # plt.legend()
# # plt.show()

# LICZENIE ENTROPI
# plt.figure(figsize=(16, 9))
# bins4=np.linspace(0,3,13)
# plt.hist(entropy, bins=bins4,label='Sample Entropy dla k=0.2', color='g',edgecolor='black')
# plt.xticks(bins4)
# plt.ylabel('Liczba')
# plt.xlabel('Entropia')
# plt.legend()
# plt.title('Sample Entropy w grupie 51 +/- 12 lat')
# plt.show()
# # bins3=np.linspace(0,40,8)
# # bins3=np.round(bins3,2)

# plt.figure(figsize=(16, 9))# Tworzenie histogramu
# bins4=np.linspace(0,2.5,13)
# # Utwórz przedziały dla słupków
# bins4=np.round(bins4,2)


# entropy1=count_elements_in_bins(entropy, bins4)
# entropy_ill1=count_elements_in_bins(entropy_ill, bins4)
# entropy_heart1=count_elements_in_bins( entropy_heart, bins4)
# entropy_heart_fauliure1=count_elements_in_bins( entropy_heart_fauliure, bins4)
# entropy_heart_AF1=count_elements_in_bins(entropy_heart_AF,bins4)


# x_axis = np.arange(len(bins4))# Rysuj słupki dla różnych kategorii
# plt.bar(x_axis+0.2,entropy1, width=0.2, label='Zdrowi',edgecolor='black')
# plt.bar(x_axis+0.2*2, entropy_heart_AF1, width=0.2, color='m', label='migotanie przedsionów', edgecolor='black')
# #plt.bar(x_axis+0.2*2, entropy_ill1, width=0.2, color='red', label='Udar niedokrwienny mózgu', edgecolor='black')
# plt.bar(x_axis + 0.2*3, entropy_heart1, width=0.2,  color='green', label='Niewydolność serca', edgecolor='black')
# plt.bar(x_axis + 0.2*4, entropy_heart_fauliure1, width=0.2, color='c', label='Nagła śmierć', edgecolor='black')
# plt.ylabel('Liczba')
# plt.xticks(x_axis+0.1,bins4)
# plt.legend()
# plt.title('Histogram Sample Entropy')
# plt.savefig('Historgram_entropii1.png')
# plt.show()

# Tworzenie histogramu
# bins=[450,500,550,600,650,700,750,800,850,900,950,1000,1050,1100]
# plt.subplot(1, 2, 2)
# #plt.hist(meanRR_list, bins=bins,label='Zdrowi')
# plt.plot((900, 900), (0, 10), color='k', label='średnia')
# #plt.hist(meanRR_list_ill,bins=bins, color='red', label='Udar niedokorwienny mózgu')
# #plt.hist(meanRR_list_heart,bins=bins,color='green', label = 'niewydolność serca')
# #plt.hist(meanRR_list_heart_failure,bins=bins,color='c', label = 'nagła śmierć')
# plt.hist(meanRR_list_heart_AF,bins=bins,color='magenta', label = 'migotanie przedsionków')
# plt.ylabel('Liczba')
# plt.legend()
# plt.title('Histogram średniej interwałów RR w grupie 51 +/- 12 lat')
