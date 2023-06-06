import math
import numpy as np
#from hrvanalysis import get_time_domain_features
import matplotlib.pyplot as plt
import statistics
from matplotlib.patches import Ellipse

file_path = '1.txt'
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
        
data_array = read_data_from_file(file_path)
mean_RR=sum(data_array)/len(data_array)
print('mean RR:',round(mean_RR,2))
print('SDNN: ', round(SDNN(data_array,mean_RR),2))
print('RMSSD: ', round(RMSSD(data_array),2))
print('pNN50: ', round(pNN50(data_array)*100,2), '%')
print('SDANN:', segmentation(data_array))
data_array0=data_array[0:len(data_array)-1]
data_array1=data_array[1:len(data_array)]
filtred_data_x,filtred_data_y=above_line(data_array0,data_array1)
x=[600,1300]
y=[600,1300]

#Porta indicator (PI) 
points_below_line=len(data_array0)-len(filtred_data_x)
P=points_below_line/len(data_array0)
print("Porta indicator:",round(P*100,3),'%')


#Guzik indicator (GI)
up= dist_from_line(filtred_data_x,filtred_data_y)
down= dist_from_line(data_array0,data_array1)
GI=round(up*100/down,3)
print("Guzik indicator:",GI,'%')


plt.figure(1)
plt.subplot(1, 1, 1) 
plt.scatter(data_array0,data_array1)
plt.plot(x,y,color="green")
plt.xlim(650, 1300) 
plt.ylim(650, 1300)  
plt.xlabel('RR_n')
plt.ylabel('RR_n+1')

#policzenie wspolczynikow elpisy 
H=SD1(data_array0)
D=SD2(data_array0,mean_RR)
ellipse = Ellipse([mean_RR,mean_RR], 2*D, 2*H, angle=45, fill=False, edgecolor='red', linewidth=2 )
plt.gca().add_patch(ellipse)
plt.show()
# plt.subplot(1, 2, 2) 
# plt.scatter(filtred_data_x,filtred_data_y)
# plt.plot(x,y,color="green")
# plt.xlim(650, 1300) 
# plt.ylim(650, 1300)  
# plt.xlabel('RR_n')
# plt.ylabel('RR_n+1')
# plt.show()


#time_domain_features = get_time_domain_features(data_array)  
#print(time_domain_features)