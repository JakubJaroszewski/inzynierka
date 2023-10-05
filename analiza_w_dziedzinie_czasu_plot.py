import math
import numpy as np
from hrvanalysis import get_time_domain_features
import matplotlib.pyplot as plt
import statistics
from matplotlib.patches import Ellipse
from scipy.fft import fft
import sys

def main(ShowPoincarePlot, file_path_get,GeneratePDF):
    file_path = file_path_get
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
    #approximate_entropy
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
        k=0.1 # przyjete 10 %
        return np.log(((N-m-2)/(N-m-1))*AlphaV(k,var,m,N,data)/AlphaV(k,var,m+1,N,data))

        
    import hrvanalysis as hrv
    data_array = read_data_from_file(file_path)
    data_array= hrv.get_nn_intervals(data_array)
    mean_RR=sum(data_array)/len(data_array)
    print('mean RR:',round(mean_RR,2))
    print('SDNN: ', round(SDNN(data_array,mean_RR),2))
    print('RMSSD: ', round(RMSSD(data_array),2))
    print('pNN50: ', round(pNN50(data_array)*100,2), '%')
    print('SDANN:', segmentation(data_array))
    data_array0=data_array[0:len(data_array)-1]
    data_array1=data_array[1:len(data_array)]
    filtred_data_x,filtred_data_y=above_line(data_array0,data_array1)
    x=[600,1500]
    y=[600,1500]

    #Porta indicator (PI) 
    points_below_line=len(data_array0)-len(filtred_data_x)
    P=points_below_line/len(data_array0)
    print("Porta indicator:",round(P*100,3),'%')

    #Guzik indicator (GI)
    up= dist_from_line(filtred_data_x,filtred_data_y)
    down= dist_from_line(data_array0,data_array1)
    GI=round(up*100/down,3)
    print("Guzik indicator:",GI,'%')

    #Wyznaczenie Entropii
    entropy = round(ApEN(2,len(data_array),data_array,round(RMSSD(data_array),2)),3)
    print("Sample Entropy:",entropy )

    #Policzenie wspolczynikow elpisy 
    H=SD1(data_array0)
    D=SD2(data_array0,mean_RR)
    print("SD1:", round(H,2))
    print("SD2:", round(D,2))
    print("Ratio SD1/SD2:", round(H/D,2))

    SD1P = [mean_RR,mean_RR+(D/np.sqrt(2))]
    SD2P = [mean_RR,mean_RR+(D/np.sqrt(2))]
    SD1H = [mean_RR,mean_RR-(H/np.sqrt(2))]
    SD2H = [mean_RR,mean_RR+(H/np.sqrt(2))]

    if ShowPoincarePlot:
        plt.figure(1)
        plt.subplot(1, 1, 1) 
        plt.scatter(data_array0,data_array1,label="Interwały NN")
        plt.plot(x,y,color="deepskyblue", label="Prosta y=x")
        plt.xlim(650, 1500) 
        plt.ylim(650, 1500)  
        plt.xlabel('$NN_{n}$')
        plt.ylabel('$NN_{n+1}$')
        plt.plot(SD1H,SD2H,color="lime", label="SD1")
        plt.plot(SD1P,SD2P,color="red", label="SD2")
        ellipse = Ellipse([mean_RR,mean_RR], 2*D, 2*H, angle=45, fill=False, edgecolor='black', linewidth=2 ,label= "Dopasowana elipsa")
        plt.gca().add_patch(ellipse)
        plt.title("Poincaré plot")
        plt.legend()
        # plt.savefig('PoincarePlot.png')
        plt.show()
        # time_domain_features = get_time_domain_features(data_array)  
        # print(time_domain_features)
    if GeneratePDF:
        data = {
        "": [
            "Mean NN [ms]",
            "SDNN [ms]" ,
            "RMSSD [ms]",
            "pNNN50 [%]",
            "SDANN [ms]",
            "Porta indicator [%]",
            "Guzik indicator [%]",
            "Entropia",
            "SD1",
            "SD2",
            "Ratio SD1/SD2"
        ],
        "Wyniki": [
            round(mean_RR,2),
            round(SDNN(data_array,mean_RR),2),
            round(RMSSD(data_array),2),
            round(pNN50(data_array)*100,2),
            segmentation(data_array),
            round(P*100,3),
            GI,
            entropy,
            round(H,2),
            round(D,2),
            round(H/D,2)
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
    ['lightgray', 'lightgray'],
    ['white', 'white' ],
    ['lightgray', 'lightgray' ],
    ['white', 'white' ],
    ['lightgray', 'lightgray' ],
    ['white', 'white' ],
    ['lightgray', 'lightgray' ]]
    the_table = ax.table(cellText=df1.values,colLabels=df1.columns,loc='top',cellColours=kolorowa_macierz, cellLoc= 'left' )
    pp = PdfPages("table11.pdf")
    pp.savefig(fig, bbox_inches='tight')
    pp.close()


if __name__ == "__main__":
    main()
# plt.subplot(1, 2, 2) 
# plt.scatter(filtred_data_x,filtred_data_y)
# plt.plot(x,y,color="green")
# plt.xlim(650, 1300) 
# plt.ylim(650, 1300)  
# plt.xlabel('RR_n')
# plt.ylabel('RR_n+1')
# plt.show()


# time_values = np.cumsum(data_array)
# time_diffs = np.diff(time_values)
# frequency_spectrum = np.fft.fft(time_diffs)
# frequencies = np.fft.fftfreq(len(time_diffs))

# plt.plot(frequencies, np.sqrt(np.abs(frequency_spectrum)))
# plt.xlabel('Częstotliwość [Hz]')
# plt.ylabel('Amplituda [ms^2]')
# plt.title('Widmo częstotliwościowe')
# plt.ylim(0,100)
# plt.xlim(0,0.5)
# plt.show()






