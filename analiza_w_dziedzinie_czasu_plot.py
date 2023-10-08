import math
import numpy as np
from hrvanalysis import get_time_domain_features
import matplotlib.pyplot as plt
import statistics
from matplotlib.patches import Ellipse
from scipy.fft import fft
import sys
import pandas as pd

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
    from scipy import signal
    from scipy.integrate import trapz
    from scipy.interpolate import interp1d
    # from hrv.io import read_from_csv
    # from hrv.detrend import sg_detrend
    def frequencydomain(rr, show_plt):
        # Interpolating rr intervals series
        fs = 4.0 # sampling
        steps = 1/fs # lenght of the step
        x = np.arange(1,np.max(rr.index),steps)
        list = []
        for i in range(0,len(x)+1):
            if i % 4 == 0:
                list.append(rr['RR'][int(i/fs)])
            else:
                list.append(None)
        rr_interp = pd.DataFrame(list)
        rr_interp = rr_interp.interpolate(method='cubic')

        # Estimating the spectral density
        fxx, pxx = signal.welch(x=rr_interp[0], fs=fs)

        band_vlf = (fxx >= 0) & (fxx < 0.04) # very low frequency
        band_lf = (fxx >= 0.04) & (fxx < 0.15) # low frequency
        band_hf = (fxx >= 0.15) & (fxx < 0.4) # high frequency

        vlf_power = trapz(pxx[band_vlf], fxx[band_vlf])
        lf_power = trapz(pxx[band_lf], fxx[band_lf])
        hf_power = trapz(pxx[band_hf], fxx[band_hf])

        total_power = vlf_power + lf_power + hf_power
        lf_hf_ratio = lf_power / hf_power
        if show_plt:
            psd_f = interp1d(fxx,pxx)
            x_vlf = np.linspace(0,0.04,100)
            x_lf = np.linspace(0.04,0.15,100)
            x_hf = np.linspace(0.15,0.4,100)

            plt.figure(figsize=(20, 7))
            plt.plot(fxx, pxx)
            plt.gca().set_xlim(0, 0.5)
            plt.gca().fill_between(x_vlf, psd_f(x_vlf), alpha = 0.2, color = 'green', label = 'VLF')
            plt.gca().fill_between(x_lf, psd_f(x_lf), alpha = 0.2, color = 'yellow', label = 'LF')
            plt.gca().fill_between(x_hf, psd_f(x_hf), alpha = 0.2,  color = 'red', label = 'HF')
            plt.title("FFT Spectrum (Welch's periodogram) ")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Density")
            plt.legend()
            plt.savefig('WykresMocySygnalu.png')
            plt.show()

        return float(lf_power), float(hf_power), float(total_power), float(lf_hf_ratio)
    import hrvanalysis as hrv

    # def read_and_detrend_hrv(file_path):
    #     data = []
    #     with open(file_path, 'r') as file:
    #         for line in file:
    #             value = float(line.strip())
    #             data.append(value)
    #     return data
    # data_array=[]
    # for i in range(2,47,1):
    #     if i == 33 or i == 32:
    #         continue    
    #     file_path='./data_zdrowi/' + str(i)+ '.txt'
    #     data_array.append(read_and_detrend_hrv(file_path))
    # data_array_NN=[]
    # for i in range(len(data_array)):
    #     data_array_NN.append(hrv.get_nn_intervals(data_array[i],verbose= False))
    #     data_array_NN[i]= pd.DataFrame({'RR' : data_array_NN[i]})
    # data_array_heart=[]
    # for i in range(15,29,1):
    #     if i ==4 or i == 26 or i == 25 or i ==18 or i == 5 or i == 6 or i == 7 or  i == 8 or i == 22 or i == 23 or i ==28 or i ==12  or i==13 or i ==11 or i ==10:
    #         continue
    #     print(file_path)
    #     file_path='./data_zastoinowa_niewydolnosc_serca/rr'+ '.txt.'+str(i)
    #     data_array_heart.append(read_and_detrend_hrv(file_path))
    # NN_heart =[]    
    # for i in range(len(data_array_heart)):
    #     NN_heart.append(hrv.get_nn_intervals(data_array_heart[i],verbose= False))
    #     NN_heart[i]= pd.DataFrame({'RR' : NN_heart[i]})
    # data_array_heart_failure=[]
    # for i in range(0,22,1):
    #     if i == 0:
    #         file_path='./data_nagla_smierc/rr'+ '.txt.'#str(i)
    #     else:
    #         file_path='./data_nagla_smierc/rr'+ '.txt.'+str(i)
    #     if i == 10 or i ==21 or i == 22 or i == 23:
    #         continue
    #     print(file_path)
    #     data_array_heart_failure.append(read_and_detrend_hrv(file_path))
    # NN_heart_F =[]
    # for i in range(len(data_array_heart_failure)):
    #     NN_heart_F.append(hrv.get_nn_intervals(data_array_heart_failure[i],verbose= False))
    #     NN_heart_F[i]= pd.DataFrame({'RR' : NN_heart_F[i]})
    # data_array_heart_AF=[]  
    # for i in range(0,25,1):
    #     if i == 0:
    #         file_path='./data_migotanie/rr'+ '.txt.'#str(i)
    #     else:
    #         file_path='./data_migotanie/rr'+ '.txt.'+str(i)
    #     if i == 1 or i ==6 or i == 18 or i == 23:
    #         continue
    #     print(file_path)
    #     data_array_heart_AF.append(read_and_detrend_hrv(file_path))
    # NN_heart_AF =[]
    # for i in range(len(data_array_heart_AF)):
    #     NN_heart_AF.append(hrv.get_nn_intervals(data_array_heart_AF[i],verbose= False))
    #     NN_heart_AF[i]= pd.DataFrame({'RR' : NN_heart_AF[i]})
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
    import pandas as pd
    data_array_freq= pd.DataFrame({'RR' : data_array})
    frequency_values=frequencydomain(data_array_freq,False)
    print(frequency_values)
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
            "Ratio SD1/SD2",
            "LF [ms^2]",
            "HF [ms^2]",
            "Total Power [ms^2]" ,
            "Ratio LF/HF"
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
            round(H/D,2),
            round(frequency_values[0],2),
            round(frequency_values[1],2),
            round(frequency_values[2],2),
            round(frequency_values[3],2)
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
        ['lightgray', 'lightgray' ],
        ['white', 'white' ],
        ['lightgray', 'lightgray' ],
        ['white', 'white' ],
        ['lightgray', 'lightgray' ]
        ]
        the_table = ax.table(cellText=df1.values,colLabels=df1.columns,loc='top',cellColours=kolorowa_macierz, cellLoc= 'left' )
        pp = PdfPages("Tabela_z_wynikami_"+file_path[2:13]+"_"+file_path[14]+".pdf")
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






