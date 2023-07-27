# from hrvanalysis import get_frequency_domain_features

# # file_path='./data_update/' + str(3)+ '.txt'
# print(get_frequency_domain_features('./data_update/3.txt'))
# import numpy as np
# import nolds

# data=[1000,900,850,870,970,988,999,945,876,956]
# def max_diff(data,data1):
#     arr=[]
#     for k in range(len(data)):
#         arr.append(abs(data[k]-data1[k]))
#     return max(arr)
# def heaviside(x):
#     return 1 if x >= 0 else 0
# def AlphaV(k,var,m,N,data):
#     alpha=0
#     for i in range(N-m+1):
#         for j in range(N-m+1):
#             if i != j:
#                 alpha+=heaviside(k*var-max_diff(data[i:i+m],data[j:j+m]))
#                 # print(max_diff(data[i:i+m],data[j:j+m]))
#                 # print(alpha)
#     return alpha
# def ApEN(m,N,data,var):
#     k=0.1 # przyjete 10 %
#     return np.log(((N-m-2)/(N-m-1))*AlphaV(k,var,m,N,data)/AlphaV(k,var,m+1,N,data))
# print(ApEN(5,len(data),data,4000))
# apen_result = nolds.sampen(data, 5, 0.1)
# print(f"ApEn: {apen_result}")