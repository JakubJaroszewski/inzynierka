import pandas as pd
# program słuzy do dopasowywania danych, tab aby je potem wykorzstac
for i in range(0,18,1):
  if i == 0:
    file_path='./data_nowe_RR_zdrowi/rr'+ '.txt.'#str(i)
  else:
    file_path='./data_nowe_RR_zdrowi/rr'+ '.txt.'+str(i)
  print(file_path)
  data = pd.read_csv(file_path, delimiter='\t')
  selected_columns = data.iloc[:,2]*1000
  selected_columns1=selected_columns[0:1000]
  selected_columns1.to_csv(file_path, sep='\n', index=False,header=False)   



