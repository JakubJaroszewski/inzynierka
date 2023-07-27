import csv
import os
# do przerzucenia plików z csv na txt
def read_csv_rows(filename):
    values = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            row_values = row[1:] 
            values.append(row_values)
    return values

filename = 'normalRRs_STR.csv'
data = read_csv_rows(filename)

sciezka_katalogu = "./data_zastoinowa_niewydolnosc_serca"
for i in range(2,len(data)+2,1):
  filename = str(i) +'.txt'
  pelna_sciezka = os.path.join(sciezka_katalogu, filename)
  file = open(pelna_sciezka, "w")
  for element in data[i-2]:
    file.write(str(element) + "\n")
  file.close()