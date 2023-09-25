# program służy do wycinania liczb po kropce 983.349 ucina 349
for i in range(1,18,1):
  if i == 0 :
     file_path='./data_zdrowi2/rr' + '.txt'#+ str(i)
  else:
     file_path='./data_zdrowi2/rr' + '.txt.'+ str(i)
  with open(file_path, 'r') as file:
      lines = file.readlines()
  modified_lines = []
  for line in lines:
      modified_line = line.split('.')[0]
      modified_lines.append(modified_line + '\n')
  with open(file_path, 'w') as file:
      file.writelines(modified_lines)