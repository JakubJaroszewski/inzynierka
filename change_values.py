# program służy do wycinania liczb po kropce 983.349 ucina 349
for i in range(2,43,1):
  file_path='./data_update1/' + str(i)+ '.txt'
  with open(file_path, 'r') as file:
      lines = file.readlines()
  modified_lines = []
  for line in lines:
      modified_line = line.split('.')[0]
      modified_lines.append(modified_line + '\n')
  with open(file_path, 'w') as file:
      file.writelines(modified_lines)