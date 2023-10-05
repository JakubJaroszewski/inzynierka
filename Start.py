import sys

ShowPoincarePlot = False
PathToFile = './data_zdrowi/5.txt' # ścieżka do pliku
GeneratePDF = True # generowanie pdf z wynikami

try:
    with open('analiza_w_dziedzinie_czasu_plot.py', 'r') as file:
        code = compile(file.read(), 'analiza_w_dziedzinie_czasu_plot.py', 'exec')
    namespace = {'__file__': 'analiza_w_dziedzinie_czasu_plot.py'}  # Tworzenie przestrzeni nazw
    exec(code, namespace)
except FileNotFoundError:
    print("Plik 'analiza_w_dziedzinie_czasu_plot.py' nie istnieje.")
except Exception as e:
    print("Wystąpił błąd:", e)

# Przekazanie argumentów do inny_plik.py
if 'namespace' in locals() and 'main' in namespace:
    namespace['main'](ShowPoincarePlot,PathToFile,GeneratePDF)  # Przekazanie argumentów do funkcji main