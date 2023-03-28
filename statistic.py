import random
import pandas as pd
import numpy as np
import statistics
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import datasets
import shutil

def generate_random_array(length):
    return [random.randint(1, 100) for i in range(length)]

def generate_random_confusion(length):
    return [random.randint(0, 1) for i in range(length)]

def print_array(array):
    return "\033[0;37mArray: " + str(array) + "\n"

def calculate_mean(array):
    media = np.mean(array)
    formula = "media = (x1 + x2 + ... + xn) / n"
    return media, formula, 1

def calculate_median(array):
    mediana = np.median(array)
    formula = "\bN impar: Mediana = x[(n+1)/2]\nN par: Mediana = (x[n/2] + x[(n/2)+1])/2"
    return mediana, formula, 1

def calculate_mode(array):
    try:
        moda = statistics.multimode(array)
    except:
        moda = "Moda não encontrada"
    if moda == array:
        moda = "Moda não encontrada"
    formula = "moda = valor com maior frequencia no dataset"
    return moda, formula, 1

def calculate_min(array):
    minimo = np.min(array)
    formula = "minimo = menor valor no dataset"
    return minimo, formula, 1

def calculate_max(array):
    maximo = np.max(array)
    formula = "maximo = maior valor no dataset"
    return maximo, formula, 1

def calculate_range(array):
    range_val = np.max(array) - np.min(array)
    formula = "range = maximo - minimo"
    return range_val, formula, 1

def calculate_variance(array):
    variancia = np.var(array)
    formula = "variancia = ((x1-media)^2 + (x2-media)^2 + ... + (xn-media)^2) / n"
    return variancia, formula, 1

def calculate_standard_deviation(array):
    desvio_padrao = np.std(array)
    formula = "desvio_padrao = sqrt(variancia)"
    return desvio_padrao, formula, 1

def calculate_coefficient_of_variation(array):
    cv = (np.std(array) / np.mean(array))
    formula = "coeficiente de variação = (desvio padrão / média)"
    return cv, formula, 1

def true_positive(y_true, y_pred):
    tp = np.sum([1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1])
    return tp

def true_negative(y_true, y_pred):
    tn = np.sum([1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 0])
    return tn

def false_positive(y_true, y_pred):
    fp = np.sum([1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1])
    return fp

def false_negative(y_true, y_pred):
    fn = np.sum([1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0])
    return fn

def print_confusion_matrix(y_true, y_pred):
    return f"\b\b\bMatrix confusão:\n" \
            + f"\033[1;32m\n\b\b\bVP: " + str(true_positive(y_true, y_pred)) \
            + "\n\b\b\bVN: " + str(true_negative(y_true, y_pred)) \
            + "\n\b\b\bFP: " + str(false_positive(y_true, y_pred)) \
            + "\n\b\b\bFN: " + str(false_negative(y_true, y_pred)) + "\n"


def calculate_accuracy(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    acc = (tp + tn) / len(y_true)
    formula = "Acurácia = VP + VN / TOTAL"
    return acc, formula, 2

def calculate_precision(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    if (tp + fp) == 0:
        prec = 0
    else:
        prec = tp / (tp + fp)
    formula = "Precisão = VP / VP + FP"
    return prec, formula, 2

def calculate_recall(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    if (tp + fn) == 0:
        rec = 0
    else:
        rec = tp / (tp + fn)
    formula = "Sensibilidade, revocação = VP / VP + FN"
    return rec, formula, 2

def calculate_specificity(y_true, y_pred):
    tn = true_negative(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    if (tn + fp) == 0:
        spec = 0
    else:
        spec = tn / (tn + fp)
    formula = "Especificidade = VN / VN + FP"
    return spec, formula, 2

def calculate_f1_score(y_true, y_pred):
    prec = calculate_precision(y_true, y_pred)
    rec = calculate_recall(y_true, y_pred)
    if (prec[0] + rec[0]) == 0:
        f1 = 0
    else:
        f1 = (2 * prec[0] * rec[0]) / (prec[0] + rec[0])
    formula = "F1 Score = (2 * Precisão * Sensibilidade) / Precisão + Sensibilidade"
    return f1, formula, 2


def print_center_text(text):
    terminal_width, _ = shutil.get_terminal_size()
    lines = text.split('\n')
    centered_text = '\n'.join(line.center(terminal_width) for line in lines)
    print(centered_text)

def run_stats_game():
    while True:
        length = random.randint(3, 5)
        array = generate_random_array(length)
        y_true = generate_random_confusion(50)
        y_pred = generate_random_confusion(50)
        results = {}
        results['MEDIA'] = calculate_mean(array)
        results['MEDIANA'] = calculate_median(array)
        results['MODA'] = calculate_mode(array)
        results['MINIMO'] = calculate_min(array)
        results['MAXIMO'] = calculate_max(array)
        results['INTERVALO'] = calculate_range(array)
        results['VARIANCIA'] = calculate_variance(array)
        results['DESVIO PADRÃO'] = calculate_standard_deviation(array)
        results['COEFICIENTE DE VARIAÇÃO'] = calculate_coefficient_of_variation(array)
        results['ACURÁCIA'] = calculate_accuracy(y_true, y_pred)
        results['PRECISÃO'] = calculate_precision(y_true, y_pred)
        results['SENSIBILIDADE, REVOCAÇÃO'] = calculate_recall(y_true, y_pred)
        results['ESPECIFICIDADE'] = calculate_specificity(y_true, y_pred)
        results['F1 SCORE'] = calculate_f1_score(y_true, y_pred)
        rand_dict = list(results.items())
        random.shuffle(rand_dict)
        for key, value in rand_dict:
            if value[2] == 1:
                print_center_text(print_array(array))
            elif value[2] == 2:
                print_center_text(print_confusion_matrix(y_true, y_pred))
            print_center_text(f"\033[1;31mCalcule: {key}\n")
            print_center_text(f"\033[1;33mFormula:\n\n\033[1;34m{value[1]}\n")
            print_center_text(f"\033[0;37mAny Key to see the result")
            reveal = input()
            print_center_text(f"\033[1;32m{key}: {value[0]}\n")
            print_center_text(f"\033[0;37mAny Key to continue\n")
            next = input()
            print("\033[H\033[2J")
        print_center_text("\033[0mJogar novamente? (y/n)")
        play_again = input()
        print("\033[H\033[2J")
        if play_again.lower() == "n":
            break

if __name__ == '__main__':
    print("\033[H\033[2J")
    run_stats_game()
