#-------------------------------------------------------------------------------
# Name:        results_analysis
# Purpose:     Проведение линейной, полиномиальной, комплексной полиномиальной,
#               логистической и других регрессий данных; кросс-валидация, оценка.
#
# Author:      Alexander Skakun
#
# Created:     07.09.2020
# Copyright:   (c) Alexander Skakun 2020
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np

def MSE(y, y_hat):

    """
    Функция, считающая среднюю квадратичную ошибку между реальными и
        предсказанными данными.

    Parameters:
        y: list OR numpy array: список реальных меток данных

        y_hat: list OR numpy array: список предсказанных меток данных

    Return: Float: 1 число, корень из средней квадратичной ошибки.

    """
    y = np.array(y)
    y_hat = np.array(y_hat)

    return np.sqrt(sum((y-y_hat)**2)/len(y))


if __name__ == '__main__':
    print('')
