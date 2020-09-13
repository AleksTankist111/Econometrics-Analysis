#-------------------------------------------------------------------------------
# Name:        delay_searching
# Purpose:      Searching parameters (array's element) with highest covariance
#               level between main function and features functions.
#
# Author:      Alexander Skakun
#
# Created:     05.09.2020
# Copyright:   (c) Alexander Skakun 2020
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np


### Инициализируем способы выделения значимых элементов:


def filtered_deviation(x):

    """
    Первый способ выделения выбросов:
        1) Проходим по массиву данных, выбрасывая все элементы вне 2 std
        2) Считаем std получившегося массива
        3) Выводим медиану нового массива и std этого массива

    Rerurn: std of filtered x

    """

    m =  np.median(x)
    std = np.std(x)
    x_filtered = []
    for val in x:
        if val<m+2*std: x_filtered.append(val)
    std = np.std(x_filtered)
    return std


def median_diviation(x):

    """
    Второй способ выделения выбросов:
        1) Находим медиану массива
        2) Считаем аналог стандартного отклонения, только вместо mean(x)
            используем median(x)

    Rerurn: median deviation of x

    """

    m = np.median(x)
    md = np.sqrt(1/len(x) * sum((x-m)**2))
    return md


def IQR_deviation(x, low=False):

    """
    Parameters:
        low: boolean - необходимость выводить нижнюю квартиль.
            По умолчанию - False

    Третий способ выделения выбросов:
        Работает по аналогии с определением выбросов в boxplot диаграмме:
        1) Находим медиану x
        2) Находим верхнее граничное значение, между которым и медианой лежат
            25% данных
        3) Находим нижнее граничное значение, между которым и медианой лежат
            25% данных
        4) Находим IQR - расстояние между границами
        5) Прибавляем IQR * 1.5 к верхней границе - получаем значение
            верхней границы выброса
        6) Вычитаем IQR*1.5 из нижней границы - получаем значение
            нижней границы выброса

    Rerurn: high_IQR-border [, low_IQR_border] - if requested

    """
    m = np.median(x)
    x = np.sort(x)
    for i in range(len(x)):
        if x[i]>m:
          high_Q = x[i:]
          low_Q = x[:i]
          break
    quarter = len(x)//4
    iqr = (high_Q[quarter]+high_Q[quarter+1])/2-(low_Q[-quarter]+low_Q[-(quarter+1)])/2
    iqr_h = iqr*1.5 + (high_Q[quarter]+high_Q[quarter+1])/2
    iqr_l = (low_Q[-quarter]+low_Q[-(quarter+1)])/2 + iqr*1.5

    if low: return iqr_h, iqr_l
    else: return iqr_h


def max_deviation(x, significance = 0.85):

    """
    Parameters:
        significance (по умолчанию 0.85): float от 0 до 1;
             порог значимости (см. далее)

    Четвертый способ выделения выбросов:

        1) Сортируем массив x по убыванию

        2) Обращаем массив x (x = 1-x) и масштабируем.
         Теперь первое значение массива - 0, последнее - 1


        3) Принимая конечное итоговое значение за 100%, принимаем за искомую
            границу величину, справа которой лежат значения >significance (%)
            Таким образом, граница ставится на (1-significance) % от максимума

    Rerurn: border

    """

    x = -np.sort(-np.abs(np.array(x)))
    rev_x = 1-x/x[0]
    rev_x /= rev_x[-1]
    for i, val in enumerate(rev_x):
        if val>significance: return (x[i]+x[i-1])/2

def sorted_deviation(x, significance = 0.5):

    """
    Parameters:
        significance (по умолчанию 0.5): float от 0 до 1;
             порог значимости (см. далее)

    Пятый способ выделения выбросов:
        1) Сортируем массив x по убыванию
        2) Считаем кумулятивную сумму массива, масштабируем её на 1
        3) Принимая конечное итоговое значение за 100%, принимаем за искомую
            границу величину, справа которой лежат значения >significance (%)

    Return: border

    """

    x = -np.sort(-x)
    cumsum = np.cumsum(x)
    cumsum /= cumsum[-1]
    for i, val in enumerate(cumsum):
        if val>significance:
            return (x[i]+x[i-1])/2


### Генерализованная функция подсчета границ:

def deviation(x, dev_func=sorted_deviation, mult_factor=None, **params):

    """
    deviation(x, dev_func=sorted_deviation, mult_factor=None, **params)

    Parameters:
        x: numpy массив или список с данными, в которых необходимо найти выбросы

        dev_func: (по умолчанию sorted_deviation) - функция подсчета границ выбросов
        Варианты:

            IQR_deviation,
            filtered_deviation,
            median_diviation,
            max_deviation,
            sorted_deviation

            За описанием обращаться к описанию соответсвующих функций;

        mult_factor: (по умолчанию None) - None или float - мультипликативный фактор
            границ функции. Нужен для уточнения границ в зависимости от метода.
            Если None, то используются значения по умолчанию,
                 соответсвующие каждой функции:

            IQR_deviation: 1,
            filtered_deviation: 4,
            median_diviation: 2,
            max_deviation: 1,
            sorted_deviation: 1

        Например, для filtered_deviation результат выполнения функции будет:

            result = MEDIAN(x) + 4 * filtered_deviation(x)

    Return: Абсолютное значение границы.

    """
    if mult_factor == None:

        __dev_coefs = {
        IQR_deviation: 1,
        filtered_deviation: 4,
        median_diviation: 2,
        max_deviation: 1,
        sorted_deviation: 1
        }

        mult_factor = __dev_coefs[dev_func]

        del __dev_coefs

    x = np.abs(x)

    if dev_func in (sorted_deviation, max_deviation):
        return dev_func(x, **params)
    else:
        return np.median(x) + mult_factor * dev_func(x, **params)


def list_dev_f():

    """
    Функция, возвращающая список всех функций подсчета границ.

    """
    dev_list = [
        IQR_deviation,
        filtered_deviation,
        median_diviation,
        max_deviation,
        sorted_deviation
        ]

    return dev_list


if __name__ == '__main__':
    print ('    Данный модуль реализует множество способов вычисления \n \
    границы значимости ковариаций. Все способы представлены функциями и \n \
    генерализованы внутри одной общей управляющей функции "deviation". \n \
    Также модуль содержит некоторые вспомогательные функции.')

