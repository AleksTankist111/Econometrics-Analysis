#-------------------------------------------------------------------------------
# Name:        fourier_processing
# Purpose:      Модуль для фильтрации в фурье-пространстве, и другими
#               преобразованиями фурье-спектра сигналов.
#
# Author:      Alexander Skakun
#
# Created:     12.09.2020
# Copyright:   (c) Alexander Skakun 2020
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np

def rfft_wide(arr, border_size = 10, border = None, **params):

    """

    Функция, производящая преобразование Фурье со сглаженными концами.

    Parameters:

        arr: np.array OR list - массив с сигналами для быстрого Фурье-преобразования

        border_size: int - размер отступа перед и после массива для сглаживания
            концов.

        border: float, list or None, deafult = None - значение элементов на отступах
            Если None, то выбираются значения первого и последнего элементов
            соответственно.
            Если list - Размер списка должен быть 2*border_size OR 2
                Если размер = 2: 1ый элемент будет использоваться для левой
                границы, второй - для правой границы.

                Если размер 2*border_size: первая половина будет использоваться
                для левой границы, вторая - для правой.

    Return:
        specter - np.array комплексных чисел.

    """
    if border == None:
        arr_new = np.concatenate(([arr[0]]*border_size,arr,[arr[-1]]*border_size))
    elif type(border) == float or type(border) == int:
        arr_new = np.concatenate(([border]*border_size,arr,[border]*border_size))
    elif len(border)==2:
        arr_new = np.concatenate(([border[0]]*border_size,arr,[border[1]]*border_size))
    else:
        arr_new = np.concatenate((border[:len(border)//2],arr,border[len(border)//2:]))

    return np.fft.rfft(arr_new, **params)

def irfft_wide(specter, border_size = 0, **params):

    """

    Функция, выполняющая обратное Фурье-преобразование:

    Parameters:

        specter: array - спектр частот

        border_size: int, deafult = 0 - ширина отступа. Будет обрезано вначале и вконце
            массива.

    Return:
        array, массив исходного сигнала.
    """

    return np.fft.irfft(specter)[border_size:-border_size]


def freques(d, border_size = 0, fdisc=1):

    """
    Функция, считающая частоты спектра фурье:

    Parameters:
        d: int - время (в секундах\кадрах), как долго считывался сигнал

        border_size: int, deafult = 0 - ширина отступа. По умолчанию, 0.

        fdisc: float, deafult=1 -   Частота дискретизации (кадров в секунду),
            по умолчанию - 1 кадр в секунду

    Return: np.array - массив частот фурье преобразования с описанными
        параметрами.


    """
    d+= 2*border_size
    k = fdisc * d #Количество измерений
    fmin = 1/d # Минимальная частота
    fmax = fdisc/2  # Максимальная частота:
    count = int(fmax/fmin) + 1

    return np.linspace(0, fmax, count)

def filtered(specter, freques, fmin=None, fmax=None):

    """
    Функция, фильтрующая данный спектр.

    Parameters:

        specter: np.array - массив, содержащий амплитуды частот (спектр)

        freques: np.array - массив того же размера, что и specter,
            содержащий соответствующие каждому элементу частоты

        fmin: float, deafult=None - Минимальная частота фильтрации. Все частоты
            ниже этой будут очищены. Если None, то fmin=0.

        fmax: float, deafult=None - Максимальная частота фильтрации. Все частоты
            выше этой будут также очищены. Если None, то выбирается максимальная
            частота из freques.

    Return: np.array - фильтрованный спектр

    """
    if fmin == None: fmin = 0
    if fmax == None: fmax = freques[-1]

    amin_idx = np.where(freques >= fmin)
    amax_idx = np.where(freques <= fmax)

    necessary_idx = list(set(amin_idx[0]) & set(amax_idx[0]))

    specter_new = np.zeros(len(freques), dtype = np.complex)
    specter_new[necessary_idx] = specter[necessary_idx]

    return specter_new



if __name__ == '__main__':
    print('Модуль для фильтрации в фурье-пространстве, и другими \n \
    преобразованиями фурье-спектра сигналов. Содержит функции для \n \
    преобразования в фурье-пространство, расчета частотного спектра, \n \
    построения аналогов вейвлета и скелетонов функций.')
