#-------------------------------------------------------------------------------
# Name:        exmaples_constructing
# Purpose:      Создание тестовых наборов данных для проверки работы программы
#
# Author:      Alexander Skakun
#
# Created:     06.09.2020
# Copyright:   (c) Alexander Skakun 2020
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np
import numpy.random as rnd
import covariance_processing as cp
import matplotlib.pyplot as plt

class Example():
    """
    Example(n_channels, max_dif = None, n_diffs = None,  diffs = None,
        noise_level = 0.5, style = 'random', random_seed = None)

    Класс объектов-примеров. Объект состоит из нескольких каналов-признаков, и
    одного канала-результата.

    Каждый канал-признак независим от остальных **(можно добавить зависимость)**
    Каналы признаки с некоторой задержкой влияют на канал-результат, при этом
    влияние может быть как единичное, так и множественное (несколько элементов
    одного канала влияет на результат). Влияние определяется линейной комбинацией
    всех значимых элементов.

    Также есть возможность добавить в результат шум.

    Parameters:

        n_channels: int:   Количество каналов-признаков.

        min_dif: int, deafult = 0: Минимально возможное значение задержки.

        max_dif: int, deafult = None:   Максимальная допустимая задержка.
            Обязательный параметр, если не заданы diffs.

        n_diffs: list, deafult = None:  Список n_channels x 1 - количество
            задержек в каждом канале.

        diffs: list of lists OR matrix, deafult = None: Список значимых задержек
            для каждого канала. Если None, определяется случайно. Конфликтует с
            max_dif.

        noise_level: float, deafult=0.5: Определяет уровень шума. Если
            noise_level=False OR 0, то не учитывается. Иначе:
                0<float - float*100 % исходного std канала-результата

        style: str or function, deafult='random':   Определяет способ задания
             случайных последовательностей (каналов):
                'random' - uniform distributed random numbers
                'normal' - normal distributed random nubers with std=5
                'dnormal' = нормально распределенные числа, каждое новое число
                    считается, принимая предыдущее значение за mean.
                    Позволяет создавать более динамичные данные.
                function object - любая функция, которая будет использоваться для
                    генерации данных. Использование происходит следующим образом:
                        F(np.arange(length))*\random_coef\

        random_seed: int, deafult=None: Сид, определяющий выбор случайных величин.

    Attributes:

        diffs_: list of lists of int: Список из n_channels списков с целыми числами -
            значимыми задержками для каждого из каналов.

        coefs_: list of lists of floats:    Список из n_channels списков, каждый из
            которых содержит своё количество float чисел, соответствующих каждой
            задержке, значимой для данного канала;

        сparams_: list of lists of floats:   Список из n_channel списков с
            параметрами для каждого из каналов. Параметры:
                mean - отклонение значение канала от нуля;
                ampl - для 'normal' равносильно std - амплитуда колебаний значений;

    Methods:

        create([random_seed]): Функция, создающая набор данных.


    """

    def __init__(self, n_channels, min_dif = 0, max_dif = None, n_diffs = None, diffs = None,
        noise_level = 0.5, style = 'random', random_seed = None):

        rnd.seed(random_seed)

        if not (max_dif or diffs): raise AttributeError('Не заданы параметры диапазона задержек')

        if diffs==None:

            if max_dif<min_dif: raise ValueError('Несовпадение количества значимых задержек')
            diffs = []

            if not n_diffs: n_diffs = rnd.randint(max_dif-min_dif, size=n_channels)
            else:
                if any(np.array(n_diffs)>max_dif-min_dif):
                    raise ValueError('Несовпадение количества значимых задержек')

            for channel in range(n_channels):
                diffs.append(rnd.choice(max_dif-min_dif, size = n_diffs[channel], replace = False)+min_dif)

        else:
            if len(diffs) != n_channels: raise ValueError('Number of channels mismatched')

        self.diffs_ = diffs
        self.n_channels = n_channels
        self.style = style
        self.noise_level = noise_level
        self.coefs_ = []
        self.cparams_ = []

        ### Создаем коэффициенты для каждой задержки каждого канала
        for channel in range(self.n_channels):
            self.cparams_.append(rnd.random(2)* 20 - 10)
            self.coefs_.append(rnd.random(len(self.diffs_[channel])) * 10 - 5)



    def create(self, length = 1000, random_seed = None):

        """
        Функция, создающая набор данных.

        Parameters:

            length: int, deafult=1000:  Длина каждого канала

            random_seed: int, deafult=None: Сид, определяющий выбор случайных величин.

        Return: X, y - сгенерированный набор данных:
            X - numpy матрица n_channels x length - каналы-признаки
            y - numpy матрица length - канал наблюдения

        """

        rnd.seed(random_seed)

        X = []
        y = []

        for channel in range(self.n_channels):
            if self.style == 'random':
                X.append(rnd.random(length) * (2 * self.cparams_[channel][1])
                 - self.cparams_[channel][1] + self.cparams_[channel][0])

            elif self.style == 'normal':
                X.append(rnd.normal(self.cparams_[channel][0], np.abs(self.cparams_[channel][1]), length))

            elif self.style == 'dnormal':
                X.append(np.zeros(length))
                X[-1][0] = rnd.random()*(2*self.cparams_[channel][1])
                -self.cparams_[channel][1]+self.cparams_[channel][0]

                for i in range(1, length):
                    X[-1][i] = rnd.normal(X[-1][i-1], np.abs(self.cparams_[channel][1]))


            else:
                X.append(self.style(np.arange(length)+rnd.random()*10)*(2*self.cparams_[channel][1])
                -self.cparams_[channel][1]+self.cparams_[channel][0])

        X = np.array(X)

        max_delay = cp.max_delay(self.diffs_)

        for t in range(max_delay, length):
            res = 0
            for channel in range(self.n_channels):
                for i, dif in enumerate(self.diffs_[channel]):
                    res+=X[channel][t-dif]*self.coefs_[channel][i]
            y.append(res)

        if self.noise_level:
            for channel in range(self.n_channels):
                X[channel] += rnd.normal(0, self.noise_level*np.abs(self.cparams_[channel][1]), length)

        y = y[-(X.shape[1]-len(y)):] + y

        y = np.array(y) + rnd.normal(0, self.noise_level*np.std(y), length)

        return X, y



if __name__ == '__main__':
    print('Данный модуль предназначен для создания тестовых данных для проверки \n \
    работы остальных модулей. Создает объект класса Example с набором \n \
    параметров, и методом create, позволяющим создать новый набор.')

    TEST = Example(4, max_dif=10)
    print('Текущие значимые задержки: ', TEST.diffs_)
    X, y = TEST.create(20, 1)

    for i, channel in enumerate(X):
        plt.plot(channel, label = 'Channel # {:d}'.format(i+1))
    plt.plot(np.arange(len(channel)-len(y), len(channel)), y, label = 'Result Channel')

    plt.legend()
    plt.grid()
    plt.show()

    TEST2 = Example(4, max_dif = 10, n_diffs=[4,2,1,0], style='dnormal')

    print('Второй вариант: ', TEST2.diffs_)
    X2, y2 = TEST2.create()

    for i, channel in enumerate(X2):
        plt.plot(channel, label = 'Channel # {:d}'.format(i+1))
    plt.plot(np.arange(len(channel)-len(y2), len(channel)), y2, label = 'Result Channel')

    plt.legend()
    plt.grid()
    plt.show()

