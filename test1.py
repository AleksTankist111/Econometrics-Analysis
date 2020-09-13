import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

import delay_searching as ds
import covariance_processing as cp
import fourier_processing as fp

import train_test_preparing as ttp
from sklearn.linear_model import RidgeCV

import results_analysis as RA

import examples_constructing as exmpl


def deep_processing(X, y, difs):

    scores, evectors = cp.PCA(X, n_components = 0.01)
    scores = np.column_stack((np.ones(scores.shape[0]), scores))

    X_train, y_train, X_test, y_test = ttp.train_test_split(scores, y, test_size = len(b)-max_delay-number_of_train_samples)
    clf = RidgeCV(alphas=[0.0001,0.001,0.01,0.1,1], cv=4).fit(X_train, y_train)

    min_dif = cp.min_delay(difs[:-1])
    MSEs = []
    y_hat = []

    if min_dif>=1:

        dsize = cp.difs_size(difs)

        X_test1 = X_test[:-min_dif]
        y_test1 = y_test[:-min_dif]
        y_hat = clf.predict(X_test1)
        MSEs.append(RA.MSE(y_test1, y_hat))

        X_test = X_test[:,1:] @ evectors.T + X.mean(axis=0)

        for cur_dif in range(1, min_dif+1):

            X_test1 = X_test1[:,1:] @ evectors.T + X.mean(axis=0)

            if cur_dif<min_dif:
                X_test1 = np.row_stack((X_test1[1:],X_test[-min_dif+cur_dif-1]))
                y_test1 = y_test[cur_dif:-min_dif+cur_dif]
            else:

                X_test1 = np.row_stack((X_test1[1:],X_test[-1]))
                y_test1 = y_test[cur_dif:]


            if cur_dif in difs[-1]:
                X_test1[:, 1+dsize - len(difs[-1]) + difs[-1].index(cur_dif)] = y_hat

            X_test1 = (X_test1-X.mean(axis=0)) @ evectors
            X_test1 = np.column_stack((np.ones(X_test1.shape[0]), X_test1))
            y_hat = clf.predict(X_test1)
            MSEs.append(RA.MSE(y_test1, y_hat))


    return MSEs



length = 1000
n_features = 4


#  Генерируем данные

Data_gen = exmpl.Example(n_features, min_dif = 0, max_dif = 14, noise_level=0.1, style = 'dnormal')
a, b = Data_gen.create(1000)


# Фильтруем данные от шума, оставляем только 10% компонент (низких частот)
for i, arr in enumerate(a):
    specter = fp.rfft_wide(arr)
    freques = fp.freques(length, 10)
    fspec = fp.filtered(specter, freques, fmax = freques[-1]/10)
    a[i] = fp.irfft_wide(fspec, border_size=10)

specter = fp.rfft_wide(b)
freques = fp.freques(length, 10)
fspec = fp.filtered(specter, freques, fmax = freques[-1]/10)
b = fp.irfft_wide(fspec, border_size=10)
# Приводим все каналы-признаки и канал-выход к одному std
a0 = cp.transform(a)
b0 = cp.transform([b])[0]

# Считаем ковариации (значимости) небольших задержек

cov_graphs = []

for i in range(n_features):
    cov_graphs.append(cp.cov_range(a0[i], b0, max_dif = 20))

##cov_graphs.append(cp.cov_range(b0, b0, min_dif = 0, max_dif = 20))
##cov_graphs[-1][0] = 0
##cov_graphs[-1][2:] = np.zeros(18)

cov_graphs = np.abs(cov_graphs)
difs=[]


# Рисуем их, попутно находя наиболее значимые задержки и помещая их в массив difs
colors = ['red', 'green', 'blue', 'yellow', 'black']

for i, graph in enumerate(cov_graphs):

    border = ds.deviation(graph, dev_func = ds.sorted_deviation, significance = 0.7)
    difs.extend(cp.important_difs([graph], border))
    plt.plot(graph, color=colors[i])
    plt.axhline(border, color=colors[i], alpha=0.5)

##border = ds.deviation(cov_graphs.reshape((len(cov_graphs)*len(cov_graphs[0]))), dev_func = ds.sorted_deviation)
##difs = cp.important_difs(cov_graphs, border)


##plt.axhline(border)
plt.grid()
plt.show()


max_delay = cp.max_delay(difs)
n_features = cp.difs_size(difs)

### Теперь, зная значимые задержки, составим матрицу для обучения линейного регрессора:

##a = np.row_stack((a,b))
X ,y = cp.get_matriсes(a, b, difs)

X = np.column_stack((np.ones(X.shape[0]), X))
number_of_train_samples = min(10*(n_features+1), len(b)-max_delay-5)

X_train, y_train, X_test, y_test = ttp.train_test_split(X, y, test_size = len(b)-max_delay-number_of_train_samples)

clf = RidgeCV(alphas=np.linspace(0,1,100), cv=4).fit(X_train, y_train)
y_hat = clf.predict(X)

# Теперь тоже самое, но с использованием PCA. Довольно удобно и выгодно!

scores, evectors = cp.PCA(X, n_components = 10)
scores = np.column_stack((np.ones(scores.shape[0]), scores))
number_of_train_samples_pca = min(20*(scores.shape[1]), len(b)-max_delay-5)

X_train1, y_train1, X_test1, y_test1 = ttp.train_test_split(scores, y, test_size = len(b)-max_delay-number_of_train_samples_pca)
clf1 = RidgeCV(alphas=[0.0001,0.001,0.01,0.1,1], cv=4).fit(X_train1, y_train1)

y_hat_pca = clf1.predict(scores)

plt.plot(b, label='original')
plt.plot(np.arange(max_delay, max_delay+len(y_hat)),
    y_hat, label = 'predict with {:d} features'.format(X_test.shape[1]-1), alpha=0.5)
plt.plot(np.arange(max_delay, max_delay+len(y_hat_pca)),
     y_hat_pca, label = 'predict with {:d} PCA-components'.format(scores.shape[1]-1), alpha=0.5)
plt.legend()
plt.grid()
plt.show()


std_b = np.std(b)

MSEs = []
for pred_len in range(10):

    difs_new = cp.reduce_delays(difs[:-1], pred_len)
    difs_new.append(difs[-1])

    X ,y = cp.get_matriсes(a, b, difs_new)
    X = np.column_stack((np.ones(X.shape[0]), X))

    resMSE = deep_processing(X, y, difs_new)
    MSEs.append(np.array(resMSE)/std_b)

for i, arr in enumerate(MSEs):
    if len(arr) != 0:
        plt.plot(arr, 'o-', label = 'Дальность: '+str(i)+' кадров' )

plt.legend()
plt.grid()
plt.show()


### Делаем предсказание вперед насколько возможно:

y_pred0 = [y_test[0]]
dsize = cp.difs_size(difs_new)

for i in range(len(X_test)):

    X_new = X_test[i]

    for idx, val in enumerate(difs_new[-1]):

        if val<=len(y_pred0):
            X_new[- len(difs[-1]) + idx] = y_pred0[-val]

    X_new = X_new.reshape(1,-1)
    y_hat = clf.predict(X_new)

    y_pred0.extend(y_hat)


scores, evectors = cp.PCA(X, n_components = 0.0001)
scores = np.column_stack((np.ones(scores.shape[0]), scores))

X_train, y_train, X_test, y_test = ttp.train_test_split(scores, y, test_size = len(b)-max_delay-number_of_train_samples)
clf = RidgeCV(alphas=[0.0001,0.001,0.01,0.1,1], cv=4).fit(X_train, y_train)

y_pred = [y_test[0]]

for i in range(len(X_test)):

    X_new = X_test[i]
    X_new = X_new[1:].reshape(1,-1) @ evectors.T + X.mean(axis=0)

    for idx, val in enumerate(difs_new[-1]):

        if val<=len(y_pred):
            X_new[0, - len(difs[-1]) + idx] = y_pred[-val]

    X_new = (X_new-X.mean(axis=0)) @ evectors
    X_new = X_new.reshape(1,-1)
    X_new = np.column_stack(([1], X_new))
    y_hat = clf.predict(X_new)

    y_pred.extend(y_hat)

plt.plot(y_pred0, label = 'deep_predict, N: '+str(X.shape[1]-1), alpha=0.5)
plt.plot(y_pred, label = 'deep_predict, escores: '+str(scores.shape[1]-1), alpha=0.5)
plt.plot(b[-len(y_pred):], label='original')
plt.legend()
plt.grid()
plt.show()

