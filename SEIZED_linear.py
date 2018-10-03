from sklearn import datasets #Для работы с датасетом
from sklearn import metrics #Для наших метрик
from sklearn import linear_model#Сам алгоритм линейной регрессии
from sklearn.preprocessing import StandardScaler #Пренумерация
from pandas import Series, DataFrame #Series - список с метками,DataF - табличная структура
import csv
import pandas as pd
import zipfile

zipfile.ZipFile('creditcard.zip').extractall('.')

open('creditcard.csv','r').readlines() #Открытие исходного файла
cols2drop = [0] #Процесс удаления столбца Time
cols = [i for i in range(32) if i not in cols2drop]
(pd.read_csv(r'/path/to/credicards.csv', usecols=cols, delim_whitespace=True)
   .to_csv(r'/path/to/result.csv', index=False, sep=' '))
dataset = np.loadcsv('result.csv', delimiter=",", dtype = float) #Работа с новым csv без Time

model = LinearRegression() #Процесс бучения модели
model.fit(dataset.data, dataset.target)
print(model)

expected = dataset.target #Процесс Предсказывания 
predicted = model.predict(dataset.data)

print(metrics.classification_report(expected, predicted)) #Вывод
print(metrics.confusion_matrix(expected, predicted))
