import os
# import pandas as pd
# from sklearn.model_selection import train_test_split #для обучения
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# import pickle #для сериализации, сохранить в виде файла
import requests
# import matplotlib.pyplot as plt
# import numpy as np
# import itertools
#
# # Загрузка данных Титаника из csv-файла
# data = pd.read_csv('titanic.csv')
# data = data [['Survived','Pclass','Age','Fare']] #столбцы для обучения
# data = data.dropna(subset=['Age'])
#
# data.info()
#
# data.Fare.describe()
#
# # Survived - выжил, по какому полю ОС, обучение с ОС
# data.drop('Survived', axis=1)
#
# # Разделение данных на обучающую и тестовую выборки
# train, test = train_test_split(data, test_size=0.2)
#
#
# """Функция для построения матрицы ошибок.
# cm - матрица ошибок
# classes - список классов
# normalize - если True, то значения матрицы ошибок нормализуются к 1
# title - заголовок графика
# cmap - цветовая схема для отображения графика"""
#
# # Функция для визуализации матрицы
# def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
#
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#
#
# def experiment(max_depth, min_samples_split):
#     # Создание и обучение модели решающего дерева
#     model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
#     model.fit(train.drop('Survived', axis=1), train['Survived'])
#
#     # Вычисление метрик
#     preds = model.predict(test.drop('Survived', axis=1))
#     acc = accuracy_score(test['Survived'], preds)
#     cm = confusion_matrix(test['Survived'], preds)
#
#     print("accuracy", acc)
#
#     # Визуализация матрицы ошибок
#     plot_confusion_matrix(cm, classes=['Not Survived', 'Survived'])
#
#     # Вывод classification report
#     report = classification_report(test['Survived'], preds, target_names=['Not Survived', 'Survived'])
#     print(report)
#
#     # Сохранение модели в формате pickle
#     with open('model2.pkl', 'wb') as f:
#         pickle.dump(model, f)
#
#
# # Определение гиперпараметров модели
# max_depth = 5
# min_samples_split = 150
#
# experiment(max_depth, min_samples_split)
#
# ## Инференс модели
#
# # Загрузка модели из файла pickle
# with open('model.pkl', 'rb') as f:
#     model = pickle.load(f)
#
#
# # Новые данные
# new_data = pd.DataFrame({
#     'Pclass': [3],
#     'Age': [5.0],
#     'Fare': [7.2500]
# })
#
#
# # Предсказание
# predictions = model.predict(new_data)
#
# # Вывод результатов
# print("Predicted Survival:", predictions)
#
# ## Test API


def predict_model(data):
    url = 'http://127.0.0.1:500/predict_model'

    # Отправка POST-запроса с данными в формате форм-данных
    response = requests.post(url, json=data)

    # Проверка статуса ответа
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Request failed with status code {response.status_code}"}

# Пример данных для предсказания
data = {
    "Pclass": 3,
    "Age": 22.0,
    "Fare": 10.250
}

# Получение предсказания
prediction = predict_model(data)
print(prediction)
