import streamlit as st
import joblib
import pandas as pd
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Загрузка модели
model = joblib.load('random_forest_model.pkl')

# Заголовок окна
st.title("Классификация поставщиков")

# Поля для ввода параметров
x1 = st.number_input("x1")
x2 = st.number_input("x2")
x3 = st.number_input("x3")
x4 = st.number_input("x4")
x5 = st.number_input("x5")
x6 = st.number_input("x6")
x7 = st.number_input("x7")
x8 = st.number_input("x8")

# Кнопка для классификации
if st.button("Осуществить классификацию"):
    # Создание DataFrame из введенных значений
    input_data = pd.DataFrame([[x1, x2, x3, x4, x5, x6, x7, x8]],
                              columns=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8'])

    # Предсказание класса
    prediction = model.predict(input_data)

    # Вывод результата
    st.write(f"Класс поставщика: {prediction[0]}")

    # Визуализация одного из деревьев решений
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(model.estimators_[0], feature_names=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8'],
               class_names=model.classes_, filled=True, ax=ax)
    st.pyplot(fig)
