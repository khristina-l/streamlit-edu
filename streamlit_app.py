import streamlit as st
import joblib
import pandas as pd
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# Загрузка модели
pipe = joblib.load('random_forest_model.pkl')

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
    prediction = pipe.predict(input_data)

    # Вывод результата
    st.write(f"Класс поставщика: {prediction[0]}")

    # 1. Достаём объекты шага препроцессинга и модели
    preprocessor = pipe.named_steps['preprocessor']      # или 'columntransformer', смотрите вывод
    rf_model     = pipe.named_steps['classifier']        # или 'randomforestclassifier'

    # 2. Получаем имена признаков
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:                               # на случай более старой версии sklearn
        feature_names = preprocessor.get_feature_names()

    # 3. Строим дерево
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        rf_model.estimators_[0],
        feature_names=feature_names,
        class_names=rf_model.classes_,
        filled=True,
        ax=ax
    )
    st.pyplot(fig)
