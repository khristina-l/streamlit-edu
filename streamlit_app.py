import streamlit as st
import joblib
import pandas as pd
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# Загрузка модели
pipe = joblib.load('pipeline.pkl')

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

    # ------------------------------------------------------------------
    # Явно приводим типы, чтобы избежать ошибки np.isnan на object-dtype
    # В пайплайне шаги названы 'num' и 'cat' (смотрите, как вы их задали
    # при обучении). Извлекаем списки колонок и делаем кастование.
    # ------------------------------------------------------------------
    preprocessor = pipe.named_steps.get('preprocessor')
    if preprocessor is not None:
        numeric_cols = []
        categorical_cols = []
        for name, transformer, cols in preprocessor.transformers_:
            if name == 'num':
                numeric_cols = list(cols)
            elif name == 'cat':
                categorical_cols = list(cols)

        # Приводим числовые признаки к float, категориальные к str
        if numeric_cols:
            input_data[numeric_cols] = input_data[numeric_cols].astype(float)
        if categorical_cols:
            input_data[categorical_cols] = input_data[categorical_cols].astype(str)

    # Если всё же остались object-dtype колонки, заменяем их на float46/str
    # Это действие «по умолчанию»: для числовых колонок ошибки будут NaN,
    # которые мы сразу заменяем нулями (модель обучена на заполненных данных).
    input_data = input_data.apply(lambda col: pd.to_numeric(col, errors='ignore'))
    input_data = input_data.fillna(0)

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
