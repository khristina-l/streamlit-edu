import streamlit as st
import pandas as pd
import joblib

# === Загрузка модели и scaler ===
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
numeric_features = joblib.load('numeric_features.pkl')

st.title("Классификация поставщика по уровню риска")
st.markdown("Введите значения по каждому показателю, чтобы получить прогноз модели.")

# === Описание признаков ===
ordered_numeric = [
    ('x1', 'x1 — Кол-во прошлых заказов'),
    ('x2', 'x2 — Среднее время доставки (дней)'),
    ('x3', 'x3 — % доставок в срок (до 100)'),
    ('x4', 'x4 — Оценка качества материалов (1–5)'),
    ('x5', 'x5 — % дефектных поставок (до 100)'),
    ('x6', 'x6 — Сумма контрактов (в рублях, например, ₽500,000)'),
    ('x7', 'x7 — Рейтинг по отзывам (1–5)'),
    ('x8', 'x8 — Оценка менеджеров (1–5)')
]

# === Ввод от пользователя ===
user_input = {}
for var, label in ordered_numeric:
    user_input[var] = st.number_input(label, value=0.0)

# === Обработка ===
input_df = pd.DataFrame([user_input])

# Масштабирование
input_df[numeric_features] = scaler.transform(input_df[numeric_features])

# === Предсказание ===
if st.button("Предсказать"):
    prediction = model.predict(input_df[numeric_features])[0]
    st.success(f"Предсказанный уровень риска поставщика: **{prediction}**")
