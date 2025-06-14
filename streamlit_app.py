import streamlit as st
import pandas as pd
import joblib

# Загрузка сохранённых объектов
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')
numeric_features = joblib.load('numeric_features.pkl')
categorical_features = joblib.load('categorical_features.pkl')

st.title("Классификация поставщика")

user_input = {}

# Ввод числовых признаков
for col in numeric_features:
    user_input[col] = st.number_input(f"{col}", value=0.0)

# Ввод категориальных признаков
for col in categorical_features:
    user_input[col] = st.text_input(f"{col}", value="")

# Преобразуем ввод в DataFrame
input_df = pd.DataFrame([user_input])

# Обработка данных
input_df[numeric_features] = scaler.transform(input_df[numeric_features])
cat_encoded = encoder.transform(input_df[categorical_features]).toarray()
cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(categorical_features))

# Объединение
X_user_final = pd.concat([input_df[numeric_features].reset_index(drop=True),
                          cat_encoded_df.reset_index(drop=True)], axis=1)

# Предсказание
if st.button("Предсказать"):
    prediction = model.predict(X_user_final)[0]
    st.success(f"Уровень риска поставщика: **{prediction}**")
