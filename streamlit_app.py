import streamlit as st
import pandas as pd
import joblib
import random
import plotly.graph_objects as go
import plotly.express as px

# === Настройки страницы ===
st.set_page_config(page_title="Оценка риска поставщика", layout="wide")

# === Загрузка модели и артефактов ===
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
numeric_features = joblib.load('numeric_features.pkl')   # ['x1', …, 'x8']

# === Читаемые подписи признаков ===
feature_names = {
    'x1': 'Кол-во прошлых заказов',
    'x2': 'Среднее время доставки',
    'x3': '% доставок в срок',
    'x4': 'Качество материалов',
    'x5': '% дефектных поставок',
    'x6': 'Сумма контрактов',
    'x7': 'Рейтинг по отзывам',
    'x8': 'Оценка менеджеров'
}

# === Данные для демонстрации ===
@st.cache_data
def load_data():
    df = pd.read_csv("tochkabalanced_risk_dataset_60_61_59.csv", sep=";")
    df = df.rename(columns={'y': 'Y'})
    df.columns = list(numeric_features) + ['Y']
    return df

df = load_data()

@st.cache_data
def fake_names(n:int):
    prefixes = ['ООО', 'ЗАО', 'ИП', 'ПАО']
    roots    = ['Гранит', 'ТехПоставка', 'Союз', 'БетонТраст', 'Ресурс', 'Индустрия']
    suf      = ['Строй', 'Металл', 'Снаб', 'Трейд']
    return [f"{random.choice(prefixes)} \"{random.choice(roots)}{random.choice(suf)}\"" for _ in range(n)]

df['supplier_name'] = fake_names(len(df))

# =====================================================================================
# SIDEBAR – AI-поиск + модельная инфа + визуализации
# =====================================================================================

st.sidebar.subheader("🤖 AI-поиск поставщика")
query = st.sidebar.text_input("Название поставщика")
if st.sidebar.button("Поиск", key="ai_search_btn") and query:
    st.sidebar.info("Поставщик найден: рейтинг 4.6/5, риск — Низкий (A)")

# Пояснение признаков
st.sidebar.markdown("**📋 Параметры модели**")
for k, v in feature_names.items():
    st.sidebar.markdown(f"`{k}` — {v}")

st.sidebar.markdown("---")

# Информация о модели
st.sidebar.markdown("### ℹ️ Модель")
acc = (model.predict(scaler.transform(df[numeric_features])) == df['Y']).mean()*100
st.sidebar.markdown(f"**Случайный лес**\n\nТочность: **{acc:.2f}%**")

# Donut – важность признаков
st.sidebar.markdown("### 📈 Важность")
fig_imp = go.Figure(go.Pie(labels=[feature_names[f] for f in numeric_features],
                           values=model.feature_importances_,
                           hole=.45,
                           textinfo='percent',
                           textfont_size=10))
fig_imp.update_layout(width=320, height=320, margin=dict(l=0, r=0, t=20, b=0))
st.sidebar.plotly_chart(fig_imp, use_container_width=False)

# Компактное распределение рисков
st.sidebar.markdown("### 📊 Классы риска")
rc = df['Y'].value_counts().sort_index()
fig_risk = px.bar(x=rc.values, y=['A','B','C'], orientation='h',
                 height=180, labels={'x':'Кол-во','y':'Класс'})
fig_risk.update_yaxes(categoryorder='array', categoryarray=['C','B','A'])
fig_risk.update_layout(showlegend=False, margin=dict(l=0,r=0,t=10,b=0))
st.sidebar.plotly_chart(fig_risk, use_container_width=True)

# =====================================================================================
# MAIN LAYOUT  – две колонки: контент + правая «панель» TOP-5
# =====================================================================================

st.markdown("<h1 style='text-align:center'>📦 Система оценки риска поставщиков</h1>", unsafe_allow_html=True)

main, right = st.columns([3,1], gap="large")

# -------------------------- MAIN : Карточка оценки -------------------------------
with main:
    with st.container():
        st.markdown("""
        <div>
        """, unsafe_allow_html=True)
        st.markdown("#### 🔍 Оценка риска нового поставщика")

        example = {k:v for k,v in zip(numeric_features,[45,7,92,4.3,5,300000,4.6,4])}
        use_ex = st.checkbox("Заполнить примером", key="ex")
        c1,c2 = st.columns(2)
        inp = {}
        for idx,f in enumerate(numeric_features):
            lbl = feature_names[f]
            with (c1 if idx%2==0 else c2):
                inp[f] = st.number_input(lbl, value=example[f] if use_ex else 0.0, key=f)

        if st.button("Предсказать риск", key="predict"):
            X_new = pd.DataFrame([inp])
            pred = model.predict(scaler.transform(X_new))[0]
            st.success({'A':'🟢 Низкий','B':'🟡 Средний','C':'🔴 Высокий'}[pred])

        st.markdown("</div>", unsafe_allow_html=True)

    # --- База поставщиков
    st.markdown("---")
    st.markdown("### 📘 База поставщиков")
    fil = st.text_input("Поиск по базе", key="filter")
    st.dataframe(df[df['supplier_name'].str.contains(fil, case=False)][['supplier_name']+numeric_features+['Y']])

# -------------------------- RIGHT : ТОП-5 ---------------------------------------
with right:
    st.markdown("""
    <div>
    <h4 style='text-align:center'>🏆 ТОП-5 A-класса</h4>
    """, unsafe_allow_html=True)
    for n in df[df['Y']=='A']['supplier_name'].head(5):
        st.markdown(f"<div style='background:#f1f1f1;border-radius:10px;padding:8px;text-align:center;margin-bottom:8px;'> {n} </div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================================================
# FOOTER
# =====================================================================================

st.markdown("---")
st.markdown("<div style='text-align:center;color:gray;'>Разработано © 2025</div>", unsafe_allow_html=True)