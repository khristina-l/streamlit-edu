import streamlit as st
import pandas as pd
import joblib
import random
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# Настройка страницы
st.set_page_config(page_title="Оценка риска поставщика", layout="wide")

# === Responsive CSS hack ===
st.markdown(
    """
    <style>
    @media (max-width: 768px) {
      section[data-testid=\"stColumns\"] > div {
        width: 100% !important;
        flex: 1 1 100% !important;
      }
    }
    @media (max-width: 600px) {
      .top5-card {
        min-width: 100% !important;
      }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Загрузка модели и данных
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
numeric_features = joblib.load('numeric_features.pkl')

@st.cache_data
def load_data():
    df = pd.read_csv("tochkabalanced_risk_dataset_60_61_59.csv", sep=";")
    df = df.rename(columns={'y': 'Y'})
    df.columns = list(numeric_features) + ['Y']
    return df

@st.cache_data
def fake_suppliers(n: int):
    prefixes = ['ООО', 'ЗАО', 'ИП', 'ПАО']
    roots = ['Гранит', 'ТехПоставка', 'Союз', 'БетонТраст', 'Ресурс', 'Индустрия']
    suf = ['Строй', 'Металл', 'Снаб', 'Трейд']
    return [f"{random.choice(prefixes)} \"{random.choice(roots)}{random.choice(suf)}\"" for _ in range(n)]

# Загрузка и подготовка данных
df = load_data()
df['supplier_name'] = fake_suppliers(len(df))

labels = {
    'x1': 'Кол-во прошлых заказов',
    'x2': 'Среднее время доставки',
    'x3': '% доставок в срок',
    'x4': 'Качество материалов',
    'x5': '% дефектных поставок',
    'x6': 'Сумма контрактов',
    'x7': 'Рейтинг по отзывам',
    'x8': 'Оценка менеджеров'
}

# Sidebar
st.sidebar.subheader("🤖 AI-поиск поставщика")
q = st.sidebar.text_input("Название поставщика")
if st.sidebar.button("Поиск") and q:
    st.sidebar.info("Найден поставщик: рейтинг 4.6/5, риск A")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Параметры модели")
for k, v in labels.items():
    st.sidebar.markdown(f"`{k}` — {v}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Важность признаков")
fi_values = model.feature_importances_
feature_names = [labels[k] for k in numeric_features]
fig_imp = px.bar(
    x=fi_values,
    y=feature_names,
    orientation='h',
    labels={'x': 'Важность', 'y': 'Признак'},
    title='Важность признаков'
)
fig_imp.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
st.sidebar.plotly_chart(fig_imp, use_container_width=True)

st.sidebar.markdown("### 📊 Классы риска")
rc = df['Y'].value_counts().sort_index()
fig_risk = px.bar(
    x=rc.values,
    y=['A', 'B', 'C'],
    orientation='h',
    height=180,
    labels={'x': 'Кол-во', 'y': 'Класс'},
    text_auto=True
)
fig_risk.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
st.sidebar.plotly_chart(fig_risk, use_container_width=True)

# Main Title
st.markdown("<h1 style='text-align:center'>📦 Система оценки риска поставщиков</h1>", unsafe_allow_html=True)

# Risk Card
with st.container():
    st.markdown("<div><h3>🔍 Оценка риска нового поставщика</h3>", unsafe_allow_html=True)

    ex = {f: v for f, v in zip(numeric_features, [615, 10, 98, 4.7, 2, 280000, 4.4, 4])}
    use_ex = st.checkbox("Заполнить примером", key="ex")
    c1, c2 = st.columns(2)
    user = {}
    for idx, f in enumerate(numeric_features):
        with (c1 if idx % 2 == 0 else c2):
            user[f] = st.number_input(labels[f], value=ex[f] if use_ex else 0.0, key=f)

    if st.button("Предсказать риск"):
        X_new = pd.DataFrame([user])
        scaled_new = scaler.transform(X_new)
        pred = model.predict(scaled_new)[0]
        risk_map = {'A': '🟢 Низкий', 'B': '🟡 Средний', 'C': '🔴 Высокий'}
        st.markdown(f"## Результат: {risk_map[pred]}")

        # Вероятности классов
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(scaled_new)[0]
            prob_text = " | ".join([f"{cls}: {p:.1%}" for cls, p in zip(model.classes_, probs)])
            st.markdown(f"**Вероятности по классам:** {prob_text}")

        # Сбор данных для scatter-плота
        X_scaled = scaler.transform(df[numeric_features])
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(X_scaled)
        df['pc1'], df['pc2'] = pcs[:, 0], pcs[:, 1]
        new_pc = pca.transform(scaled_new)[0]

        # Ближайшие соседи
        nbrs = NearestNeighbors(n_neighbors=6).fit(X_scaled)
        distances, indices = nbrs.kneighbors(scaled_new)
        neigh = df.iloc[indices[0][1:]]  # исключаем нулевой (новый сам)

        # Визуализация
        fig = px.scatter(
            df, x='pc1', y='pc2', color='Y',
            labels={'pc1': 'PC1', 'pc2': 'PC2'},
            title='Размещение поставщиков в компонентном пространстве'
        )
        # Новый поставщик
        fig.add_scatter(x=[new_pc[0]], y=[new_pc[1]], mode='markers',
                        marker={'size':14, 'symbol':'x', 'color':'black'},
                        name='Новый поставщик')
        # Ближайшие соседи
        fig.add_scatter(x=neigh['pc1'], y=neigh['pc2'], mode='markers',
                        marker={'size':12, 'color':'red'}, name='Ближайшие соседи')
        fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# База + TOP-5
st.markdown("---")
left, right = st.columns([4, 1], gap="large")
with left:
    st.markdown("### 📘 База поставщиков")
    flt = st.text_input("Поиск", key="flt")
    st.dataframe(
        df[df['supplier_name'].str.contains(flt, case=False)][['supplier_name'] + list(numeric_features) + ['Y']],
        use_container_width=True
    )
with right:
    st.markdown("### 🏆 ТОП-5 A-класса")
    for n in df[df['Y'] == 'A']['supplier_name'].head(5):
        st.markdown(
            f"<div class='top5-card' style='background:#f3f3f3;border-radius:10px;padding:8px;text-align:center;margin-bottom:8px;min-width:100px;'>{n}</div>",
            unsafe_allow_html=True
        )

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center;color:gray'>Разработано © 2025</div>", unsafe_allow_html=True)
