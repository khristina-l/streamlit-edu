Ниже — «пошаговый рецепт», который можно вставить в ваш ноутбук, чтобы  
а) модель обучалась и сохранялась как единый Pipeline,  
б) весь существующий блок визуализации (дерево, матрица ошибок, classification _report_) продолжил работать.

──────────────────────────────────────────
Где и что менять в Randomforest2.ipynb
──────────────────────────────────────────

1. Шаг 1 (загрузка CSV) - остаётся как есть.
2. Шаг 2 (`X = df.drop(...);  Y = df['y']`) - остаётся как есть.

3. Вставьте НОВУЮ ячейку **сразу после шага 2** (то есть она станет «Шаг 3»).  
   В неё перенесём всё, что раньше делали вручную (масштабирование + one-hot), но уже внутри `Pipeline`.

```python
# Шаг 3: строим препроцессор и Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

numeric_features     = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(),                 numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

pipe = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('classifier',   RandomForestClassifier(n_estimators=300, random_state=42))
    ]
)
```

4. **Старый Шаг 3** (где вы вручную скейлили, кодировали и собирали `X_final`)  
   либо закомментируйте, либо просто удалите — он больше не нужен.

5. Поменяйте ячейку с train/test-split и обучением.

```python
# Шаг 4: делим данные (берём «сырые» X, без X_final)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# Шаг 5: обучение Пайплайна
pipe.fit(X_train, Y_train)
```

6. Предсказания и метрики (была ваша ячейка «Шаг 6») — замените `model` на `pipe`.

```python
# Шаг 6: прогноз и метрики
Y_pred = pipe.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, Y_pred)
print(f'Accuracy: {accuracy:.2f}')
```

7. Визуализация дерева.  
   В старом коде было:

```python
single_tree = model.estimators_[0]
dot_data    = export_graphviz(... feature_names=X_train.columns ...)
```

Замените на:

```python
from sklearn.tree import export_graphviz

single_tree   = pipe.named_steps['classifier'].estimators_[0]
feature_names = pipe.named_steps['preprocessor'].get_feature_names_out()

dot_data = export_graphviz(
    single_tree,
    out_file=None,
    feature_names=feature_names,
    class_names=pipe.named_steps['classifier'].classes_.astype(str),
    filled=True, rounded=True, special_characters=True
)
```

Всё остальное (Graphviz визуализация) оставляйте без изменений.

8. Матрица ошибок и `classification_report` уже используют `Y_pred`/`Y_test` — там заменять ничего не нужно; всё продолжит работать.

9. **Сохранение модели** (раньше вы делали `joblib.dump(model, ...)`).  
   Добавьте/замените в конце ноутбука:

```python
import joblib
joblib.dump(pipe, 'pipeline.pkl')
print('✔ pipeline.pkl сохранён')
```

10. В `streamlit_app.py` убедитесь, что грузите именно новый файл:

```python
pipe = joblib.load('pipeline.pkl')
```

И не забудьте, что в этом файле для визуализации дерева используются точно такие же строчки:

```python
preprocessor = pipe.named_steps['preprocessor']
rf_model     = pipe.named_steps['classifier']
feature_names = preprocessor.get_feature_names_out()
```

──────────────────────────────────────────
Что получится в итоге
──────────────────────────────────────────
• На этапе обучения весь препроцессинг живёт внутри `Pipeline`;  
• При инференсе (Streamlit) вы подаёте «сырые» x1–x8, и пайплайн сам делает scaling/one-hot → никаких ошибок про `feature names`;  
• Визуализация дерева, матрицы ошибок и отчёт по метрикам работают без перестроения колоночных названий, потому что мы достаём их прямо из шагов пайплайна.

Используйте этот план — и модель + визуализации останутся функциональными, а ошибка с «unseen feature names» исчезнет окончательно.
