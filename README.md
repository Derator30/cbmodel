# CatBoost Sentiment Analysis

Анализ тональности отзывов IMDB с помощью CatBoost.

## 📊 Метрики

| Модель    | Accuracy | AUC  | Precision | Recall |
|-----------|----------|------|-----------|--------|
| CatBoost  | ~89%     | ~94% | ~90%      | ~88%   |

## 🛠 Технологии

- **Python** — основной язык
- **CatBoost** — обучение модели с нативной обработкой текста
- **Pandas, NumPy, Scikit-learn** — обработка данных, метрики
- **Matplotlib** — визуализация метрик и матрицы ошибок

## 📁 Структура проекта

<pre>
Корень проекта
├── dataset.csv                Датасет IMDB
├── main.py                    Обучение + визуализация
├── predict.py                 Интерактивные предсказания
├── sentiment_model_best.cbm   Сохранённая модель
└── README.md                  Документация
</pre>

## 📦 Установка зависимостей

```bash
pip install pandas catboost scikit-learn matplotlib
```

## 🧠 Обучение

```bash
python main.py
```

В main.py уже есть перебор гиперпараметров + визуализация метрик: **Accuracy**, **Precision**, **Recall**, **AUC**, **Logloss**

## 🚀 Запуск

```bash
python predict.py
```

### Это демонстрационный проект.
