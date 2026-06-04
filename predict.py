import pandas as pd
from catboost import CatBoostClassifier, Pool
import warnings

warnings.filterwarnings('ignore')

MODEL_PATH = 'sentiment_model_best.cbm'

model = CatBoostClassifier()
model.load_model(MODEL_PATH)
print(f"Модель загружена из {MODEL_PATH}.")

print("\n" + "=" * 60)
print("ИНТЕРАКТИВНЫЙ РЕЖИМ ПРЕДСКАЗАНИЯ")
print("=" * 60)
print("\nВведите текст отзыва (или 'exit' для выхода):")
print("-" * 60)

while True:
    user_input = input("\n> ").strip()

    if user_input.lower() == 'exit':
        print("\nДо свидания!")
        break

    if not user_input:
        print("Пожалуйста, введите текст отзыва.")
        continue

    df = pd.DataFrame({'review': [user_input]})
    pool = Pool(data=df, text_features=['review'])

    prediction = model.predict(pool)[0]
    probabilities = model.predict_proba(pool)[0]

    sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
    prob_positive = probabilities[1]
    prob_negative = probabilities[0]

    print("\n" + "-" * 40)
    print(f"Результат: {sentiment}")
    print(f"  Positive: {prob_positive:.2%}")
    print(f"  Negative: {prob_negative:.2%}")
    print("-" * 40)