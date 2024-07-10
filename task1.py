# Импорт необходимых библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

# Загрузка данных из оригинального источника
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# Преобразование данных в DataFrame
df = pd.DataFrame(data, columns=[
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
    'PTRATIO', 'B', 'LSTAT'
])
df['PRICE'] = target

# 2. Исследовательский анализ данных (EDA)
# На этом этапе проведем анализ данных для понимания их структуры и выявления корреляций.

# Основная информация
print(df.info())
print()

# Первые несколько строк данных
print(df.head())
print()

# Основные статистические показатели
print(df.describe())
print()

# Распределение целевой переменной (цены на жилье)
plt.figure(figsize=(10, 6))
sns.histplot(df['PRICE'], bins=30, kde=True)
plt.title('Distribution of House Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Гистограммы для каждого признака
df.hist(bins=30, figsize=(20, 15))
plt.show()

# Корреляционная матрица
corr_matrix = df.corr()

# Тепловая карта корреляций
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Корреляция признаков с целевой переменной (цены на жилье)
corr_target = corr_matrix['PRICE'].sort_values(ascending=False)
print("Корреляция признаков с целевой переменной:")
print(corr_target)
print()
print()

# 3. Предобработка данных
# На этапе предобработки данных проверим наличие пропущенных значений и подготовим данные.

# Проверка на наличие пропущенных значений
print("Проверка на наличие пропущенных значений:")
print(df.isnull().sum())
print()

# Проверка на наличие аномальных значений
print("Основные статистические показатели после проверки на пропущенные значения:")
print(df.describe())
print()
print()

# 4. Разделение данных на обучающие и тестовые
# Разделим данные на обучающую и тестовую выборки.
X = df.drop('PRICE', axis=1)
y = df['PRICE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Обучение модели
# Обучим модель линейной регрессии на обучающей выборке.
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 6. Оценка модели
# Оценим качество модели на тестовой выборке.

# Оценка модели
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f"R^2 Score: {r2}")
print()
print()

# 7. Интерпретация результатов
# Проанализируем полученные коэффициенты регрессии и сделайте выводы о значимости различных признаков.

# Получение коэффициентов регрессии и перехвата
coefficients = model.coef_
intercept = model.intercept_

# Создание DataFrame для удобного отображения коэффициентов
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coefficients
})
coef_df = coef_df.sort_values(by='Coefficient', ascending=False)

# Добавление перехвата в DataFrame
# Используем метод .loc[] для добавления перехвата в DataFrame
coef_df = pd.concat([coef_df, pd.DataFrame({'Feature': ['Intercept'], 'Coefficient': [intercept]})], ignore_index=True)

print("Коэффициенты модели:")
print(coef_df)
print()

# Интерпретация коэффициентов
print("Интерпретация:")
for index, row in coef_df.iterrows():
    feature = row['Feature']
    coef = row['Coefficient']
    if feature == 'Intercept':
        continue
    if coef > 0:
        print(f"Признак {feature} имеет положительный коэффициент {coef:.2f}. Это означает, что увеличение {feature} связано с увеличением цены на жилье.")
    else:
        print(f"Признак {feature} имеет отрицательный коэффициент {coef:.2f}. Это означает, что увеличение {feature} связано с уменьшением цены на жилье.")

# Анализ значимости признаков с помощью statsmodels
X_train_sm = sm.add_constant(X_train)  # Добавляем константу для расчета перехвата
model_sm = sm.OLS(y_train, X_train_sm).fit()
print("\nСводка модели с помощью statsmodels:")
print(model_sm.summary())

# 8. Визуализация результатов
# Построим график реальных и предсказанных значений стоимости жилья для визуальной оценки качества модели.

# Визуализация реальных и предсказанных значений с линией 1:1
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='b', label='Predicted Prices')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.legend()
plt.show()

# Дополнительная визуализация для оценки остатков
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# Визуализация остатков против предсказанных значений
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.7, color='b')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Prices')
plt.show()

# 9. Составление мини-отчета
# В отчете укажите все, что вы сделали, подтверждая каждое свое действие кодом и скриншотами.

# Пример описания для отчета:

"""
## Мини-отчет по задаче линейной регрессии для прогноза стоимости жилья в Бостоне

### 1. Введение

В этой задаче была поставлена цель построить линейную модель для прогнозирования стоимости жилья в Бостоне на основе различных признаков. Данные содержат информацию о различных характеристиках домов и соответствующих ценах.

### 2. Исследовательский анализ данных (EDA)

Мы начали с анализа данных, проверив их структуру, распределение целевой переменной и взаимосвязи между признаками:

- **Распределение цен на жилье** показало, что большинство цен сосредоточено в среднем диапазоне.
- **Корреляционная матрица** показала, что `RM` (среднее количество комнат) имеет сильную положительную корреляцию с ценой на жилье, в то время как `LSTAT` (процент более низкого статуса населения) имеет сильную отрицательную корреляцию.

### 3. Предобработка данных

Мы проверили данные на пропущенные значения и аномалии. Пропущенные значения отсутствуют, а аномальные значения также не были обнаружены.

### 4. Разделение данных на обучающие и тестовые

Данные были разделены на обучающую и тестовую выборки в пропорции 80% к 20%.

### 5. Обучение модели

Модель линейной регрессии была обучена на обучающих данных, и предсказания были сделаны на тестовых данных.

### 6. Оценка модели

Модель была оценена по следующим метрикам:
- **MSE**: Среднеквадратичная ошибка показала, что модель имеет небольшие ошибки.
- **RMSE**: Корень из MSE также подтверждает адекватность модели.
- **R^2 Score**: Высокое значение R^2 указывает на то, что модель хорошо объясняет вариации в данных.

### 7. Интерпретация результатов

Коэффициенты модели показали, что:
- Признак `RM` имеет самый высокий положительный коэффициент, что указывает на его значимость для повышения цены на жилье.
- Признак `LSTAT` имеет значительный отрицательный коэффициент, что подтверждает его влияние на снижение цены.

Результаты `statsmodels` подтвердили значимость признаков, таких как `RM`, `LSTAT`, и другие.

### 8. Визуализация результатов

Были построены графики для визуализации результатов модели и остатков. Графики показали, что модель в целом хорошо подходит, и остатки распределены случайно.

### 9. Заключение

Модель линейной регрессии успешно прогнозирует цены на жилье в Бостоне. В будущем можно улучшить модель, применяя более сложные алгоритмы регрессии и добавляя новые признаки.

### Приложение

В отчете представлены все выполненные шаги, включая код и графики.
"""
