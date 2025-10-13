import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

# Указываем путь к нашему кастомному импутеру
import sys
sys.path.append('../src')
from custom_imputer import KZImputer

# Настройки для графиков
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 7)

# --- 2. ПРОВЕРКА НА СИНТЕТИЧЕСКИХ ДАННЫХ ---
print("--- Проверка на синтетических данных ---")

# Создаем простой ряд
data = np.arange(1, 31, dtype=float)
s_synth = pd.Series(data)

# Создаем пропуски разных типов
s_gappy = s_synth.copy()
s_gappy.iloc[0] = np.nan # 1-gap left
s_gappy.iloc[14] = np.nan # 1-gap middle
s_gappy.iloc[29] = np.nan # 1-gap right
s_gappy.iloc[3:5] = np.nan # 2-gap middle
s_gappy.iloc[20:25] = np.nan # 5-gap middle

print("Исходный ряд с пропусками:")
print(s_gappy.values)

# Применяем наш импутер
imputer_kz = KavunZamulaImputer(max_gap_size=5)
s_imputed = imputer_kz.transform(s_gappy)

print("\nРяд после импутации:")
print(s_imputed.values)

# --- 3. РАБОТА С РЕАЛЬНЫМИ ДАННЫМИ ---
print("\n--- Анализ на реальных данных ---")
df = pd.read_csv('../data/T1.csv')
df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='%d %m %Y %H:%M')
df = df.set_index('Date/Time')
# Возьмем только интересующую нас колонку
power_series = df['ActivePower (kW)']

# Найдем непрерывный кусок данных без пропусков для нашего эксперимента
clean_slice = power_series.loc['2018-01-15':'2018-01-25'].dropna()
print(f"Взят чистый срез данных длиной {len(clean_slice)} точек.")

# Создадим искусственные пропуски
np.random.seed(42)
test_series = clean_slice.copy()
# 10 одиночных пропусков
for _ in range(10):
    idx = np.random.randint(10, len(test_series) - 10)
    test_series.iloc[idx] = np.nan

# 3 пропуска по 3 значения
for _ in range(3):
    idx = np.random.randint(10, len(test_series) - 15)
    test_series.iloc[idx:idx+3] = np.nan

# 2 пропуска по 5 значений
for _ in range(2):
    idx = np.random.randint(10, len(test_series) - 20)
    test_series.iloc[idx:idx+5] = np.nan

# Сохраним индексы, где мы создали пропуски
missing_indices = test_series.isna()
print(f"Всего создано {missing_indices.sum()} пропущенных значений.")


# --- 4. СРАВНИТЕЛЬНЫЙ АНАЛИЗ ---

# Методы для сравнения
imputers = {
    "Kavun-Zamula": KavunZamulaImputer(max_gap_size=5),
    "Mean": SimpleImputer(strategy='mean'),
    "Median": SimpleImputer(strategy='median'),
    "Forward Fill": 'ffill',
    "Backward Fill": 'bfill',
    "Linear Interpolate": 'linear',
    "Spline Interpolate": 'spline',
    "KNN (k=5)": KNNImputer(n_neighbors=5),
    "IterativeImputer": IterativeImputer(max_iter=10, random_state=0)
}

results = {}

# Для импутеров, которым нужен 2D массив, преобразуем данные
test_series_2d = test_series.to_frame()

for name, imputer in imputers.items():
    imputed_series = None
    if isinstance(imputer, str):
        if name in ["Linear Interpolate", "Spline Interpolate"]:
             imputed_series = test_series.interpolate(method=name, order=3 if name == "Spline Interpolate" else None)
        else: # ffill, bfill
            imputed_series = test_series.fillna(method=imputer)
    elif name == "Kavun-Zamula":
        imputed_series = imputer.transform(test_series)
    else: # Scikit-learn imputers
        imputed_data = imputer.fit_transform(test_series_2d)
        imputed_series = pd.Series(imputed_data.flatten(), index=test_series.index)

    # Вычисляем RMSE только на тех значениях, которые были пропущены
    true_values = clean_slice[missing_indices]
    imputed_values = imputed_series[missing_indices]
    rmse = np.sqrt(mean_squared_error(true_values, imputed_values))
    results[name] = rmse
    print(f"RMSE для '{name}': {rmse:.4f}")

# Визуализация результатов
results_df = pd.DataFrame.from_dict(results, orient='index', columns=['RMSE']).sort_values('RMSE')

plt.figure(figsize=(12, 8))
ax = sns.barplot(x=results_df.index, y=results_df['RMSE'])
ax.set_title('Сравнение методов импутации по RMSE')
ax.set_ylabel('Root Mean Squared Error (RMSE)')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f')
plt.tight_layout()
plt.show()

# Визуализация одного из пропусков
# Найдем 5-пропуск
gap_start = test_series.rolling(5).apply(lambda x: x.isna().all()).idxmax()
gap_start_index = test_series.index.get_loc(gap_start) - 4

plot_slice = slice(gap_start_index - 10, gap_start_index + 15)

plt.figure(figsize=(15, 8))
plt.plot(clean_slice.iloc[plot_slice], 'o-', label='Оригинальные данные', color='gray', alpha=0.7)
plt.plot(test_series.iloc[plot_slice], 's', markersize=10, label='Пропуски', color='red')

# Нарисуем результаты лучших методов
best_methods = results_df.head(3).index
for name in best_methods:
    if isinstance(imputers[name], str):
        if name in ["Linear Interpolate", "Spline Interpolate"]:
            imputed = test_series.interpolate(method=name, order=3 if name=="Spline Interpolate" else None)
        else:
            imputed = test_series.fillna(method=imputers[name])
    elif name == "Kavun-Zamula":
        imputed = KavunZamulaImputer(max_gap_size=5).transform(test_series)
    else:
        imputed_data = imputers[name].fit_transform(test_series_2d)
        imputed = pd.Series(imputed_data.flatten(), index=test_series.index)
        
    plt.plot(imputed.iloc[plot_slice], '.-', label=f'{name} (RMSE: {results[name]:.2f})')

plt.title('Визуальное сравнение импутации для 5-значного пропуска')
plt.legend()
plt.ylabel('ActivePower (kW)')
plt.show()