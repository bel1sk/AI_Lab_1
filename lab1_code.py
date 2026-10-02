try:
    display
except NameError:
    display = print


import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_percentage_error

from statsmodels.stats.outliers_influence import variance_inflation_factor

sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

dataset_path = 'hyundi.xls'
df_raw = pd.read_csv(dataset_path, encoding='utf-8')

print("=" * 80)
print(f"ФАЙЛ УСПЕШНО ЗАГРУЖЕН: {dataset_path}")
print(f"Форма датасета (Shape): {df_raw.shape[0]} строк (наблюдений), {df_raw.shape[1]} столбцов (признаков)")
print("=" * 80)
print("\nПервые 5 строк исходного датасета:")
display(df_raw.head())

print("\nОбщая информация о типах данных и заполненности столбцов:")
display(df_raw.info())

df = df_raw.rename(columns=lambda col: 'tax' if 'tax' in col.lower() else col)

features = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
target = 'price'

df_selected = df[features + [target]].copy()

missing_check = pd.DataFrame({
    'Тип признака': df_selected.dtypes,
    'Количество пропусков (NaN)': df_selected.isnull().sum(),
    'Доля пропусков (%)': (df_selected.isnull().sum() / len(df_selected)) * 100
})
print("=" * 80)
print("РЕЗУЛЬТАТЫ АУДИТА ПРОПУСКОВ:")
display(missing_check)

zero_engines_df = df_selected[df_selected['engineSize'] == 0]
print(f"\nОбнаружено автомобилей с объемом двигателя 0.0 л: {len(zero_engines_df)} шт. ({len(zero_engines_df)/len(df_selected)*100:.2f}%)")
if len(zero_engines_df) > 0:
    print("Примеры записей с объемом двигателя 0.0 л:")
    display(zero_engines_df.head(3))

df_clean = df_selected[df_selected['engineSize'] > 0].reset_index(drop=True)
print(f"Размер выборки после удаления аномалий: {df_clean.shape[0]} строк, {df_clean.shape[1]} столбцов")
print("=" * 80)

stats_table = df_clean.describe().T
stats_table['skewness (асимметрия)'] = df_clean.skew()
stats_table['kurtosis (эксцесс)'] = df_clean.kurtosis()
print("\nСводная таблица описательных статистик и формы распределений:")
display(stats_table[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skewness (асимметрия)', 'kurtosis (эксцесс)']].round(2))

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

all_vars = features + [target]
plot_titles = [
    'Год выпуска (year)',
    'Пробег, мили (mileage)',
    'Дорожный налог, £ (tax)',
    'Расход топлива, mpg (mpg)',
    'Объем двигателя, л (engineSize)',
    'ЦЕНА АВТОМОБИЛЯ, £ (price) [Целевая]'
]
colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

for i, col in enumerate(all_vars):
    sns.histplot(df_clean[col], kde=True, ax=axes[i], color=colors[i], bins=30, alpha=0.55)
    axes[i].set_title(plot_titles[i], fontsize=12, fontweight='bold')
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Частота / Плотность')

    mean_val = df_clean[col].mean()
    median_val = df_clean[col].median()
    axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Среднее: {mean_val:.1f}')
    axes[i].axvline(median_val, color='black', linestyle=':', linewidth=1.5, label=f'Медиана: {median_val:.1f}')
    axes[i].legend(loc='upper right', fontsize=9)

plt.suptitle('Гистограммы распределений признаков и целевой переменной с кривыми KDE', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
for i, feature in enumerate(features):
    sns.regplot(
        data=df_clean, 
        x=feature, 
        y=target, 
        ax=axes[i], 
        scatter_kws={'alpha': 0.25, 'color': '#2b5c8f', 's': 20},
        line_kws={'color': 'crimson', 'linewidth': 2}
    )
    axes[i].set_title(f'Price vs {feature}', fontsize=12, fontweight='bold')
    axes[i].set_xlabel(feature, fontsize=11)
    axes[i].set_ylabel('Price (£)' if i == 0 else '', fontsize=11)

plt.suptitle('Парные диаграммы рассеяния зависимости цены от независимых признаков с линиями регрессионного тренда', 
             fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
plt.show()

corr_matrix = df_clean[features + [target]].corr(method='pearson')

plt.figure(figsize=(9, 7))
sns.heatmap(
    corr_matrix, 
    annot=True, 
    fmt='.3f', 
    cmap='coolwarm', 
    vmin=-1.0, 
    vmax=1.0, 
    center=0,
    square=True, 
    linewidths=1.5, 
    cbar_kws={'shrink': 0.8, 'label': 'Коэффициент линейной корреляции Пирсона (r)'}
)
plt.title('Матрица парных коэффициентов корреляции Пирсона', fontsize=14, fontweight='bold', pad=15)
plt.show()

X_features = df_clean[features]

scaler_for_vif = StandardScaler()
X_vif_input = scaler_for_vif.fit_transform(X_features)

vif_results = pd.DataFrame({
    'Признак': features,
    'Коэффициент VIF': [variance_inflation_factor(X_vif_input, i) for i in range(X_vif_input.shape[1])],
    'Вспомогательный R² (R²_aux)': [1.0 - (1.0 / variance_inflation_factor(X_vif_input, i)) for i in range(X_vif_input.shape[1])],
    'Степень коллинеарности': [
        'Умеренная коллинеарность (VIF > 2.0)' if variance_inflation_factor(X_vif_input, i) > 2.0 
        else 'Слабая зависимость (VIF < 1.5)' for i in range(X_vif_input.shape[1])
    ]
}).sort_values(by='Коэффициент VIF', ascending=False).reset_index(drop=True)

print("=" * 90)
print("РЕЗУЛЬТАТЫ РАСЧЕТА КОЭФФИЦИЕНТОВ ИНФЛЯЦИИ ДИСПЕРСИИ (VIF):")
print("=" * 90)
display(vif_results.round(4))

X = df_clean[features]
y = df_clean[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=True
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=features)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=features)

print("=" * 80)
print("ПАРАМЕТРЫ РАЗБИЕНИЯ ВЫБОРКИ:")
print(f"Размер обучающего набора (Train set, 80%): {X_train.shape[0]} строк")
print(f"Размер тестового набора (Test set, 20%):   {X_test.shape[0]} строк")
print("=" * 80)
print("\nПроверка средних значений и стандартных отклонений после Z-стандартизации (Train):")
scaling_check = pd.DataFrame({
    'Среднее значение (mu)': X_train_scaled_df.mean(),
    'Стандартное отклонение (sigma)': X_train_scaled_df.std(),
    'Минимум (min)': X_train_scaled_df.min(),
    'Максимум (max)': X_train_scaled_df.max()
})
display(scaling_check.round(4))

models_baseline = {
    'Линейная регрессия (OLS)': LinearRegression(),
    'Гребневая регрессия (Ridge, alpha=1.0)': Ridge(alpha=1.0, random_state=42),
    'Лассо регрессия (Lasso, alpha=1.0)': Lasso(alpha=1.0, random_state=42, max_iter=10000)
}

def evaluate_regression_models(models_dict, X_tr, X_te, y_tr, y_te, cv_folds=5):
    results_list = []
    fitted_models = {}
    test_predictions = {}

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for name, model in models_dict.items():

        model.fit(X_tr, y_tr)
        fitted_models[name] = model

        y_pred = model.predict(X_te)
        test_predictions[name] = y_pred

        rmse_val = root_mean_squared_error(y_te, y_pred)
        r2_val = r2_score(y_te, y_pred)
        mape_val = mean_absolute_percentage_error(y_te, y_pred) * 100.0

        cv_scores = cross_val_score(model, X_tr, y_tr, cv=kf, scoring='r2')

        results_list.append({
            'Модель': name,
            'RMSE (£)': rmse_val,
            'R² (Тест)': r2_val,
            'MAPE (%)': mape_val,
            'R² (CV-5 среднее)': cv_scores.mean(),
            'R² (CV-5 ст. откл.)': cv_scores.std()
        })

    return pd.DataFrame(results_list), fitted_models, test_predictions

df_metrics_baseline, trained_baseline, preds_baseline = evaluate_regression_models(
    models_baseline, X_train_scaled, X_test_scaled, y_train, y_test
)

print("=" * 95)
print("СВОДНАЯ ТАБЛИЦА МЕТРИК КАЧЕСТВА МОДЕЛЕЙ (ДО УСТРАНЕНИЯ МУЛЬТИКОЛЛИНЕАРНОСТИ):")
print("=" * 95)
display(df_metrics_baseline.round(4))

coef_table = pd.DataFrame({
    'Признак': features,
    'OLS (МНК)': trained_baseline['Линейная регрессия (OLS)'].coef_,
    'Ridge (L2)': trained_baseline['Гребневая регрессия (Ridge, alpha=1.0)'].coef_,
    'Lasso (L1)': trained_baseline['Лассо регрессия (Lasso, alpha=1.0)'].coef_
})

print("\nВесовые коэффициенты стандартизированных признаков:")
display(coef_table.round(2))

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

model_keys = list(preds_baseline.keys())
short_labels = ['OLS Регрессия', 'Ridge Регрессия', 'Lasso Регрессия']

for i, (m_key, label) in enumerate(zip(model_keys, short_labels)):
    y_pred = preds_baseline[m_key]
    residuals = y_test - y_pred

    axes[i].scatter(y_pred, residuals, alpha=0.35, color='#2b5c8f', edgecolor='k', s=35)

    axes[i].axhline(0, color='red', linestyle='--', linewidth=2.5, label='Опорный уровень e = 0')

    std_res = residuals.std()
    axes[i].axhline(2 * std_res, color='gray', linestyle=':', label='Граница ±2σ')
    axes[i].axhline(-2 * std_res, color='gray', linestyle=':')

    axes[i].set_title(f'График остатков: {label}', fontsize=12, fontweight='bold')
    axes[i].set_xlabel('Предсказанная цена ŷ (£)', fontsize=11)
    axes[i].set_ylabel('Остаток e = y - ŷ (£)' if i == 0 else '', fontsize=11)
    axes[i].legend(loc='upper right', fontsize=9)
    axes[i].grid(True, alpha=0.5)

plt.suptitle('Диагностика гомоскедастичности: графики остатков моделей регрессии относительно предсказанных цен', 
             fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
plt.show()

residuals_ols = y_test - preds_baseline['Линейная регрессия (OLS)']
plt.figure(figsize=(9, 4))
sns.histplot(residuals_ols, kde=True, color='teal', bins=40, alpha=0.5)
plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Нулевой уровень')
plt.title('Гистограмма распределения остатков классической линейной регрессии OLS', fontsize=13, fontweight='bold')
plt.xlabel('Величина остатка e = y - ŷ (£)', fontsize=11)
plt.ylabel('Плотность / Частота', fontsize=11)
plt.legend()
plt.tight_layout()
plt.show()

pca_full = PCA(random_state=42)
pca_full.fit(X_train_scaled)

explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
eigenvalues_lambda = pca_full.explained_variance_

fig, ax1 = plt.subplots(figsize=(10, 6))

component_names = [f'PC{i+1}' for i in range(len(explained_variance))]
x_ticks_pos = np.arange(1, len(explained_variance) + 1)

bars = ax1.bar(x_ticks_pos, explained_variance * 100.0, alpha=0.75, color='#2b5c8f', label='Индивидуальная объясненная дисперсия (%)')
ax1.set_xlabel('Главные компоненты', fontsize=12, fontweight='bold')
ax1.set_ylabel('Объясненная дисперсия компоненты (%)', fontsize=12, color='#2b5c8f', fontweight='bold')
ax1.set_xticks(x_ticks_pos)
ax1.set_xticklabels(component_names, fontsize=11)
ax1.set_ylim(0, 50)

ax2 = ax1.twinx()
ax2.plot(x_ticks_pos, cumulative_variance * 100.0, color='crimson', marker='o', linewidth=2.5, markersize=8, label='Кумулятивная дисперсия (%)')
ax2.axhline(80, color='gray', linestyle=':', linewidth=1.5, label='Порог 80% дисперсии')
ax2.axhline(90, color='darkgreen', linestyle='--', linewidth=1.5, label='Порог 90% дисперсии')
ax2.set_ylabel('Кумулятивная объясненная дисперсия (%)', fontsize=12, color='crimson', fontweight='bold')
ax2.set_ylim(0, 105)

for bar in bars:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, h + 1.0, f'{h:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.title('График каменистой осыпи (Scree Plot) и кумулятивной объясненной дисперсии', fontsize=14, fontweight='bold', pad=15)
fig.legend(loc='lower left', bbox_to_anchor=(0.14, 0.16), frameon=True, fontsize=10)
plt.tight_layout()
plt.show()

pca_analysis_table = pd.DataFrame({
    'Главная компонента': component_names,
    'Собственное значение (λ)': eigenvalues_lambda,
    'Объясненная дисперсия (%)': explained_variance * 100.0,
    'Кумулятивная дисперсия (%)': cumulative_variance * 100.0,
    'Критерий Кайзера (λ > 1)': ['Сохранить (λ > 1)' if ev > 1.0 else 'Исключить (λ < 1)' for ev in eigenvalues_lambda]
})

print("=" * 90)
print("ФАКТОРНЫЙ АНАЛИЗ: СОБСТВЕННЫЕ ЗНАЧЕНИЯ И ОБЪЯСНЕННАЯ ДИСПЕРСИЯ:")
print("=" * 90)
display(pca_analysis_table.round(4))

X_train_pca_transformed = pca_full.transform(X_train_scaled)
vif_after_pca = pd.DataFrame({
    'Компонента': component_names,
    'Коэффициент VIF': [variance_inflation_factor(X_train_pca_transformed, i) for i in range(X_train_pca_transformed.shape[1])],
    'Статус мультиколлинеарности': ['Полная ортогональность (VIF = 1.0)' for _ in range(X_train_pca_transformed.shape[1])]
})

print("\nПроверка устранения мультиколлинеарности на пространстве главных компонент (VIF):")
display(vif_after_pca.round(4))

k_comps = 3
pca_optimal = PCA(n_components=k_comps, random_state=42)

X_train_pca = pca_optimal.fit_transform(X_train_scaled)
X_test_pca = pca_optimal.transform(X_test_scaled)

models_on_pca = {
    'Линейная регрессия (PCA-3)': LinearRegression(),
    'Гребневая регрессия (Ridge-PCA, alpha=1.0)': Ridge(alpha=1.0, random_state=42),
    'Лассо регрессия (Lasso-PCA, alpha=1.0)': Lasso(alpha=1.0, random_state=42, max_iter=10000)
}

df_metrics_pca, trained_pca, preds_pca = evaluate_regression_models(
    models_on_pca, X_train_pca, X_test_pca, y_train, y_test
)

df_final_comparison = pd.concat([df_metrics_baseline, df_metrics_pca], ignore_index=True)

print("=" * 95)
print("ИТОГОВОЕ СРАВНЕНИЕ КАЧЕСТВА МОДЕЛЕЙ ДО И ПОСЛЕ СНИЖЕНИЯ РАЗМЕРНОСТИ (PCA):")
print("=" * 95)
display(df_final_comparison.round(4))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.barplot(data=df_final_comparison, x='Модель', y='R² (Тест)', palette='Blues_r', ax=axes[0])
axes[0].set_title('Коэффициент детерминации R² (Тест)', fontsize=12, fontweight='bold')
axes[0].set_ylim(0.65, 0.72)
axes[0].tick_params(axis='x', rotation=35)
for p in axes[0].patches:
    axes[0].annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', xytext=(0, 4), textcoords='offset points', fontweight='bold', fontsize=9)

sns.barplot(data=df_final_comparison, x='Модель', y='RMSE (£)', palette='Oranges_r', ax=axes[1])
axes[1].set_title('Среднеквадратическая ошибка RMSE (£)', fontsize=12, fontweight='bold')
axes[1].set_ylim(2800, 3200)
axes[1].tick_params(axis='x', rotation=35)
for p in axes[1].patches:
    axes[1].annotate(f"£{p.get_height():.1f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', xytext=(0, 4), textcoords='offset points', fontweight='bold', fontsize=9)

sns.barplot(data=df_final_comparison, x='Модель', y='MAPE (%)', palette='Reds_r', ax=axes[2])
axes[2].set_title('Средняя процентная ошибка MAPE (%)', fontsize=12, fontweight='bold')
axes[2].set_ylim(18, 25)
axes[2].tick_params(axis='x', rotation=35)
for p in axes[2].patches:
    axes[2].annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', xytext=(0, 4), textcoords='offset points', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.show()
