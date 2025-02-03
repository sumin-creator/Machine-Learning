import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt

# 特徴量とターゲットを定義する
features = ['duration_ms', 'explicit', 'danceability', 'energy', 'key',
            'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
            'liveness', 'valence', 'tempo', 'time_signature', 'track_genre', 'artists']
target = 'popularity'

# カテゴリカル特徴量のカラムを指定
categorical_features = ['key', 'mode', 'time_signature', 'track_genre', 'artists']

# 数値特徴量のカラムを指定
numeric_features = ['duration_ms', 'danceability', 'energy', 'loudness', 'speechiness',
                    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'explicit']

# 特徴量をエンコード
X = d_train[features]
y = d_train[target]

# 数値特徴量とカテゴリカル特徴量の分離
X_numeric = X[numeric_features]
X_categorical = X[categorical_features]

# 数値特徴量の相関係数を計算
correlation_matrix = X_numeric.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numeric Features')
plt.show()

# 相関係数が低い特徴量を手動で除外
# 例えば、相関係数が0.1未満の特徴量を除外する
low_correlation_features = correlation_matrix.columns[abs(correlation_matrix).max() < 0.1]
print(f'Low correlation features: {low_correlation_features}')

# 除外する特徴量を設定
features_to_remove = low_correlation_features
numeric_features = [feature for feature in numeric_features if feature not in features_to_remove]

# 重要度評価のためのLightGBMモデルの訓練
X_transformed = pd.get_dummies(X)
X_train_transformed, X_val_transformed, y_train, y_val = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# LightGBMモデルを訓練して特徴量の重要度を取得
lgb_train = lgb.Dataset(X_train_transformed, y_train)
lgb_val = lgb.Dataset(X_val_transformed, y_val, reference=lgb_train)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 80,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'lambda_l1': 0.2,
    'lambda_l2': 0.2,
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1000,
                valid_sets=[lgb_val])

# 特徴量の重要度を表示
importance = gbm.feature_importance(importance_type='split')
feature_names = X_transformed.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)

# 重要度の低い特徴量を除外する
# 例えば、重要度が5未満の特徴量を除外する
low_importance_features = importance_df[importance_df['Importance'] < 5]['Feature']
print(f'Low importance features: {low_importance_features}')

# 最終的な特徴量セット
final_features = [feature for feature in feature_names if feature not in low_importance_features]
print(f'Final feature set: {final_features}')

# パイプラインを使用して前処理を定義
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 訓練データとテストデータに4:1の比率で分割
X = d_train[final_features]
y = d_train[target]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 5-foldクロスバリデーションを実行して最良のモデルを見つける
kf = KFold(n_splits=10, shuffle=True, random_state=42)
best_rmse = float('inf')
best_model = None
best_predictions = None

for fold, (train_index, val_index) in enumerate(kf.split(X_train), start=1):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    X_train_transformed = preprocessor.fit_transform(X_train_fold)
    X_val_transformed = preprocessor.transform(X_val_fold)

    lgb_train = lgb.Dataset(X_train_transformed, y_train_fold)
    lgb_val = lgb.Dataset(X_val_transformed, y_val_fold, reference=lgb_train)

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=1000,
                    valid_sets=[lgb_val])

    y_val_pred = gbm.predict(X_val_transformed, num_iteration=gbm.best_iteration)
    rmse = mean_squared_error(y_val_fold, y_val_pred, squared=False)
    print(f'Fold {fold} - Validation RMSE: {rmse}')

    if rmse < best_rmse:
        best_rmse = rmse
        best_model = gbm
        best_predictions = gbm.predict(preprocessor.transform(d_test[final_features]), num_iteration=gbm.best_iteration)

# 最良のモデルでの予測を保存する
final_predictions = pd.DataFrame({'track_id': d_test['track_id'], 'popularity': best_predictions})
final_predictions.to_csv('predictions_lgbm_best_model.csv', index=False)

# 全ての値が0のファイルと計算対象データの差をmseで計算する
zeros = np.zeros_like(best_predictions)
mse = mean_squared_error(zeros, best_predictions)
print(f'MSE between predicted values and zeros: {mse}')
