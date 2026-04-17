import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow import keras
# ==========================================
# 단계 1: Dataset 전처리
# ==========================================

# 1-1 Data 불러오기
fires = pd.read_csv("./sanbul2district-divby100.csv", sep=",")

# 1-2 head(), info(), describe(), value_counts() 출력
print("\n===== 1-2 fires.head() =====")
print(fires.head())
print("\n===== fires.info() =====")
fires.info()
print("\n===== fires.describe() =====")
print(fires.describe())
print("\n===== fires['month'].value_counts() =====")
print(fires["month"].value_counts())
print("\n===== fires['day'].value_counts() =====")
print(fires["day"].value_counts())

# 1-3 데이터 시각화 (전체 수치형 속성 히스토그램)
fires.hist(bins=50, figsize=(15, 10))
plt.tight_layout()
plt.show()

# 1-4 특성 burned_area 왜곡 현상 개선을 위해 로그 함수(y=ln(burned_area+1)) 변환
fires['burned_area'].hist(bins=50, figsize=(8, 6))
plt.title("Before Transformation: burned_area") 
plt.show()

fires['burned_area'] = np.log(fires['burned_area'] + 1)

fires['burned_area'].hist(bins=50, figsize=(8, 6))
plt.title("After Transformation: ln(burned_area+1)") 
plt.show()

# 1-5 Scikit-Learn의 train_test_split 및 StratifiedShuffleSplit
train_set, test_set = train_test_split(fires, test_size=0.2, random_state=42)
print("\n===== test_set.head() =====")
print(test_set.head())

fires["month"].hist()
plt.show()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(fires, fires["month"]):
    strat_train_set = fires.loc[train_index]
    strat_test_set = fires.loc[test_index]

print("\nMonth category proportion (Test): \n",
      strat_test_set["month"].value_counts() / len(strat_test_set))
print("\nOverall month category proportion: \n",
      fires["month"].value_counts() / len(fires))

# 1-6 Pandas scatter_matrix() 함수를 이용한 matrix 출력
scatter_matrix(fires[['burned_area', 'max_temp', 'avg_temp', 'max_wind_speed']],
               figsize=(12, 8))
plt.show()

# 1-7 지역별로 'burned_area'에 대해 plot 하기
fires.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
           s=fires["max_temp"], label="max_temp",
           c="burned_area", cmap=plt.get_cmap("jet"), colorbar=True)
plt.show()

# 1-8 카테고리형 특성 OneHotEncoder() 인코딩/출력
fires = strat_train_set.drop(["burned_area"], axis=1)  # drop labels for training set
fires_labels = strat_train_set["burned_area"].copy()
fires_num = fires.drop(["month", "day"], axis=1)

fires_cat = fires[["month", "day"]]
cat_encoder = OneHotEncoder()
fires_cat_1hot = cat_encoder.fit_transform(fires_cat)

print("\n===== fires_cat_1hot =====")
print(fires_cat_1hot)
# ==========================================

print("\n===== cat_month_encoder.categories_ =====")
print(cat_encoder.categories_[0])
print("\n===== cat_day_encoder.categories_ =====")
print(cat_encoder.categories_[1])
print("\n===== cat_month_encoder.categories_ =====")
print(cat_encoder.categories_[0])
print("\n===== cat_day_encoder.categories_ =====")
print(cat_encoder.categories_[1])

# 1-9 Pipeline, StandardScaler를 이용한 카테고리형 특성 인코딩
print("\n\n########\n########\n##########################")
print("Now let's build a pipeline for preprocessing the numerical attributes:")

num_pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
])

num_attribs = ['longitude', 'latitude', 'avg_temp', 'max_temp', 'max_wind_speed', 'avg_wind']
cat_attribs = ['month', 'day']

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

fires_prepared = full_pipeline.fit_transform(fires)
print("\n===== fires_prepared shape =====")
print(fires_prepared.shape)

# 테스트 데이터 분리 및 전처리 적용 (모델 평가를 위해 필수)
fires_test = strat_test_set.drop(["burned_area"], axis=1)
fires_test_labels = strat_test_set["burned_area"].copy()
fires_test_prepared = full_pipeline.transform(fires_test)


# ==========================================
# 단계 2: 모델 개발 (Keras Regression MLP)
# ==========================================

X_train, X_valid, y_train, y_valid = train_test_split(
    fires_prepared, fires_labels, test_size=0.2, random_state=42
)
X_test, y_test = fires_test_prepared, fires_test_labels

np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(1)
])

model.summary()

model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.SGD(learning_rate=1e-3))

history = model.fit(
    X_train, y_train,
    epochs=200,
    validation_data=(X_valid, y_valid)
)

# Keras 모델 저장
model.save('./models/fires_model.h5')

# ==========================================
# 파이프라인 (전처리 도구) 저장
# ==========================================
import os
os.makedirs('./models', exist_ok=True)
joblib.dump(full_pipeline, './models/pipeline.pkl')
print("Pipeline saved to ./models/pipeline.pkl successfully.")


# evaluate model
X_new = X_test[:3]
predicted_log = model.predict(X_new)
print("\nnp.round(predicted_log, 2) [log(burned_area+1) 예측값]:\n",
      np.round(predicted_log, 2))

print("\nnp.round(model.predict(X_new), 2): \n",
      np.round(model.predict(X_new), 2))

predicted_actual = np.exp(predicted_log) - 1
print("\nnp.round(predicted_actual, 2) [실제 산불 예측 면적]: \n", 
      np.round(predicted_actual, 2))