import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import holidays

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor 


# ================== Display options ==================
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", None)

# ================== File paths ==================
TRIP_CSV_PATH = r"C:\Users\ALIENWARE\Desktop\divvy_trip_data_202312.csv"
WEATHER_CSV_PATH = r"C:\Users\ALIENWARE\Desktop\data.csv"
COMMUNITY_GEO_PATH = r"C:\Users\ALIENWARE\Desktop\Boundaries_-_Community_Areas_20251130.geojson"

# ================== 1) Load trip data ==================
trip = pd.read_csv(TRIP_CSV_PATH, low_memory=False)

# Time handling: coerce, drop NA, floor to hour
trip["started_at"] = pd.to_datetime(trip["started_at"], errors="coerce")
trip = trip.dropna(subset=["started_at"])
trip["started_at"] = trip["started_at"].dt.floor("h")

# Time features
trip["hour"] = trip["started_at"].dt.hour
trip["day_of_week"] = trip["started_at"].dt.dayofweek
trip["is_weekend"] = (trip["day_of_week"] >= 5).astype(int)

# Holiday flag (US, 2023)
us_holidays = holidays.US(years=[2023])
trip["is_holiday"] = trip["started_at"].dt.date.map(lambda d: 1 if d in us_holidays else 0)

# ================== 2) Weather (Dec 2023) ==================
weather = pd.read_csv(WEATHER_CSV_PATH)
weather_202312 = weather[(weather["YEAR"] == 2023) & (weather["MO"] == 12)].copy()

# Build hourly timestamp
weather_202312["datetime_hour"] = pd.to_datetime(
    weather_202312[["YEAR", "MO", "DY", "HR"]].rename(
        columns={"YEAR": "year", "MO": "month", "DY": "day", "HR": "hour"}
    )
)

# Simple weather classifier
def classify_weather(row):
    prcp = row["PRCP"]
    hmdt = row["HMDT"]
    if prcp > 0:
        return "Rainy"
    elif hmdt >= 70:
        return "Cloudy"
    else:
        return "Sunny"

weather_202312["weather_type"] = weather_202312.apply(classify_weather, axis=1)
weather_simple = weather_202312[["datetime_hour", "weather_type"]].drop_duplicates()

# Join weather by hour
trip = trip.merge(weather_simple, left_on="started_at", right_on="datetime_hour", how="left")
trip["weather_type"] = trip["weather_type"].fillna("Unknown")

# ================== 3) Spatial join to community areas ==================
community = gpd.read_file(COMMUNITY_GEO_PATH).to_crs("EPSG:4326")
community_small = community[["area_numbe", "community", "geometry"]]

# Drop rows without coordinates
trip = trip.dropna(subset=["start_lat", "start_lng"])

# Convert to GeoDataFrame
gtrip = gpd.GeoDataFrame(
    trip.copy(),
    geometry=gpd.points_from_xy(trip["start_lng"], trip["start_lat"]),
    crs="EPSG:4326"
)

# Spatial join (points within polygons)
gmerged = gpd.sjoin(gtrip, community_small, how="left", predicate="within")

# Rename columns and remove geo artifacts
gmerged = gmerged.rename(columns={"area_numbe": "community_area", "community": "community_name"})
df_final = pd.DataFrame(gmerged.drop(columns=["geometry", "index_right"], errors="ignore"))

# Drop unused fields to keep the table compact
cols_to_drop = [
    "ride_id", "start_station_name", "start_station_id",
    "end_station_id", "end_station_name", "end_lat", "end_lng",
    "ended_at", "start_lat", "start_lng", "day_of_week",
    "started_at", "datetime_hour"
]
df_final = df_final.drop(columns=cols_to_drop, errors="ignore")

# Remove rows without a matched community
df_final = df_final.dropna(subset=["community_area"])

# ================== 4) Build targets & aggregate ==================
df = df_final.copy()
df["community_area"] = df["community_area"].astype(str)

group_cols = ["community_area", "hour", "is_weekend", "is_holiday", "weather_type"]

# Four disjoint indicator columns
df["member_classic_bike"] = ((df["member_casual"] == "member") & (df["rideable_type"] == "classic_bike")).astype(int)
df["member_electric_bike"] = ((df["member_casual"] == "member") & (df["rideable_type"] == "electric_bike")).astype(int)
df["casual_classic_bike"] = ((df["member_casual"] == "casual") & (df["rideable_type"] == "classic_bike")).astype(int)
df["casual_electric_bike"] = ((df["member_casual"] == "casual") & (df["rideable_type"] == "electric_bike")).astype(int)

agg_cols = ["member_classic_bike", "member_electric_bike", "casual_classic_bike", "casual_electric_bike"]

agg_df = (
    df.groupby(group_cols, as_index=False)[agg_cols]
      .sum()
)

agg_df["ride_count_total"] = agg_df[agg_cols].sum(axis=1)

# ================== 5) Merge low-activity areas ==================
community_trips = agg_df.groupby("community_area")["ride_count_total"].sum().sort_index()
print("\n=== Total rides per community ===")
print(community_trips)

MIN_RIDES_PER_COMMUNITY = 100
low_activity_areas = community_trips[community_trips < MIN_RIDES_PER_COMMUNITY].index
print("\nCommunities merged into 'low_activity_area':")
print(list(low_activity_areas))

agg_df["community_area"] = agg_df["community_area"].replace(low_activity_areas, "low_activity_area")

# ================== 6) Features & targets ==================
feature_cols = ["community_area", "hour", "is_weekend", "is_holiday", "weather_type"]
target_cols = [
    "member_classic_bike",
    "member_electric_bike",
    "casual_classic_bike",
    "casual_electric_bike",
    "ride_count_total"
]

X = agg_df[feature_cols].copy()
y = agg_df[target_cols].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_features = ["hour", "is_weekend", "is_holiday"]
cat_features = ["community_area", "weather_type"]

# NOTE: if memory is tight, set sparse_output=True and ensure estimators accept CSR.
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
    ]
)

def run_model(model_name, estimator, param_grid):
    """
    Unified pipeline:
        preprocess -> MultiOutputRegressor(estimator)
    """
    print(f"\n================= {model_name} =================")

    def collect_metrics(y_true, y_pred, label):
        rows = []
        for i, t in enumerate(target_cols):
            r2 = r2_score(y_true.iloc[:, i], y_pred[:, i])
            rmse = np.sqrt(mean_squared_error(y_true.iloc[:, i], y_pred[:, i]))
            rows.append({"Model": f"{model_name} ({label})", "Target": t, "R2": r2, "RMSE": rmse})
        return pd.DataFrame(rows)

    # -------- Default model --------
    pipe_default = Pipeline([
        ("preprocess", preprocess),
        ("regressor", MultiOutputRegressor(estimator, n_jobs=-1))
    ])

    pipe_default.fit(X_train, y_train)
    y_train_pred = pipe_default.predict(X_train)
    y_test_pred  = pipe_default.predict(X_test)

    df_test_default = collect_metrics(y_test, y_test_pred, "default-test")
    print("\nDefault model summary:")
    print(df_test_default)
    print("Avg Test R2:", df_test_default["R2"].mean())
    print("Avg Test RMSE:", df_test_default["RMSE"].mean())

    # -------- Tuned model (GridSearchCV) --------
    pipe_tuned = Pipeline([
        ("preprocess", preprocess),
        ("regressor", MultiOutputRegressor(estimator, n_jobs=-1))
    ])

    search = GridSearchCV(
        pipe_tuned,
        param_grid=param_grid,         # keys must be 'regressor__estimator__*'
        scoring="r2",
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    search.fit(X_train, y_train)

    print("\nBest Params:", search.best_params_)
    best_model = search.best_estimator_

    y_test_pred_tuned = best_model.predict(X_test)
    df_test_tuned = collect_metrics(y_test, y_test_pred_tuned, "tuned-test")

    print("\nTuned model summary:")
    print(df_test_tuned)
    print("Avg Test R2:", df_test_tuned["R2"].mean())
    print("Avg Test RMSE:", df_test_tuned["RMSE"].mean())

    return {
        "default_test": df_test_default,
        "tuned_test": df_test_tuned
    }

# ================== 7) Model zoo ==================
models = {
    "Ridge": {
        "estimator": Ridge(),
        "param_grid": {
            "regressor__estimator__alpha": [0.1, 1, 5, 10, 50]
        }
    },
    "Decision Tree": {
        "estimator": DecisionTreeRegressor(random_state=42),
        "param_grid": {
            "regressor__estimator__max_depth": [8, 12, 16, None],
            "regressor__estimator__min_samples_split": [2, 5, 10],
            "regressor__estimator__min_samples_leaf": [1, 2, 5]
        }
    },
    "Random Forest": {
        "estimator": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "param_grid": {
            "regressor__estimator__n_estimators": [100, 150, 200],
            "regressor__estimator__max_depth": [10, 20],
            "regressor__estimator__min_samples_split": [2, 5],
            "regressor__estimator__min_samples_leaf": [1, 3]
        }
    },
    "XGBoost": {
        "estimator": XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            tree_method="hist",
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6
        ),
        "param_grid": {
            "regressor__estimator__n_estimators": [200, 400],
            "regressor__estimator__max_depth": [4, 6, 8],
            "regressor__estimator__learning_rate": [0.05, 0.1],
            "regressor__estimator__subsample": [0.8, 1.0],
            "regressor__estimator__colsample_bytree": [0.8, 1.0]
        }
    },
     "Gradient Boosting": {  
        "estimator": GradientBoostingRegressor(random_state=42),
        "param_grid": {
            "regressor__estimator__n_estimators": [100, 200],
            "regressor__estimator__learning_rate": [0.05, 0.1],
            "regressor__estimator__max_depth": [2, 3],
            "regressor__estimator__min_samples_leaf": [1, 5]
        }
    }
}

results = {name: run_model(name, m["estimator"], m["param_grid"]) for name, m in models.items()}

# ============= Summary over all models =============
summary_rows = []

for model_name, res in results.items():
    df_default = res["default_test"]   # R2 / RMSE for each target (default)
    df_tuned   = res["tuned_test"]     # R2 / RMSE for each target (tuned)

    # Average across the five targets
    avg_default_r2   = df_default["R2"].mean()
    avg_default_rmse = df_default["RMSE"].mean()

    avg_tuned_r2     = df_tuned["R2"].mean()
    avg_tuned_rmse   = df_tuned["RMSE"].mean()

    summary_rows.append({
        "Model": model_name,
        "Setting": "default",
        "Avg Test R2": avg_default_r2,
        "Avg Test RMSE": avg_default_rmse
    })

    summary_rows.append({
        "Model": model_name,
        "Setting": "tuned",
        "Avg Test R2": avg_tuned_r2,
        "Avg Test RMSE": avg_tuned_rmse
    })

summary_df = pd.DataFrame(summary_rows)

print("\n================ Overall Test Summary (averaged over 5 targets) ================")
print(summary_df)



