import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score




pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

df = pd.read_csv("/Users/macos/Documents/archive/FullTable.csv")

# ---------------------------------------------------------------------
# 1.DISTRIBUTION OF promotion delay

df["Department"] = df["Department"].str.strip().astype(str)
df_enc = pd.get_dummies(df, columns=["Department"])
dummy_columns = df_enc.select_dtypes(include=["bool"]).columns
df_enc[dummy_columns] = df_enc[dummy_columns].astype(int)
# Distribution across departments
df.groupby(["YearsSinceLastPromotion", "Department"])[
    "EmployeeID"
].count().unstack().plot(kind="bar", stacked=True)
# Median
sns.histplot(df["YearsSinceLastPromotion"], bins=10, kde=True)
plt.axvline(
    df["YearsSinceLastPromotion"].median(),
    color="red",
    linestyle="dashed",
    label="Median",
)
plt.axvline(
    df["YearsSinceLastPromotion"].quantile(0.75),
    color="blue",
    linestyle="dashed",
    label="75th Percentile",
)
plt.legend()
plt.title("Distribution of Years Since Last Promotion")
plt.show()


bins = [
    0,
    100000,
    200000,
    300000,
    400000,
    500000,
    float("inf"),
]  # Не забыть проверить можно ли атоматизировать выборку зарплат с интервалом в 100к
labels = [
    "under 100k",
    "100k-200k",
    "200k-300k",
    "300k-400k",
    "400k-500k",
    "500k+",
]
df["salary_category"] = pd.cut(df["Salary"], bins=bins, labels=labels, right=True)
df.groupby(["YearsSinceLastPromotion", "salary_category"])[
    "Salary"
].count().unstack().plot(kind="bar", stacked=True)

# ---------------------------------------------------------------------
# 2.CORRELATION relationship with promotion delay

cols = [
    "YearsSinceLastPromotion",
    "YearsAtCompany",
    "StockOptionLevel",
    "Salary",
    "SelfRating",
    "ManagerRating",
]
dummy_columns = df_enc.filter(like="Department_").columns  # Get department dummies
all_cols = cols + list(dummy_columns)  # Combine key features + department dummies

corr_matrix = df_enc[all_cols].corr()
plt.figure(figsize=(12, 8))  # Adjust figure size
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

plt.title("Correlation Heatmap for Promotion Delay Analysis")
plt.show()

# ---------------------------------------------------------------------
# 3.Promotion Delays Across Departments

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="Department", y="YearsSinceLastPromotion", palette="coolwarm")

plt.xlabel("Department")
plt.ylabel("Years Since Last Promotion")
plt.title("Comparison of Promotion Delays Across Departments")
plt.xticks(rotation=0)  # Rotate department labels for better readability
plt.show()

# ---------------------------------------------------------------------
# 4.Impact salaries on promotion delay

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df, x="Salary", y="YearsSinceLastPromotion", alpha=0.6, edgecolor=None
)
plt.xlabel("Salary")
plt.ylabel("Years Since Last Promotion")
plt.title("Salary vs. Promotion Delay")
plt.show()

# ---------------------------------------------------------------------
# 5. Linear Regression for Promotion Delay Analysis

# 1)Prepare the Data
features = [
    "YearsAtCompany",
    "Salary",
    "StockOptionLevel",
    "SelfRating",
    "ManagerRating",
]
df_enc = pd.get_dummies(
    df, columns=["Department"], drop_first=True
)  # Encode department

X = df_enc[features]  # Independent variables
y = df_enc["YearsSinceLastPromotion"]  # Target variable

# Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2)Train the Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Print regression coefficients (importance of each feature)
coefficients = pd.DataFrame(model.coef_, index=features, columns=["Coefficient"])
print(coefficients)

# 3)Evaluate Model Performance
y_pred = model.predict(X_test)  # Make predictions

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# 4)Visualizing Predictions vs. Actual Values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.xlabel("Actual Years Since Last Promotion")
plt.ylabel("Predicted Years Since Last Promotion")
plt.title("Actual vs. Predicted Promotion Delays")
plt.show()

# ------------------------------------------------------------------------
# 6.Logistic Regression

df["PromotionDelayed"] = (df["YearsSinceLastPromotion"] > 3).astype(int)
# Select features
features = [
    "YearsAtCompany",
    "Salary",
    "StockOptionLevel",
    "SelfRating",
]

X = df[features]
y = df["PromotionDelayed"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
model.coef_


rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame(
    {"Feature": features, "Importance": importances}
).sort_values(by="Importance", ascending=False)

print(feature_importance_df)
