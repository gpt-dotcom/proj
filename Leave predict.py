import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


df = pd.read_csv("/Users/macos/Documents/archive/Employee.csv")


df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})


survival_data = df[
    [
        "YearsAtCompany",
        "Attrition",
        "Department",
        "Salary",
        "StockOptionLevel",
        "OverTime",
    ]
]

kmf = KaplanMeierFitter()


kmf.fit(survival_data["YearsAtCompany"], event_observed=survival_data["Attrition"])


plt.figure(figsize=(8, 5))
kmf.plot_survival_function()
plt.title("Kaplan-Meier Survival Curve - Employee Retention")
plt.xlabel("Years at Company")
plt.ylabel("Survival Probability")
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))

for dept in survival_data["Department"].unique():
    kmf.fit(
        survival_data["YearsAtCompany"][survival_data["Department"] == dept],
        event_observed=survival_data["Attrition"][survival_data["Department"] == dept],
        label=dept,
    )
    kmf.plot_survival_function()

plt.title("Kaplan-Meier Curves - Retention by Department")
plt.xlabel("Years at Company")
plt.ylabel("Survival Probability")
plt.legend()
plt.grid()
plt.show()
