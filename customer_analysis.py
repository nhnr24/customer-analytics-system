import pandas as pd
import matplotlib.pyplot as plt

# LOAD DATA
df = pd.read_csv("customers.csv")
print("Original Data:")
print(df)

# DATA CLEANING
df = df.dropna()
df.to_csv("cleaned_customers.csv", index=False)

# FEATURE ENGINEERING
df["income_group"] = pd.cut(df["income"],
                           bins=[0,30000,45000,60000],
                           labels=["Low","Medium","High"])

# SEGMENTATION
def segment(score):
    if score >= 80:
        return "High Value"
    elif score >= 60:
        return "Medium Value"
    else:
        return "Low Value"

df["segment"] = df["spending_score"].apply(segment)

# ANALYSIS
print("\nAverage Spending:", df["spending_score"].mean())
print("\nCity-wise:")
print(df.groupby("city")["spending_score"].mean())
print("\nTop Spending City:")
print(df.groupby("city")["spending_score"].mean().idxmax())
print("\nIncome Group:")
print(df.groupby("income_group")["spending_score"].mean())
print("\nSegments:")
print(df["segment"].value_counts())

# SAVE RESULTS
df.to_csv("results.csv", index=False)

# VISUALIZATION
df.groupby("city")["spending_score"].mean().plot(kind="bar")
plt.title("Spending by City")
plt.show()

df["segment"].value_counts().plot(kind="bar")
plt.title("Customer Segments")
plt.show()

df.groupby("income_group")["spending_score"].mean().plot(kind="bar")
plt.title("Income vs Spending")
plt.show()