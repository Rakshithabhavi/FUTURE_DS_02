import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#  1. Sample Customer Dataset 
data = {
    'CustomerID': range(1, 21),
    'Age': [25,30,35,40,45,50,55,60,28,33,
            38,43,48,53,58,27,32,37,42,47],
    'Subscription_Length': [1,2,3,4,5,6,7,8,1,2,
                            3,4,5,6,7,1,2,3,4,5],
    'Monthly_Bill': [50,60,70,80,90,100,110,120,55,65,
                     75,85,95,105,115,52,62,72,82,92],
    'Total_Usage': [100,200,300,400,500,600,700,800,150,250,
                    350,450,550,650,750,120,220,320,420,520],
    'Churn': [1,0,0,1,0,1,0,0,1,0,
              0,1,0,0,1,1,0,0,1,0]
}

df = pd.DataFrame(data)

#  2. Basic Analysis
print("=== Customer Data Overview ===")
print(df.head())
print("\n=== Churn Count ===")
print(df['Churn'].value_counts())
print("\n=== Churn Rate ===")
churn_rate = df['Churn'].mean() * 100
print(f"Churn Rate: {churn_rate:.1f}%")
print("\n=== Average Monthly Bill by Churn ===")
print(df.groupby('Churn')['Monthly_Bill'].mean())

#  3. Churn Distribution 
plt.figure(figsize=(7,5))
churn_counts = df['Churn'].value_counts()
labels = ['Retained', 'Churned']
colors = ['green', 'red']
plt.pie(churn_counts.values, labels=labels,
        autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Customer Churn Distribution')
plt.tight_layout()
plt.savefig('churn_distribution.png')
plt.show()

#  4. Age vs Churn 
plt.figure(figsize=(8,5))
sns.boxplot(x='Churn', y='Age', data=df,
            palette=['green','red'])
plt.title('Age vs Churn')
plt.xlabel('Churn (0=Retained, 1=Churned)')
plt.ylabel('Age')
plt.tight_layout()
plt.savefig('age_vs_churn.png')
plt.show()

#  5. Monthly Bill vs Churn 
plt.figure(figsize=(8,5))
sns.boxplot(x='Churn', y='Monthly_Bill', data=df,
            palette=['green','red'])
plt.title('Monthly Bill vs Churn')
plt.xlabel('Churn (0=Retained, 1=Churned)')
plt.ylabel('Monthly Bill')
plt.tight_layout()
plt.savefig('bill_vs_churn.png')
plt.show()

#  6. Subscription Length vs Churn 
plt.figure(figsize=(8,5))
sns.barplot(x='Churn', y='Subscription_Length', data=df,
            palette=['green','red'])
plt.title('Subscription Length vs Churn')
plt.xlabel('Churn (0=Retained, 1=Churned)')
plt.ylabel('Subscription Length (Months)')
plt.tight_layout()
plt.savefig('subscription_vs_churn.png')
plt.show()

#  7. Correlation Heatmap 
plt.figure(figsize=(8,6))
corr = df[['Age','Subscription_Length',
           'Monthly_Bill','Total_Usage','Churn']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()

print("\n=== Analysis Complete! Charts Saved! ===")