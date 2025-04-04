import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder #COMMENTs
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt


#### IMPORT CSV FILE ###

data_path = r"C:\Users\steph\Documents\personal_finance_tracker\data\expenses_income_summary.csv"

df = pd.read_csv(data_path) #making dataframe
#print(df.head()) #show first 5 rows

#filters to only show rows where the type is expenses
df_expenses = df[df['type'] == 'EXPENSE'].copy()


#### PREP DATA FOR TRAINING ###

#clean the amount column: remove commas and convert to float
df_expenses['amount'] = df_expenses['amount'].str.replace(',', '').astype(float)

#filters to only see relevant columns
needed_columns = df_expenses[['title', 'category', 'amount']] 
#print(needed_columns)

#Remoces rows with ANY missing values
needed_columns = needed_columns.dropna()
#print(needed_columns.head(50))

#encode "title" columns using one hot encoding
needed_columns = pd.get_dummies(needed_columns, columns=['title'])

#encode catigory column (the target)
le = LabelEncoder()
needed_columns['category_encoded'] = le.fit_transform(needed_columns['category'])

#define features and target
X = needed_columns.drop(columns=['category', 'category_encoded'])
y = needed_columns['category_encoded']

###training the model###
model = DecisionTreeClassifier(random_state=42) #select decision tree

###evaluate model performance with k fold cross validation###
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv)

print("Accuracy for each fold:", scores)
print("Average accuracy:", scores.mean())


plt.figure(figsize=(8, 5))
plt.bar([f'Fold {i+1}' for i in range(len(scores))], scores * 100, color='skyblue')
plt.axhline(scores.mean() * 100, color='green', linestyle='--', label='Average Accuracy')
plt.title('Model Accuracy per Fold (5-Fold Cross-Validation)')
plt.ylabel('Accuracy (%)')
plt.ylim(80, 100)
plt.legend()
plt.tight_layout()
plt.show()

# export graph images
plt.savefig(r"C:\Users\steph\Documents\personal_finance_tracker\results\accuracy_chart.png")
