import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load data
file_path = "C:\\Users\\chz\\Desktop\\T13.xlsx"
data = pd.read_excel(file_path)

# Define features and target variable
features = [
    '原始读段数',
    '在参考基因组上比对的比例',
    '重复读段的比例',
    '唯一比对的读段数',
    'GC含量',
    '13号染色体的Z值',
    '18号染色体的Z值',
    '21号染色体的Z值',
    'X染色体的Z值',
    'X染色体浓度',
    '13号染色体的GC含量',
    '18号染色体的GC含量',
    '21号染色体的GC含量',
    '被过滤掉读段数的比例'
]
target = '染色体的非整倍体'

# Prepare the data
X = data[features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
