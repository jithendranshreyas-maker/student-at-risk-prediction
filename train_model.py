import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

df=pd.read_csv("student_statistics_v2.csv")

X=df.drop(columns=["FinalMarks","AtRisk"])
y=df["AtRisk"].map({"No":0,"Yes":1})

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_numeric=[
    "Attendance",
    "InternalMarks",
    "AssignmentMarks",
    "StudyHours",
    "PreviousGPA",
    "PreviousFailures"
]

X_categoric=[
    "Department",
    "Gender"
]

scaler=StandardScaler()
encoder=OneHotEncoder(handle_unknown="ignore")

preprocessor=ColumnTransformer(
    transformers=[
        ("num",scaler,X_numeric),
        ("cat",encoder,X_categoric)
    ]
)

logistic_model=Pipeline([
    ("preprocessor",preprocessor),
    ("model",LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

logistic_model.fit(X_train,y_train)

X_train_transformed=logistic_model.named_steps["preprocessor"].transform(X_train)

joblib.dump(logistic_model,"student_risk_model.pkl")
joblib.dump(X_train_transformed,"X_train_transformed.pkl")
joblib.dump(X_test,"X_test.pkl")
joblib.dump(y_test,"y_test.pkl")

print("Model training complete")
print("Train shape:",X_train.shape)
print("Test shape:",X_test.shape)
print("Files saved successfully")
