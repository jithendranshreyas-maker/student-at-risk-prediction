import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import plotly.express as px

if "history" not in st.session_state:
    st.session_state.history=[]

def reset_inputs():
    st.session_state.attendance=75.0
    st.session_state.internal_marks=50.0
    st.session_state.assignment_marks=50.0
    st.session_state.study_hours=5.0
    st.session_state.previous_gpa=7.0
    st.session_state.previous_failures=0
    st.session_state.department="CSE"
    st.session_state.gender="Female"

model=joblib.load("student_risk_model.pkl")
X_train=joblib.load("X_train_transformed.pkl")
X_test=joblib.load("X_test.pkl")
y_test=joblib.load("y_test.pkl")



explainer=shap.LinearExplainer(
    model.named_steps["model"],
    X_train
)

st.title("Student At-Risk Prediction System")

with st.sidebar:
    st.header("Project Information")
    st.write("Student At-Risk Prediction System")
    st.write("Machine Learning Project")
    
    st.markdown("### Model")
    st.write("Logistic Regression")

    st.markdown("### Features")
    st.write("• Attendance")
    st.write("• Internal Marks")
    st.write("• Assignment Marks")
    st.write("• Study Hours")
    st.write("• Previous GPA")
    st.write("• Previous Failures")
    st.write("• Department")
    st.write("• Gender")

    st.markdown("### Status")
    st.success("Model Ready")

    st.markdown("### Version")
    st.write("v1.0")

st.write(
    "An end-to-end machine learning application that predicts "
    "whether a student may be academically at risk based on "
    "academic performance and student characteristics."
)

st.divider()

st.write("Enter the student's details below:")

attendance=st.number_input("Attendance: (%)",min_value=0.0,max_value=100.0,value=75.0,key="attendance")
internal_marks=st.number_input("Internal Marks:",min_value=0.0,max_value=100.0,value=50.0,key="internal_marks")
assignment_marks=st.number_input("Assignment Marks:",min_value=0.0,max_value=100.0,value=50.0,key="assignment_marks")
study_hours=st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    key="study_hours"
)

previous_gpa=st.number_input(
    "Previous GPA",
    min_value=0.0,
    max_value=10.0,
    value=7.0,
    key="previous_gpa"
)

previous_failures=st.number_input(
    "Previous Failures",
    min_value=0,
    max_value=10,
    value=0,
    key="previous_failures"
)

department=st.selectbox(
    "Department",
    ["CSE","ECE","EEE","IT","ME"],
    key="department"
)

gender=st.selectbox(
    "Gender",
    ["Female","Male"],key="gender")

st.button("Reset Inputs",on_click=reset_inputs)

if st.button("Predict"):

    
    st.subheader("Student Information")

    student_info=pd.DataFrame({
        "Feature":[
            "Attendance",
            "Internal Marks",
            "Assignment Marks",
            "Study Hours",
            "Previous GPA",
            "Previous Failures",
            "Department",
            "Gender"
        ],
        "Value":[
            f"{attendance}%",
            internal_marks,
            assignment_marks,
            study_hours,
            previous_gpa,
            previous_failures,
            department,
            gender
        ]
    })

    st.table(student_info)

    input_data=pd.DataFrame({
        "Attendance":[attendance],
        "InternalMarks":[internal_marks],
        "AssignmentMarks":[assignment_marks],
        "StudyHours":[study_hours],
        "PreviousGPA":[previous_gpa],
        "PreviousFailures":[previous_failures],
        "Department":[department],
        "Gender":[gender]
    })

    prediction=model.predict(input_data)[0]
    probability=model.predict_proba(input_data)[0][1]

    if probability >= 0.8 or probability <= 0.2:
        confidence="High"
    elif probability >= 0.6 or probability <= 0.4:
        confidence="Moderate"
    else:
        confidence="Low"

    st.write(f"Model Confidence: {confidence}")

    st.session_state.history.append({
    "Attendance":attendance,
    "Internal Marks":internal_marks,
    "Assignment Marks":assignment_marks,
    "Study Hours":study_hours,
    "Previous GPA":previous_gpa,
    "Previous Failures":previous_failures,
    "Department":department,
    "Gender":gender,
    "Probability":f"{probability:.2%}",
    "Prediction":"At Risk" if prediction==1 else "Not At Risk"
    })

    prediction=model.predict(input_data)[0]
    probability=model.predict_proba(input_data)[0][1]

    transformed_input=model.named_steps["preprocessor"].transform(input_data)

    shap_values=explainer(transformed_input)

    if probability>=0.7:
        st.error("High Risk")
    elif probability>=0.4:
        st.warning("Moderate Risk")
    else:
        st.success("Low Risk")

    st.write(f"Probability of being at risk: {probability:.2%}")

    fig,ax=plt.subplots(figsize=(8,1.2))
    ax.barh(["Risk"],[probability],height=0.08)
    ax.text(probability,0,f" {probability:.1%}",va="center")
    ax.set_xlim(0,1)
    ax.set_xlabel("Probability")
    ax.set_title("At-Risk Probability")
    st.pyplot(fig)
    plt.close()

    st.caption("0% = Low Risk     |     50% = Moderate     |     100% = High Risk")

    st.subheader("Prediction Explanation")

    feature_names=model.named_steps["preprocessor"].get_feature_names_out()

    shap_explanation=shap.Explanation(
        values=shap_values.values.flatten(),
        base_values=shap_values.base_values,
        data=transformed_input[0],
        feature_names=feature_names
    )

    shap.plots.waterfall(shap_explanation,show=False)
    st.pyplot(plt.gcf())
    plt.close()

    st.subheader("Model Performance")

    metrics=pd.DataFrame({
        "Metric":["ROC-AUC","F1 Score","Precision","Recall"],
        "Score":[0.888,0.678,0.588,0.800]
    })

    st.table(metrics)


    st.subheader("Prediction Interpretation")

    if probability>=0.7:
        st.write(
            "The model estimates a relatively high probability that "
            "this student may be academically at risk."
        )
    elif probability>=0.4:
        st.write(
            "The model estimates a moderate probability that "
            "this student may be academically at risk."
        )
    else:
        st.write(
            "The model estimates a relatively low probability that "
            "this student may be academically at risk."
        )

    if prediction==1:
        st.write("Classification: At Risk")
    else:
        st.write("Classification: Not At Risk")


    report=pd.DataFrame({
        "Feature":[
            "Attendance",
            "Internal Marks",
            "Assignment Marks",
            "Study Hours",
            "Previous GPA",
            "Previous Failures",
            "Department",
            "Gender",
            "Risk Probability",
            "Classification"
        ],
        "Value":[
            attendance,
            internal_marks,
            assignment_marks,
            study_hours,
            previous_gpa,
            previous_failures,
            department,
            gender,
            f"{probability:.2%}",
            "At Risk" if prediction==1 else "Not At Risk"
        ]
    })

    report_csv=report.to_csv(index=False)

    st.download_button(
        "Download Prediction Report",
        report_csv,
        "student_prediction_report.csv",
        "text/csv"
    )


    st.subheader("Confusion Matrix")

    y_pred=model.predict(X_test)

    cm=confusion_matrix(y_test,y_pred)
    fig,ax=plt.subplots()
    sns.heatmap(cm,annot=True,fmt="d",xticklabels=["Not At Risk","At Risk"],yticklabels=["Not At Risk","At Risk"],ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)
    plt.close()

    st.subheader("Feature Importance")

    feature_names=model.named_steps["preprocessor"].get_feature_names_out()
    coefficients=model.named_steps["model"].coef_[0]

    importance=pd.DataFrame({
        "Feature":feature_names,
        "Importance":coefficients
    })

    importance["Absolute Importance"]=importance["Importance"].abs()
    importance=importance.sort_values(
        "Absolute Importance",
        ascending=False
    )

    fig,ax=plt.subplots()

    importance_top=importance.head(10).sort_values("Importance")

    ax.barh(
        importance_top["Feature"],
        importance_top["Importance"]
    )

    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

    st.pyplot(fig)
    plt.close()

st.subheader("Model Comparison")

st.caption(
    "F1 scores shown are from the evaluated model configurations "
    "used during experimentation."
)

comparison=pd.DataFrame({
    "Model":[
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost"
    ],
    "F1 Score":[
        0.678,
        0.333,
        0.541,
        0.579
    ]
})

fig=px.bar(
    comparison,
    x="Model",
    y="F1 Score",
    range_y=[0,1]
)

st.plotly_chart(fig,use_container_width=True)

st.subheader("Model Evaluation")

evaluation=pd.DataFrame({
    "Metric":[
        "F1 Score",
        "Precision",
        "Recall",
        "ROC-AUC"
    ],
    "Score":[
        0.678,
        0.588,
        0.800,
        0.882
    ]
})

st.dataframe(
    evaluation,
    use_container_width=True
)

st.subheader("Prediction History")

history_df=pd.DataFrame(st.session_state.history)

if not history_df.empty:
    history_df.insert(0,"No.",range(1,len(history_df)+1))
    st.dataframe(history_df,use_container_width=True)

if st.button("Clear History"):
    st.session_state.history.clear()
    st.rerun()
else:
    st.write("No predictions yet.")

if not history_df.empty:
    csv=history_df.to_csv(index=False)

    st.download_button(
        "Download Prediction History",
        csv,
        "prediction_history.csv",
        "text/csv"
    )

st.subheader("Dataset Information")

dataset_info=pd.DataFrame({
    "Property":[
        "Total Samples",
        "Features",
        "Target",
        "At Risk",
        "Not At Risk"
    ],
    "Value":[
        500,
        8,
        "AtRisk",
        "25%",
        "75%"
    ]
})

st.dataframe(
    dataset_info,
    use_container_width=True
)

st.subheader("Machine Learning Workflow")

workflow=[
    "1. Data Collection",
    "2. Data Cleaning",
    "3. Exploratory Data Analysis",
    "4. Feature Engineering",
    "5. Data Preprocessing",
    "6. Model Training",
    "7. Model Evaluation",
    "8. Hyperparameter Tuning",
    "9. SHAP Explainability",
    "10. Prediction"
]

for step in workflow:
    st.write(step)

st.subheader("Project Architecture")

st.write(
    "Dataset → Data Cleaning → EDA → Preprocessing → "
    "Model Training → Evaluation → Hyperparameter Tuning → "
    "SHAP Explainability → Streamlit Prediction"
)

st.subheader("Model Details")

model_details=pd.DataFrame({
    "Property":[
        "Algorithm",
        "Preprocessing",
        "Numerical Features",
        "Categorical Features",
        "Class Balancing",
        "Explainability"
    ],
    "Value":[
        "Logistic Regression",
        "StandardScaler + OneHotEncoder",
        "6",
        "2",
        "class_weight='balanced'",
        "SHAP"
    ]
})

st.dataframe(
    model_details,
    use_container_width=True
)

st.subheader("Project Description")

st.write(
    "This project demonstrates an end-to-end machine learning pipeline "
    "for identifying students who may be academically at risk. "
    "The system performs data preprocessing, exploratory analysis, "
    "model training, evaluation, hyperparameter tuning, and model "
    "explainability using SHAP. A Streamlit interface allows users "
    "to enter student information and receive an individual prediction."
)

st.subheader("About the Model")

st.write(
    "This system uses Logistic Regression to predict whether a student "
    "is academically at risk based on attendance, internal marks, "
    "assignment marks, study hours, previous GPA, previous failures, "
    "department, and gender."
)

st.write(
    "The model uses preprocessing with StandardScaler for numerical "
    "features and OneHotEncoder for categorical features."
)

st.write(
    "Model evaluation includes F1 Score, Precision, Recall, ROC-AUC, "
    "cross-validation, confusion matrix, and SHAP-based explainability."
)

st.subheader("Model")

with open("student_risk_model.pkl","rb") as file:
    model_file=file.read()

st.download_button(
    "Download Trained Model",
    model_file,
    "student_risk_model.pkl",
    "application/octet-stream"
)

st.subheader("Model Limitations")

st.write(
    "This model is intended for educational and demonstration purposes. "
    "Predictions are based on patterns learned from the available dataset "
    "and should not be treated as definitive academic decisions."
)

st.write(
    "The model's predictions may be affected by dataset size, feature "
    "quality, class imbalance, and differences between the training data "
    "and real-world students."
)