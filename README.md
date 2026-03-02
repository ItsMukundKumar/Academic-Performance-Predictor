🎓 Student GPA Prediction using Machine Learning
A Streamlit web application that predicts a student’s GPA using a trained KNN Regression model, based on academic behavior, parental support, and extracurricular activities.
This project demonstrates an end-to-end Machine Learning workflow:
data preprocessing → model training → model serialization → web deployment.

📌 Project Overview
The application takes multiple student-related inputs such as:
Weekly study time
Absences
Tutoring support
Parental support
Extracurricular participation
and predicts the expected Grade Point Average (GPA).

The model is trained using K-Nearest Neighbors (KNN) Regression with proper feature scaling to ensure accuracy and consistency between training and deployment.

🧠 Machine Learning Details
Algorithm: KNN Regression
Scaler: StandardScaler
Target Variable: GPA
Model Accuracy: ~90% (on validation data)
Features Used for Prediction
The model expects inputs in the following order:

StudyTimeWeekly
Absences
Tutoring
ParentalSupport
Extracurricular
Sports
Music
GradeClass

⚠️ Feature order and scaling are critical for correct predictions.

🖥️ Web Application (Streamlit)

The Streamlit UI allows users to:
Enter student details through an interactive form
Predict GPA with one click
View the exact input data used for prediction
Get the same prediction as the Jupyter Notebook (no mismatch)

📂 Project Structure
student-gpa-predictor/<br>
│
├── app.py # Streamlit application<br>
├── knn_model.pkl # Trained KNN model<br>
├── scaler.pkl # Fitted StandardScaler<br>
├── requirements.txt # Dependencies<br>
└── README.md # Project documentation<br>

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/student-gpa-predictor.git
cd student-gpa-predictor
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run the Streamlit App
streamlit run app.py

The app will open in your browser automatically.

📦 Requirements
streamlit
scikit-learn
pandas
numpy

🔒 Important Notes
The model is not retrained inside the app
The scaler used in prediction is the same scaler used during training
Predictions in Jupyter Notebook and Streamlit are identical
This project is intended for educational and demonstration purposes

🚀 Future Improvements
Add GPA interpretation (Low / Average / High)
Add sample test data button
Improve UI/UX design
Add model explainability
Deploy on Streamlit Cloud

👨‍💻 Author
Mukund Shah
Machine Learning & Data Science Enthusiast
