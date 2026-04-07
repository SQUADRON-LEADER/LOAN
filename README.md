# 🚀 Loan Prediction Web App


🔗 **Live Demo:** [https://loan02.streamlit.app/](https://loan02.streamlit.app/)

📂 **Repository:** [https://github.com/SQUADRON-LEADER/LOAN](https://github.com/SQUADRON-LEADER/LOAN)

---

## 📌 Overview

This project is a **Machine Learning-based Loan Prediction Web Application** built using **Streamlit**.
It predicts whether a user is eligible for a loan based on various financial and personal details.

The application provides a simple and interactive UI where users can input their details and instantly get a prediction.

---

## 🎯 Features

* ✅ User-friendly web interface using Streamlit
* ✅ Real-time loan eligibility prediction
* ✅ Machine Learning model integration
* ✅ Handles categorical & numerical inputs
* ✅ Fast and interactive results

---

## 🧠 How It Works

1. User enters details like:

   * Income
   * Employment Status
   * Loan Amount
   * Marital Status
   * Property Area
   * Loan Purpose

2. Data is:

   * Encoded (categorical → numerical)
   * Scaled (for better model performance)

3. ML model processes the input

4. Output:

   * ✅ **Eligible for Loan**
   * ❌ **Not Eligible**

---

<img width="1903" height="855" alt="Screenshot 2026-04-07 154542" src="https://github.com/user-attachments/assets/1f69a27c-0e11-487a-94d3-d8cc91bd3913" />
<br>
<br>


<img width="1897" height="854" alt="Screenshot 2026-04-07 154549" src="https://github.com/user-attachments/assets/bf2f29af-754c-4497-843b-42d00208bcc2" />
<br>
<br>


<img width="1906" height="862" alt="Screenshot 2026-04-07 154555" src="https://github.com/user-attachments/assets/73ac4025-14e7-4b44-8408-532e5166d459" />
<br>
<br>

<img width="1799" height="347" alt="Screenshot 2026-04-07 155000" src="https://github.com/user-attachments/assets/472d3cc3-5327-41a1-aa9a-0e4acc74b581" />



## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **Machine Learning:** Scikit-learn
* **Data Processing:** Pandas, NumPy
* **Model Storage:** Joblib / Pickle

---

## 📁 Project Structure

```
LOAN/
│
├── app.py                # Main Streamlit application
├── model.pkl             # Trained ML model
├── scaler.pkl            # Data scaler
├── requirements.txt      # Dependencies
├── dataset.csv           # Dataset used
└── README.md             # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/SQUADRON-LEADER/LOAN.git
cd LOAN
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the app

```bash
streamlit run app.py
```

---

## 📊 Model Details

* Type: Classification Model
* Goal: Predict loan approval
* Input Features:

  * Income
  * Employment Status
  * Loan Amount
  * Credit-related features
* Output:

  * Approved / Not Approved

---

## 🚀 Future Improvements

* 🔹 Add credit score integration
* 🔹 Improve model accuracy with advanced algorithms
* 🔹 Add user authentication system
* 🔹 Store prediction history in database
* 🔹 Deploy with custom domain

---

## 👨‍💻 Author

**SQUADRON-LEADER**

* GitHub: [https://github.com/SQUADRON-LEADER](https://github.com/SQUADRON-LEADER)

---

## 📜 License

This project is for educational purposes. Feel free to use and modify.

---

## ⭐ Support

If you like this project:

* ⭐ Star the repo
* 🍴 Fork it
* 📢 Share it
