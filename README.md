# 💳 Credit-Spec: End-to-End Credit Score Prediction System

### **🚀 Project Overview**
**Credit-Spec** is a professional-grade Machine Learning application designed to predict an individual's creditworthiness. Built with a robust **Scikit-Learn** backend and an interactive **Streamlit** frontend, this tool transforms complex financial data into actionable credit insights (**High, Average, Low**).

---

### **🧠 Technical Core & Logic**
This system isn't just a simple script; it's a complete **ML Pipeline**:
* **Data Persistence:** Uses serialized `.pkl` files for the **Trained Model**, **StandardScaler**, and **Feature Metadata**.
* **Preprocessing Engine:** Automated handling of categorical encoding and numerical scaling to ensure inference matches training logic perfectly.
* **Multi-Class Classification:** Specifically tuned to categorize risk into three distinct tiers.



---

### **🛠️ The Tech Stack**
* **Frontend:** `Streamlit` (Interactive UI)
* **ML Framework:** `Scikit-Learn` (Random Forest / Logistic Regression)
* **Data Handling:** `Pandas` & `NumPy`
* **Serialization:** `Pickle`

---

### **📊 Input Features Evaluated**
The model makes decisions based on the following weighted parameters:
* **Financials:** Annual Income, Home Ownership.
* **Demographics:** Age, Gender, Education, Marital Status.
* **Stability:** Number of Children (Dependents).

---

### **🚦 Prediction Logic (Output Mapping)**
| Class ID | Credit Category | Risk Level |
| :--- | :--- | :--- |
| **1** | **High** | Low Financial Risk |
| **0** | **Average** | Moderate Risk |
| **2** | **Low** | High Financial Risk |



---

### **🚀 How to Run Locally**
1. **Clone the repo:** `git clone <your-repo-url>`
2. **Install requirements:** `pip install -r requirements.txt`
3. **Launch App:** `streamlit run app.py`
