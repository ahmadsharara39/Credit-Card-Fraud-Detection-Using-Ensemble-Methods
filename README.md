# 💳 Credit Card Fraud Detection using Ensemble Methods

This project builds a robust machine learning pipeline for detecting **credit card fraud** using **ensemble learning** techniques. It handles class imbalance using **SMOTE**, explores multiple models, and combines them using a **Stacking Classifier** to improve performance.

---

## 📥 Getting Started

Clone the repository:

```bash
git clone https://github.com/ahmadsharara39/Credit-Card-Fraud-Detection-Using-Ensemble-Methods
cd Credit-Card-Fraud-Detection-Using-Ensemble-Methods
```

---

## 📄 Dataset

The dataset used is `creditcard.csv`, which includes:

- 284,807 transactions
- 492 fraud cases (Class = 1)
- 30 numerical input features (V1–V28 + Amount + Time)

---

## 🔍 Workflow Summary

1. **Data Cleaning**  
   - Handle missing values and duplicates
   - Drop unused or skewed features (e.g., `Time`)

2. **Exploratory Data Analysis (EDA)**  
   - Class distribution visualization  
   - Amount distribution and boxplots

3. **Preprocessing**
   - Apply `RobustScaler` to `Amount`
   - Use `SMOTE` to balance classes in the training set

4. **Modeling**
   - Train four classifiers:  
     ✅ Random Forest  
     ✅ Logistic Regression  
     ✅ Naive Bayes  
     ✅ MLP (Neural Network)

5. **Stacking Ensemble**
   - Combine base models using `StackingClassifier`
   - Meta-learner: Logistic Regression

6. **Evaluation Metrics**
   - Accuracy  
   - Precision  
   - Recall  
   - F1 Score  
   - Confusion Matrices  
   - Bar graphs comparing performance **before and after stacking**

---

## 📦 Requirements

```bash
pip install pandas matplotlib seaborn scikit-learn imbalanced-learn
```

---

## 📁 File Structure

```
📦 Credit-Card-Fraud-Detection-Using-Ensemble-Methods/
 ┣ 📄 creditcard.csv          ← Dataset
 ┣ 📄 ML_Project_2_final.ipynb ← Main analysis notebook
 ┗ 📄 README.md               ← Project documentation (this file)
```

---

## 📊 Sample Results

📌 Example (varies depending on random seed and model):

```
Random Forest:      Accuracy: 0.9993, Recall: 0.8061
Logistic Regression:Accuracy: 0.9766, Recall: 0.6735
Naive Bayes:        Accuracy: 0.9703, Recall: 0.7755
MLP:                Accuracy: 0.9986, Recall: 0.7755
Stacking:           Accuracy: 0.9995, Recall: 0.8571
```

✅ Stacking improved recall and F1-score across the board.

---

## 🧠 Libraries Used

- `pandas`, `numpy` – Data processing
- `seaborn`, `matplotlib` – Visualization
- `scikit-learn` – Models & evaluation
- `imbalanced-learn` – SMOTE for balancing
- `sklearn.ensemble.StackingClassifier` – Ensemble learning

---

## ✍️ Author

**Ahmad Sharara**  
GitHub: [@ahmadsharara39](https://github.com/ahmadsharara39)

---

## 📄 License

This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/).

