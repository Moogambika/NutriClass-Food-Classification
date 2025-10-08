# 🍎 NutriClass: Food Classification Using Nutritional Data

**NutriClass** is a machine learning project that classifies food items based on their nutritional composition.
By analyzing key nutrients such as calories, protein, fat, carbohydrates, sugar, and fiber, the model predicts the **food name** and provides insights into its nutritional importance.

---

## 📘 Project Overview

The goal of this project is to build a **supervised classification model** that learns from nutritional data and predicts the corresponding **food item**.
This project also explores the importance of various nutrients in food classification and evaluates multiple machine learning models for best performance.

---

## 🧠 Key Objectives

* Preprocess and analyze a **synthetic food dataset**
* Handle **imbalanced data** using under-sampling
* Train and evaluate multiple ML models:
  Logistic Regression, Decision Tree, Random Forest, KNN, SVM, XGBoost and Gradient Boosting
* Perform **hyperparameter tuning** using `GridSearchCV`
* Visualize **feature importance** for better interpretability
* Make real-time predictions using user-input nutritional values

---

## 📊 Dataset

* **Name:** `synthetic_food_dataset_imbalanced.csv`
* **Target Variable:** `Food_Name`
* **Feature Variables:**
  `Calories`, `Protein`, `Fat`, `Carbs`, `Sugar`, `Fiber`, `Sodium`, `Cholesterol`, `Glycemic_Index`, `Water_Content`, `Serving_Size`

---

## ⚙️ Technologies Used

| Category      | Tools                                                     |
| ------------- | --------------------------------------------------------- |
| Programming   | Python                                                    |
| Libraries     | pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost |
| Environment   | Google Colab                                              |
| Visualization | Matplotlib, Seaborn                                       |

---

## 🧩 Machine Learning Workflow

1. **Data Loading and Exploration**

   * Load dataset and understand structure and balance

2. **Data Preprocessing**

   * Handle missing values
   * Encode target (`Food_Name`) using `LabelEncoder`
   * Scale numerical features with `StandardScaler`
   * Handle class imbalance via **Random Under-Sampling**

3. **Model Building and Evaluation**

   * Train multiple models and compare their performance
   * Use `f1_macro` and accuracy metrics
   * Perform **cross-validation**

4. **Hyperparameter Tuning**

   * Use `GridSearchCV` for RandomForest to find optimal parameters

5. **Feature Importance Analysis**

   * Plot feature importances using RandomForest and XGBoost

6. **Prediction on New Data**

   * Accept user input (nutritional values) and predict corresponding food name

---

## 🧮 Evaluation Metrics

* **Accuracy**
* **Precision, Recall, F1-Score (Macro)**
* **Cross-Validation Mean F1 Score**

---

## 🚀 Results Summary

| Model               | Accuracy  | Macro F1  | Remarks               |
| ------------------- | --------- | --------- | --------------------- |
| XGBoost             | ⭐ Highest | ⭐ Highest | Best performing model |
| Gradient Boosting   | High      | High      | Good generalization   |
| Random Forest       | Moderate  | Moderate  | Balanced trade-off    |
| Logistic Regression | Moderate  | Moderate  | Linear model baseline |

---

## 🌿 Example Prediction

```python
# Example input (nutrition values)
sample_features = np.array([[266, 11, 10, 33, 3.6, 2.3, 640, 22, 75, 45, 100]])
sample_scaled = scaler.transform(sample_features)
predicted_class = xgb_clf.predict(sample_scaled)[0]
predicted_food = le.inverse_transform([predicted_class])[0]
print("Predicted Food Name:", predicted_food)
```

**Output:**

```
Predicted Food Name: Pizza
```

---

## 📈 Visualizations

* Nutrient Distribution per Food Category
* Correlation Heatmap
* Model Accuracy & F1 Comparison Bar Graph
* Feature Importance (RandomForest, XGBoost)

---

## 🧩 Future Enhancements

* Add a **Health Classification System** (Healthy / Unhealthy)
* Integrate **Streamlit or Flask Web App** for real-time predictions
* Extend dataset with more diverse food items
* Use **Deep Learning (ANN)** for advanced pattern detection

---

## 👩‍💻 Author

**Moogambika Govindaraj**
💡 AI & ML Enthusiast | Data Science Learner

---

## 🏫 Acknowledgments

This project was completed as part of an academic exploration in Machine Learning.
I would like to express my sincere gratitude to **GUVI** and to the **external and internal professors** for their valuable guidance, support and encouragement throughout this project.
Their insights and teaching made it easier to understand complex topics like model evaluation, data preprocessing and feature importance and helped me successfully complete this project with confidence.

---

### 📜 License

This project is open-sourced under the MIT License.
