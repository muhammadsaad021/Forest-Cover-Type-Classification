# 🌲 Forest Cover Type Classification

This project predicts the **forest cover type** (the predominant kind of tree cover) from cartographic variables using **machine learning classification models**.  
I implemented and compared **XGBoost** and **Random Forest** models, with results visualized using confusion matrices.  

The dataset used is the [UCI Forest Cover Type Dataset](https://archive.ics.uci.edu/dataset/31/covertype), a popular benchmark dataset in environmental data science.

---

## 📊 Features Used

The dataset contains **54 features**, including:

- Elevation
- Aspect
- Slope
- Horizontal & Vertical Distances
- Hillshade indices (9am, Noon, 3pm)
- Wilderness Area indicators
- Soil Type indicators

---

## 🧪 Models & Methods

- **Random Forest Classifier** (`sklearn.ensemble.RandomForestClassifier`)
- **XGBoost Classifier** (`xgboost.XGBClassifier`)
- **Classification Report** and **Confusion Matrix** to evaluate performance
- **Matplotlib** & **Seaborn** for data visualization and results comparison

---

## 📈 Results

- **Random Forest** achieved higher accuracy compared to **XGBoost**
- Visualized **confusion matrices** highlight model performance across multiple forest cover classes
- Example performance metrics:
  - Random Forest Accuracy: `~95.5%`
  - XGBoost Accuracy: `~87.1%`

---

## 🚀 How to Use

1. **Clone or download this repository**  
2. Install dependencies:  
   ```bash
   pip install pandas numpy scikit-learn seaborn matplotlib xgboost
   ```
1. **Clone or download this repository**.
2. **Open the provided Google Colab notebook**.
3. **Upload the dataset**:
   - Download the dataset file `covtype.data` (attached in this repo).
   - In Colab, upload it.
4. The notebook will handle data preprocessing, model training, and evaluation.


---

## 📁 Dataset

- **Source:** [UCI Machine Learning Repository – Forest Cover Type Dataset](https://archive.ics.uci.edu/dataset/31/covertype)
- **Classes:** 7 forest cover types
- **Samples:** 581,012
- **Features:** 54 cartographic variables

The dataset file `covtype.data` is included in this repository.
---

## 🧠 Motivation

Predicting forest cover type has applications in:

- Environmental conservation
- Land management
- Forestry planning
- Ecological research

---

## 📌 Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib
- xgboost

---

## 📜 License

This project is for educational and research purposes only. Dataset license details can be found at the source.

---

## 👤 Author

Muhammad Saad  
GitHub: [@muhammadsaad021](https://github.com/muhammadsaad021)
