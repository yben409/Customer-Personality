# 🧠 Clustering KMeans: Customer Personality Analysis

This project focuses on clustering customer profiles using **KMeans** to uncover meaningful segments based on personality traits and consumption behavior. It includes:

- Data cleaning  
- Exploratory Data Analysis (EDA)  
- Dimensionality reduction using PCA  
- Clustering with KMeans  
- Insightful visualizations

---

## 📁 Project Structure

- `fichier.ipynb`: Jupyter notebook with the complete workflow  
- `train.csv`: Dataset for training  
- `test.csv`: Dataset for testing or validation  
- `sample_submission.csv`: Sample format for prediction output  
- `steel-plate-defect-prediction-with-roc-auc.ipynb`: *(Unrelated project)* ML pipeline for defect classification

---

## 🔧 Technologies Used

- Python 3.x  
- Pandas, NumPy – data manipulation  
- Matplotlib, Seaborn – visualization  
- Scikit-learn – KMeans, PCA, StandardScaler

---

## 📊 Data Cleaning

- Inspecting missing values using heatmaps:
```python
sns.heatmap(df.isnull().transpose(), cmap="rocket_r")
```
---

## 🔍 Exploratory Data Analysis (EDA)

- Plotted histograms and correlation heatmaps
- Analyzed feature relationships and clusters
  
---

## ⚙️ Data Preprocessing

- Scaled data using StandardScaler
- Reduced dimensions with PCA:
```python
from sklearn.decomposition import PCA
X_pca = PCA(n_components=2).fit_transform(X_scaled)
```

---

## 📌 Clustering with KMeans

- Used the elbow method to find optimal k:
```python
distortions = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i).fit(X_scaled)
    distortions.append(kmeans.inertia_)
```
- Trained final model and assigned clusters:
```python
kmeans = KMeans(n_clusters=optimal_k)
labels = kmeans.fit_predict(X_scaled)
```

---

## 📈 Visualizations

- Displayed PCA-reduced clusters:
```python
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
Visualized feature trends by cluster
```
- Additional insights shown using Seaborn & Matplotlib
  
---

## 📬 Contact :

- Youssef Benaouda, Freelancer in AI
- Mail : benaoudayoussef123@gmail.com
- Active on LinkedIn
