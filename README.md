# Linear Regression - Height-Weight Prediction

A comprehensive implementation of Linear Regression for predicting height based on weight using Python and scikit-learn.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project demonstrates the implementation of Simple Linear Regression to predict a person's height based on their weight. The project includes data exploration, visualization, model training, evaluation, and comparison with OLS (Ordinary Least Squares) method.

## 📊 Dataset

- **File**: `height-weight.csv`
- **Features**: 
  - `Weight` (Independent variable)
  - `Height` (Dependent variable)
- **Size**: 23 observations
- **Correlation**: 0.931 (Strong positive correlation)

## ✨ Features

- Data loading and exploration
- Correlation analysis
- Data visualization using:
  - Scatter plots
  - Pair plots (Seaborn)
  - Regression line plots
- Data preprocessing:
  - Train-test split (75%-25%)
  - Standard Scaling (Z-score normalization)
- Model implementation using:
  - Scikit-learn's LinearRegression
  - Statsmodels' OLS
- Performance metrics evaluation:
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R² Score
  - Adjusted R² Score

## 🔧 Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/linear-regression-height-weight.git

# Navigate to the project directory
cd linear-regression-height-weight

# Install required packages
pip install -r requirements.txt
```

### Required Libraries
```
pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
jupyter
```

## 💻 Usage

### Running the Jupyter Notebook
```bash
# Start Jupyter Notebook
jupyter notebook

# Open Linear_reg.ipynb
```

### Code Example
```python
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv('height-weight.csv')

# Prepare features and target
X = df[['Weight']]
y = df['Height']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
regression = LinearRegression(n_jobs=-1)
regression.fit(X_train_scaled, y_train)

# Make predictions
y_pred = regression.predict(X_test_scaled)

# Predict for new data
new_weight = [[72]]
predicted_height = regression.predict(scaler.transform(new_weight))
print(f"Predicted Height: {predicted_height[0]:.2f}")
```

## 📈 Model Performance

### Linear Regression Results
- **Coefficient (Slope)**: 1.048
- **Intercept**: 80.527
- **Mean Squared Error**: 7276.93
- **Mean Absolute Error**: 82.98
- **Root Mean Squared Error**: 85.30
- **R² Score**: -15.72

### OLS Regression Results
- **Coefficient**: 2.104
- **R-squared**: 0.986 (uncentered)
- **F-statistic**: 1133
- **P-value**: 2.80e-16

## 🛠 Technologies Used

- **Python 3.12.5**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical data visualization
- **Scikit-learn**: Machine learning implementation
- **Statsmodels**: Statistical modeling and OLS regression

## 📊 Results

### Visualization Examples

The project includes several visualizations:

1. **Scatter Plot**: Shows the relationship between weight and height
2. **Correlation Heatmap**: Displays the correlation matrix
3. **Pair Plot**: Shows distributions and relationships
4. **Regression Line**: Visual representation of the fitted model

### Model Equation
```
Height = 80.527 + 1.048 × Weight
```

### Sample Predictions

| Weight | Predicted Height |
|--------|------------------|
| 48     | ~130 cm         |
| 60     | ~143 cm         |
| 72     | ~156 cm         |

## 📁 Project Structure
```
linear-regression-height-weight/
│
├── Linear_reg.ipynb          # Main Jupyter notebook
├── height-weight.csv         # Dataset
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
└── .gitignore               # Git ignore file
```

## 🚀 Getting Started

1. **Clone the repository**
```bash
   git clone https://github.com/yourusername/linear-regression-height-weight.git
```

2. **Install dependencies**
```bash
   pip install -r requirements.txt
```

3. **Run the notebook**
```bash
   jupyter notebook Linear_reg.ipynb
```

## 📝 Key Learnings

- Implementation of simple linear regression
- Data preprocessing and standardization
- Model evaluation metrics
- Comparison between scikit-learn and statsmodels
- Visualization of regression results

## ⚠️ Notes

- The dataset is small (23 observations), which may affect model performance
- Standard scaling is applied to improve model convergence
- The negative R² on test set suggests potential overfitting or data quality issues
- For production use, consider additional validation techniques

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/Praveen23-kk)
- LinkedIn: [Your LinkedIn](https://www.linkedin.com/in/k-praveen-kumar-6223aa280/)
- Email: Praveennaaz23@gamil.com

## 🙏 Acknowledgments

- Inspired by Krish Naik's Machine Learning tutorials
- Dataset source: [Specify if applicable]
- Thanks to the scikit-learn and statsmodels communities

## 📚 Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Statsmodels Documentation](https://www.statsmodels.org/)
- [Linear Regression Theory](https://en.wikipedia.org/wiki/Linear_regression)

---

**Note**: This is a learning project demonstrating basic linear regression concepts. For production use, consider additional validation, cross-validation, and feature engineering techniques.

⭐ Star this repo if you find it helpful!
