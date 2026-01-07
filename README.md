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
