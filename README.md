# Financial Analysis & Investment Prediction

## Summary
This project focuses on financial data analysis and predicting investment preferences using machine learning. The analysis covers key findings from the dataset, including missing values, correlations, and investment trends. A machine learning model is developed to predict an individual's preferred investment avenue.

## 1. Data Analysis Key Findings
### Data Shape
- The dataset consists of **24 columns**.
- The exact number of rows was determined during data exploration using `df.shape`.

### Missing Values
- The analysis identified missing values in some columns.
- The specific count of missing values was determined using `df.isnull().sum()`, but not all details are shown in the summary due to output truncation.

### Correlations
- A **correlation matrix and heatmap** were generated for numerical features.
- Non-numeric columns such as **'Stock_Marktet'** and **'Factor'** were excluded.
- This analysis helps in identifying relationships between investment-related numerical variables.

### Investment Trends by Gender
- The analysis revealed differences in investment preferences based on gender.
- The average investment in **Mutual Funds** by gender was calculated using:
  ```python
  df.groupby('gender')['Mutual_Funds'].mean()
  ```

### Investment Duration by Avenue
- The distribution of investment duration across different investment avenues was analyzed.
- The following code was used to normalize and visualize the distribution:
  ```python
  df.groupby('Avenue')['Duration'].value_counts(normalize=True).unstack()
  ```

## 2. Insights & Next Steps
### Address Data Quality Issues
- Investigate and handle missing values in the dataset.
- Convert non-numeric categorical columns (**'Stock_Marktet'**, **'Factor'**) to numerical format for better analysis.
- Improve feature selection to enhance predictive modeling.

### Deep Dive into Key Relationships
- Analyze relationships between investment choices and demographics (**age, gender, income levels**).
- Visualizations such as **box plots** or **scatter plots** can be used to explore investment preferences effectively.
- Examine **investment objectives** to understand key drivers behind different investment decisions.

## 3. Machine Learning Model Summary
- A **Random Forest Classifier** was used to predict investment preferences.
- **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to balance the dataset.
- **Hyperparameter tuning** was performed using **GridSearchCV** to improve model performance.
- Model accuracy and classification reports were generated to evaluate results.

## 4. How to Run
### Prerequisites
- Python environment with the following libraries installed:
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
  ```
- Jupyter Notebook or Google Colab for execution.

### Execution Steps
1. Upload the `Finance_data.csv` file.
2. Run the Python script for data exploration and analysis.
3. Train and evaluate the model using Random Forest.
4. Interpret model results and optimize further.

---


