# used-car-analysis
A project to predict the price of used cars using Python and Machine Learning.

This project is a machine learning model that predicts the price of used cars. It uses features like the car's year, mileage, and brand to estimate the final price. The goal is to find the best linear model that minimizes the prediction error.

### Dataset
The data for this project is from the [Used Car Price Prediction Dataset](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset) on Kaggle.

### Process
I followed the complete machine learning workflow to build this model:

* **Data Cleaning:** Loaded the raw data from the Kaggle dataset. Cleaned columns like `price` and `milage` to convert them from text to numbers. Filled missing (NaN) values in the data using the mode (the most frequent value).

* **Feature Engineering:** The model needs numbers, not text. I used One-Hot Encoding to convert categorical columns (like `brand` and `fuel_type`) into a numerical format (columns of 0s and 1s).

* **Modeling:** I used the Scikit-learn library to build a `LinearRegression` model. The data was split into a training set (80%) and a testing set (20%) to ensure a fair evaluation and prevent overfitting. The model was trained on the training data using the `.fit()` command.

* **Evaluation and Refinement:** The model's performance was evaluated on the unseen test data using the RMSE (Root Mean Squared Error) metric. I created a scatter plot to visualize the model's predictions vs. the real prices. After analyzing the results, I identified that a few very expensive cars (outliers) were making the error high. I ran a second experiment without these outliers, which created a better model for most cars.

### How to Use
To run this project, follow these steps:
1.  Download the files from this repository.
2.  Make sure you have Python installed with the following libraries: `pandas`, `scikit-learn`, `matplotlib`, `seaborn`.
3.  Open the `.ipynb` file in a Jupyter environment (like VS Code or Jupyter Notebook).
4.  Run the cells from top to bottom.

### Key Learnings
* This project helped me understand how a regression model works, including the importance of a cost function and Gradient Descent.
* I learned that the model is very sensitive to outliers, and removing them can greatly improve performance for the majority of the data.
* I understand the strengths and limits of this model. For the future, a more complex model like Random Forest could be used to get even better results.
