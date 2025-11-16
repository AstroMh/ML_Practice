# Linear Regression – California Housing (sklearn)

This project uses **Linear Regression** to predict **median house values** in California districts using the classic **California Housing** dataset from scikit-learn.

It’s an end-to-end example of:

- Loading a built-in regression dataset from `sklearn`
- Performing basic exploratory data analysis (EDA)
- Training a linear regression model
- Evaluating it with standard regression metrics
- Visualizing predictions, residuals, and feature coefficients
- Saving all results to files so they can be viewed directly on GitHub

---

## 🔧 What the script does

`california_housing_linear_regression.py`:

1. **Loads** the California Housing dataset via `sklearn.datasets.fetch_california_housing(as_frame=True)`
2. **Explores** the data by generating:
   - A pairplot of selected features and the target (`MedHouseVal`)
   - A correlation heatmap for all numeric columns
3. **Prepares features**:
   - `MedInc` – median income in the block group  
   - `HouseAge` – median house age  
   - `AveRooms` – average number of rooms per household  
   - `AveBedrms` – average number of bedrooms per household  
   - `Population` – block group population  
   - `AveOccup` – average number of household members  
   - `Latitude` – block group latitude  
   - `Longitude` – block group longitude  
4. **Uses the target**:
   - `MedHouseVal` – median house value (in units of \$100,000)
5. **Splits** the data into train and test sets
6. **Trains** a `LinearRegression` model on the training data
7. **Evaluates** the model on the test data using:
   - Mean Absolute Error (MAE)
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - R² score
8. **Generates plots**:
   - Predicted vs Actual median house values (with y = x reference line)
   - Residuals distribution (Actual − Predicted)
   - Bar plot of feature coefficients
9. **Saves results** into a `results/` directory:
   - Plots as `.png`
   - Metrics and coefficients as `.txt` and `.csv`

---

## 📚 Dataset description (`sklearn.datasets.fetch_california_housing`)

This project uses the **California Housing** dataset provided by scikit-learn, which is based on **1990 U.S. Census** data for California.

Key details:

- **Samples**: 20,640 districts (one sample per block group)  
- **Features (8)**:
  - `MedInc` – median income in the block group (in tens of thousands of dollars)  
  - `HouseAge` – median house age in years  
  - `AveRooms` – average number of rooms per household  
  - `AveBedrms` – average number of bedrooms per household  
  - `Population` – total population of the block group  
  - `AveOccup` – average number of occupants per household  
  - `Latitude` – geographical latitude of the block group  
  - `Longitude` – geographical longitude of the block group  
- **Target**:
  - `MedHouseVal` – median house value for households within a block group, in units of \$100,000  
- **Task type**: Regression – predict continuous house values  
- **Origin**: Derived from the **California housing data** originally made available by Pace & Barry, constructed from the 1990 census.  

The dataset is often used as a real-world, medium-sized regression benchmark and is a popular alternative to the older Boston Housing dataset.

---

## 📁 Outputs (in `results/`)

After running the script, you’ll get:

**Plots**

- `california_pairplot.png`  
  Pairplot of selected features (`MedInc`, `HouseAge`, `AveRooms`, `AveOccup`) plus the target (`MedHouseVal`).

- `california_corr_heatmap.png`  
  Correlation matrix heatmap showing relationships between all features and the target.

- `california_pred_vs_actual.png`  
  Scatter plot of predicted vs actual median house values, with a y = x reference line.

- `california_residuals.png`  
  Distribution of residuals (actual − predicted), useful for checking model assumptions.

- `california_coefficients.png`  
  Horizontal bar chart of linear regression coefficients for each feature.

**Text & Tables**

- `california_linear_regression_report.txt`  
  MAE, MSE, RMSE, and R² for the trained model.

- `california_coefficients.csv`  
  Table of feature names and their corresponding coefficients (CSV format).

- `california_coefficients.txt`  
  Human-readable list of feature coefficients, one per line.

These files can be viewed directly on GitHub so anyone can inspect the model’s performance and interpretation without running the code.
