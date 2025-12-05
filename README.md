# Required Assignment 11.1 – What Drives the Price of a Car?

UC Berkeley Professional Certificate in Machine Learning & AI  
Practical Application – Used Car Price Modeling

## 1. Project Overview

A used car dealership wants to better understand **what drives the price of a used car** and use data to fine-tune its **inventory and pricing strategy**.

Using a Kaggle dataset of used vehicles, this project:

- Explores which features (age, mileage, manufacturer, body type, fuel, etc.) most influence price
- Builds regression models to **predict price**
- Interprets model coefficients to provide **business recommendations** for the dealership

This work follows the **CRISP-DM** process:

1. Business Understanding  
2. Data Understanding  
3. Data Preparation  
4. Modeling  
5. Evaluation  
6. Deployment / Recommendations

---

## 2. Data

- **Source:** Kaggle Used Cars dataset (426K rows from the original 3M+ used car records)
- **Target variable:** `price` (continuous, USD)
- **Key features used:**
  - Numerical: `year`, `odometer`
  - Categorical: `manufacturer`, `condition`, `cylinders`, `fuel`, `title_status`,
    `transmission`, `drive`, `size`, `type`, `paint_color`, `state`, etc.

Columns such as `id`, `VIN`, `region`, and `model` were excluded from modeling to avoid ID-like fields and very high cardinality features.

The raw dataset contains missing values and outliers, which are handled in the notebook.

---

## 3. Methods

### 3.1 Data Preparation

Main preparation steps (implemented in the notebook):

- Removed rows with missing `price`
- Filtered **unrealistic values**:
  - `price` kept between 500 and 150,000 USD  
  - `year` limited to [1980, 2022]  
  - `odometer` limited to [0, 1,000,000] miles
- Dropped rows missing key information: `year`, `odometer`, `manufacturer`, `fuel`, `state`
- Selected a subset of numeric and categorical features for modeling

A **scikit-learn `ColumnTransformer`** was used to:

- Standardize numeric features (`StandardScaler`)
- One-hot encode categorical features (`OneHotEncoder`, with `handle_unknown="ignore"`)

### 3.2 Models

Three regression models were trained:

1. **Linear Regression**  
   - Baseline model without regularization.

2. **Ridge Regression**  
   - Linear model with L2 regularization.  
   - Hyperparameter `alpha` tuned via **GridSearchCV** with 3-fold cross-validation.

3. **Lasso Regression**  
   - Linear model with L1 regularization, which can shrink coefficients to zero and act as feature selection.  
   - Due to the large dataset size and high computational cost of Lasso, a selected `alpha` value (0.001) was used directly instead of running a full GridSearchCV.  
   - Regularized modeling and hyperparameter tuning are still demonstrated through Ridge Regression, which includes full 3-fold cross-validation with GridSearchCV.

### 3.3 Evaluation

Models were evaluated using:

- **RMSE (Root Mean Squared Error)** – in USD
- **R² (Coefficient of Determination)** – proportion of variance in price explained

Each model was evaluated on both the **training set** and a held-out **test set** (80/20 split).

---

## 4. Key Findings

### 4.1 Model Performance (example narrative)

- **Linear Regression**  
  - Test RMSE ≈ 8472.1781 USD  
  - Test R² ≈ 0.6519

- **Ridge Regression (best alpha = 1.0)**  
  - Similar or slightly better RMSE and R² compared to the baseline  
  - Reduced overfitting due to regularization

- **Lasso Regression (best alpha = 0.01) subset**  
  - Comparable performance to Ridge  
  - Produced a **sparser model**, highlighting a smaller set of the most important features

Overall, regularized models (Ridge/Lasso) provide **more stable and interpretable** coefficients than plain Linear Regression.

### 4.2 What Drives Price?

From the coefficient analysis of the regularized models (especially Ridge/Lasso):

- **Age and mileage dominate**  
  - Newer `year` strongly increases predicted price  
  - Higher `odometer` (more miles) strongly decreases price

- **Manufacturer and body type matter**  
  - Certain manufacturers have consistently higher prices, even after controlling for age and mileage  
  - SUVs and trucks tend to sell at higher prices than small sedans and coupes

- **Condition, drive type, and fuel type have secondary impacts**  
  - Better condition ratings (e.g., _excellent_ vs _fair_) are associated with higher prices  
  - 4WD/AWD can add a premium in some segments  
  - Some fuel types may be associated with discounts or premiums depending on market

---

## 5. Business Recommendations

For a non-technical dealership audience:

1. **Inventory Strategy**
   - Prioritize **newer, low-mileage vehicles**, especially in popular SUV and truck segments.
   - Be cautious buying **older, high-mileage sedans**, since the model assigns them much lower value.

2. **Pricing Guidance**
   - Use the predicted price as a **reference range** when setting list prices, then adjust for:
     - Local market conditions
     - Options and trim levels not captured in the data
   - Price vehicles with better condition ratings and more desirable attributes (e.g., 4WD) at the higher end of the range.

3. **Reconditioning Decisions**
   - Because condition has a measurable effect on price, invest in **basic reconditioning** (cleaning, minor cosmetic fixes) for newer or mid-priced cars where a small investment could yield a meaningful price uplift.

---

## 6. Limitations & Future Work

- Dataset may contain noisy entries (incorrect prices, odometer readings, or mislabeled fields).
- Some important predictors (trim/package, accident history, service records, exact options) are not present.
- Next steps could include:
  - Trying more advanced regressors (e.g., tree-based models) if/when covered in the course
  - Segmenting by region or vehicle class
  - Including text features (listing descriptions) or images for richer modeling

---

## 7. Repository Structure

- `prompt_II.ipynb` – Main Jupyter notebook with all analysis and modeling  
- `vehicles.csv.zip` – Compressed input data (from Kaggle)  
- `README.md` – This project summary and documentation  

---

## 8. How to Run

1. Clone the repository or download the files.
2. Unzip the vehicles.csv.zip
3. Install Python dependencies (e.g., using `pip`):

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
