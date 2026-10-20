# Salary Prediction with an Artificial Neural Network (Regression)

A regression project that predicts an employee's salary from demographic and work-experience features, using a feed-forward Artificial Neural Network (ANN) built with Keras/TensorFlow and tuned with Optuna.

## Project Overview

| Item | Details |
|---|---|
| **Task** | Regression |
| **Target** | `Salary` |
| **Model** | ANN (2 hidden layers, ReLU output) |
| **Tuning** | Optuna (hyperparameter search) |
| **Loss** | Mean Absolute Error (MAE) |
| **Metric** | R² score |
| **Notebook** | `ANN_Regression_task_G30.ipynb` |

## Dataset

**File:** `Salary_Data.csv`, with 6,704 rows and 6 columns. After dropping rows with missing values, 6,698 rows remain.

| Column | Type | Description |
|---|---|---|
| `Age` | Numeric | Age of the employee (21–62) |
| `Gender` | Categorical | Male / Female / Other |
| `Education Level` | Categorical | High School, Bachelor's Degree, Master's Degree, PhD |
| `Job Title` | Categorical | 193 distinct job titles |
| `Years of Experience` | Numeric | Total years of experience (0–34) |
| `Salary` | Target | Annual salary (about 115,000 on average, up to 250,000) |

### Data cleaning

- **Missing values:** a few rows had missing values in several columns and were dropped.
- **Inconsistent education labels:** the raw data contained variants of the same level, which were merged:
  - `Bachelor's` → `Bachelor's Degree`
  - `Master's` → `Master's Degree`
  - `phD` → `PhD`

### Feature engineering

Two features were created from `Job Title`, which was then dropped:

| Feature | Description |
|---|---|
| `exp_mean_by_job` | Average years of experience for that job title |
| `edu_nunique_by_job` | Number of distinct education levels within that job title |

## Workflow

1. **Read libraries and data**
2. **Descriptive statistics** — `data.describe(include='all')`
3. **Check for and drop missing values**
4. **Merge inconsistent education categories**
5. **Create job-title-based features** and drop `Job Title`
6. **Define target and inputs, then split** — `train_test_split` (80% train / 20% test, `random_state=42`)
7. **Encode categorical variables** — `LabelEncoder` fitted on the training set and applied to the test set
8. **Scale the inputs** — `StandardScaler` fitted on the training set only
9. **Build the ANN and tune it with Optuna** — 2 hidden layers (ReLU) and a ReLU output layer, trained with MAE loss. Optuna ran 30 trials, maximizing test R², over:
   - units in each hidden layer (16–128)
   - optimizer (Adam, RMSprop)
   - learning rate (0.01–1.0, log scale)
   - number of epochs (50–200)
   - batch size (16–64)
10. **Train and evaluate the final model** with the best hyperparameters and report R² on the train and test sets.

## Results

**Final model (retrained with the best Optuna hyperparameters):**

| Dataset | R² |
|---|---|
| Train | 0.944 |
| Test | 0.930 |

The model explains about 93% of the variance in salary on unseen data. The gap between train and test is small (about 1.4 points), so there is only mild overfitting.

**Best hyperparameters found by Optuna (30 trials, best trial R² 0.944):**

| Hyperparameter | Value |
|---|---|
| Units in hidden layer 1 | 87 |
| Units in hidden layer 2 | 78 |
| Optimizer | Adam |
| Learning rate | 0.309 |
| Epochs | 166 |
| Batch size | 39 |

The R² of the best trial during the search and the final retrained model differ slightly because the network starts from new random weights each time it is trained.

### Notes on tuning

The salary values are large (around 100,000), so with a small learning rate (1e-5 to 1e-2), few epochs and small layers, the network got stuck predicting 0 and gave a negative R². Widening the Optuna ranges fixed this: larger learning rates, more units and more epochs, with the optimizer list restricted to Adam and RMSprop. Almost all of the 30 trials then reached R² between 0.83 and 0.94, and the best ones used Adam with a large learning rate (0.1 to 0.3).

## Technologies

- Python 3
- pandas, NumPy
- scikit-learn
- TensorFlow / Keras
- Optuna
- Jupyter Notebook

## How to Run

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install pandas numpy scikit-learn tensorflow optuna jupyter
jupyter notebook ANN_Regression_task_G30.ipynb
```

Keep `Salary_Data.csv` in the same folder as the notebook.

## Repository Structure

```
├── ANN_Regression_task_G30.ipynb
├── Salary_Data.csv
└── README.md
```

## Key Takeaways

- Cleaning inconsistent category labels before encoding prevents the same level from being treated as different categories.
- Categorical variables are label-encoded and inputs are standardized, with both fitted on the training data only to avoid data leakage.
- Hyperparameter ranges must suit the scale of the target: with a target in the 100,000s, a wider learning-rate range was essential.
- Optuna automates the search for the best network size, optimizer, learning rate, epochs and batch size.
