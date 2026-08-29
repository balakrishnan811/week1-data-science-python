# Week 1 – Data Science with Python

## Virtual Data Science with Python Apprentice Intern

### Project: Data Acquisition, Cleaning and Exploratory Data Analysis

This project demonstrates the basic data science workflow using the publicly available Titanic passenger dataset.

## Objectives

- Acquire a publicly available dataset.
- Inspect the dataset using Python.
- Identify and handle missing values.
- Detect and remove duplicate records.
- Check and correct data types.
- Perform exploratory data analysis (EDA).
- Create visualizations and identify useful patterns.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn

## Dataset

The analysis uses the Titanic dataset available through the Seaborn public dataset collection. The dataset contains passenger information such as survival status, passenger class, gender, age, family information, fare and embarkation port.

The Python script loads the dataset with:

```python
sns.load_dataset("titanic")
```

## Data Cleaning

The project includes:

1. Checking the dataset structure and summary statistics.
2. Checking missing values.
3. Filling missing numerical values such as age using the median.
4. Filling missing categorical values using the mode.
5. Removing duplicate records.
6. Checking data types after preprocessing.

## Exploratory Data Analysis

The project creates the following visualizations:

1. Missing values before cleaning
2. Passenger age distribution
3. Survival status
4. Survival by gender
5. Correlation heatmap

## Key Insights

- Survival was not equally distributed between passengers.
- Gender showed a noticeable relationship with survival.
- Passenger class can be investigated as an important factor affecting survival.
- Age has a broad distribution and can be further analysed by survival status.
- Correlation analysis helps identify numerical variables that may be useful for future modelling.

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python week1_titanic_analysis.py
```

The generated graphs will be saved in the `visualizations` folder.

## Internship Deliverable

This repository supports the Week 1 internship report submitted for the Virtual Data Science with Python Apprentice Intern program.
