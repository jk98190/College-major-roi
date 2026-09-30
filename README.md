# Import the Libraries

```python
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from statsmodels.stats.multitest import multipletests
```

# Loading the datasets and Basic analysis

```python
df1 = pd.read_csv("ai_job_impact_cleaned.csv")
df2 = pd.read_csv("college_major_roi_cleaned.csv")
df3 = pd.read_csv("data_dictionary_cleaned.csv")
 
# The analyses below use `a` (AI jobs) and `c` (college). Derived columns are
# created HERE, so every section works once this block has been run.
a = df1.copy()
c = df2.copy()
 
c["debt_ratio"] = c.debt_usd / c.net_cost_usd
c["log_added"] = np.log(c.added_earnings_10yr_usd)
a["sal_chg_pct"] = (a.Salary_After_AI / a.Salary_Before_AI - 1) * 100
a["exp2"] = a.Years_Experience ** 2                       # used in A6
c["for_profit"] = (c.institution_tier == "for_profit").map({True: "for_profit", False: "other"})  # used in C5
```

```python
#Basic analysis using function df. describe 
print("===Basic stats for Dataset 1(df1)")
print(df1.describe())
```

**Output:**

```text
===Basic stats for Dataset 1(df1)
               Age  Years_Experience  Salary_Before_AI  Salary_After_AI  \
count  2000.000000       2000.000000       2000.000000      2000.000000   
mean     40.558000         16.663500      73942.072500     78428.642500   
std      10.786418         10.746675      26055.823793     29351.599013   
min      22.000000          0.000000      30036.000000     24447.000000   
25%      32.000000          8.000000      51665.500000     54086.750000   
50%      40.000000         16.000000      74620.000000     76820.500000   
75%      50.000000         26.000000      95418.250000    100730.250000   
max      59.000000         37.000000     119976.000000    161745.000000   

       Work_Hours_Per_Week  Job_Satisfaction  Productivity_Change_Pct  
count           2000.00000       2000.000000              2000.000000  
mean              44.85100          6.020500                 9.785820  
std                5.71254          2.006263                17.187882  
min               35.00000          3.000000               -19.990000  
25%               40.00000          4.000000                -5.355000  
50%               45.00000          6.000000                 9.850000  
75%               50.00000          8.000000                24.582500  
max               54.00000          9.000000                39.990000  
```

```python
#Basic analysis using function df. describe 
print("===Basic stats for Dataset 2(df2)")
print(df2.describe())
```

**Output:**

```text
===Basic stats for Dataset 2(df2)
       institution_selectivity_pctile           gpa  had_internship  \
count                    30000.000000  30000.000000    30000.000000   
mean                        55.072133      3.093337        0.436067   
std                         18.559406      0.439060        0.495904   
min                         16.000000      1.500000        0.000000   
25%                         42.000000      2.790000        0.000000   
50%                         55.000000      3.100000        0.000000   
75%                         68.000000      3.400000        1.000000   
max                        100.000000      4.000000        1.000000   

       completed_on_time   net_cost_usd       debt_usd  hs_baseline_10yr_usd  \
count       30000.000000   30000.000000   30000.000000          30000.000000   
mean            0.682367   88621.496667   38498.063333         359893.416667   
std             0.465564   33576.903160   37801.748200          39923.943255   
min             0.000000   28500.000000       0.000000         250000.000000   
25%             0.000000   65500.000000       0.000000         333100.000000   
50%             1.000000   81000.000000   38700.000000         359700.000000   
75%             1.000000  103600.000000   62300.000000         387300.000000   
max             1.000000  352900.000000  331700.000000         480000.000000   

       added_earnings_10yr_usd  earnings_10yr_usd   net_roi_usd       roi_pct  \
count             3.000000e+04       3.000000e+04  3.000000e+04  30000.000000   
mean              3.554752e+05       7.153680e+05  2.668537e+05    346.153987   
std               2.948109e+05       2.977847e+05  2.949958e+05    398.666879   
min               1.000000e+04       2.723000e+05 -3.052000e+05    -92.100000   
25%               1.179000e+05       4.835000e+05  3.280000e+04     39.700000   
50%               2.836500e+05       6.458000e+05  1.981000e+05    230.900000   
75%               5.093000e+05       8.717000e+05  4.235250e+05    537.400000   
max               2.894900e+06       3.245500e+06  2.755100e+06   3618.900000   

       positive_roi     high_roi  
count  30000.000000  30000.00000  
mean       0.835167      0.25000  
std        0.371036      0.43302  
min        0.000000      0.00000  
25%        1.000000      0.00000  
50%        1.000000      0.00000  
75%        1.000000      0.25000  
max        1.000000      1.00000  
```

```python
#Basic analysis using function df. describe 
print("===Basic stats for Dataset 3(df3)")
print(df3.describe())
```

**Output:**

```text
===Basic stats for Dataset 3(df3)
         column dtype                  description
count        18    18                           18
unique       18     3                           18
top     grad_id   int  Unique graduate identifier.
freq          1    11                            1
```

# AI Adoption level and salary change by industry

```python
#Salary change by industry using groupby, sal_chg_pct.median & unstack
df1["sal_chg_pct"] = (df1.Salary_After_AI / df1.Salary_Before_AI - 1) * 100
df1.groupby(["Industry", "AI_Adoption_Level"]).sal_chg_pct.median().unstack()
```

**Output:**

```text
AI_Adoption_Level       High       Low    Medium
Industry                                        
Education           9.223230  2.114308  5.442603
Finance             7.466105  2.652021  3.335113
Healthcare         13.304642  3.404959  4.539073
IT                 13.553013  2.371708  4.475848
Manufacturing       6.042324  2.272512  2.875174
Marketing          12.699283  3.349335  3.645704
Retail              9.525975  1.566035  5.320235
```

# C1. Risk profile by major: percentiles + share negative

```python
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order)

ax.axvline(0, color="red", lw=0.8)

ax.set_xlabel("Net ROI (USD): P10 - median - P90")
ax.set_title("ROI Range by Major")

plt.tight_layout()

# Create the output folder if it doesn't exist
import os
os.makedirs("figs", exist_ok=True)

# Save the chart
plt.savefig("figs/c1_roi_range.png", dpi=120)

plt.close()

print("C1 analysis completed successfully.")
print("Chart saved to: figs/c1_roi_range.png")
```

**Output:**

```text
C1 analysis completed successfully.
Chart saved to: figs/c1_roi_range.png
```

# Quantile regression: internship / on-time / GPA effect at P10/50/90

```python
# C4. Internship effect via propensity-score matching (ATT)
# -------------------------------------------------------------------

print("\nC4. Internship effect on 10-year earnings: matched vs naive")

# ------------------------------------------------------------
# 1. Prepare variables for propensity-score model
# ------------------------------------------------------------

X = pd.get_dummies(
    c[["major", "institution_tier", "region"]],
    drop_first=True
).astype(float)

X["gpa"] = c["gpa"].values

y = c["had_internship"].astype(int)


# ------------------------------------------------------------
# 2. Standardize predictors
# ------------------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ------------------------------------------------------------
# 3. Estimate propensity scores
# ------------------------------------------------------------

ps_model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

ps_model.fit(X_scaled, y)

ps = ps_model.predict_proba(X_scaled)[:, 1]

# Prevent propensity scores of exactly 0 or 1
ps = np.clip(ps, 0.001, 0.999)


# ------------------------------------------------------------
# 4. Convert propensity scores to log-odds
# ------------------------------------------------------------

logit = np.log(ps / (1 - ps)).reshape(-1, 1)


# ------------------------------------------------------------
# 5. Function to calculate ATT
# ------------------------------------------------------------

def att(idx):

    d = c.iloc[idx]
    l = logit[idx]

    treated = d["had_internship"].values == 1
    untreated = d["had_internship"].values == 0

    # Check that both groups exist
    if treated.sum() == 0 or untreated.sum() == 0:
        return np.nan

    # Nearest-neighbor matching
    nn = NearestNeighbors(n_neighbors=1)

    nn.fit(l[untreated])

    matched_indices = nn.kneighbors(
        l[treated],
        return_distance=False
    ).ravel()

    treated_outcome = (
        d["earnings_10yr_usd"].values[treated].mean()
    )

    matched_control_outcome = (
        d["earnings_10yr_usd"].values[untreated][matched_indices].mean()
    )

    return treated_outcome - matched_control_outcome


# ------------------------------------------------------------
# 6. Calculate matched ATT on full dataset
# ------------------------------------------------------------

full_idx = np.arange(len(c))

point = att(full_idx)


# ------------------------------------------------------------
# 7. Bootstrap confidence interval
# ------------------------------------------------------------

rng = np.random.default_rng(42)

boots = []

for _ in range(100):

    # Bootstrap sample of the same size as the original dataset
    idx = rng.integers(
        0,
        len(c),
        size=len(c)
    )

    result = att(idx)

    if not np.isnan(result):
        boots.append(result)


# ------------------------------------------------------------
# 8. Calculate bootstrap 95% confidence interval
# ------------------------------------------------------------

ci_low = np.percentile(
    boots,
    2.5
)

ci_high = np.percentile(
    boots,
    97.5
)


# ------------------------------------------------------------
# 9. Calculate naive difference
# ------------------------------------------------------------

treated_mean = (
    c.loc[
        c["had_internship"] == 1,
        "earnings_10yr_usd"
    ].mean()
)

control_mean = (
    c.loc[
        c["had_internship"] == 0,
        "earnings_10yr_usd"
    ].mean()
)

naive = treated_mean - control_mean


# ------------------------------------------------------------
# 10. Display results
# ------------------------------------------------------------

print(f"Naive difference : ${naive:,.0f}")

print(
    f"Matched ATT      : ${point:,.0f}"
)

print(
    f"Bootstrap 95% CI : "
    f"${ci_low:,.0f} to ${ci_high:,.0f}"
)

print(
    f"\nNumber of bootstrap samples: {len(boots)}"
)
```

**Output:**

```text

C4. Internship effect on 10-year earnings: matched vs naive
Naive difference : $92,590
Matched ATT      : $42,617
Bootstrap 95% CI : $34,311 to $47,602

Number of bootstrap samples: 100
```

# Leakage-free high_roi model + permutation importance 

```python
print("\nC7. Predicting high_roi WITHOUT leaky columns")
print("=" * 55)

# Create features from categorical variables
feats = pd.get_dummies(
    c[["major", "institution_tier", "region"]],
    dtype=int
)

# Add numerical/binary variables
feats[
    [
        "gpa",
        "had_internship",
        "completed_on_time",
        "net_cost_usd",
        "debt_usd",
        "institution_selectivity_pctile"
    ]
] = c[
    [
        "gpa",
        "had_internship",
        "completed_on_time",
        "net_cost_usd",
        "debt_usd",
        "institution_selectivity_pctile"
    ]
]

# Train/test split
Xtr, Xte, ytr, yte = train_test_split(
    feats,
    c["high_roi"],
    test_size=0.25,
    random_state=0,
    stratify=c["high_roi"]
)

# Train model
gb = HistGradientBoostingClassifier(
    random_state=0
).fit(Xtr, ytr)

# Evaluate using ROC-AUC
test_auc = roc_auc_score(
    yte,
    gb.predict_proba(Xte)[:, 1]
)

print(f"Test AUC: {test_auc:.3f}")

# Permutation importance
pi = permutation_importance(
    gb,
    Xte,
    yte,
    scoring="roc_auc",
    n_repeats=5,
    random_state=0
)

imp = (
    pd.Series(
        pi.importances_mean,
        index=feats.columns
    )
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 Feature Importances:")
print(imp.round(4).to_string())
```

**Output:**

```text

C7. Predicting high_roi WITHOUT leaky columns
=======================================================
Test AUC: 0.900

Top 10 Feature Importances:
institution_selectivity_pctile    0.0335
major_electrical_engineering      0.0282
major_communications              0.0237
major_psychology                  0.0227
major_criminal_justice            0.0210
major_fine_arts                   0.0197
major_business_analytics          0.0189
major_humanities                  0.0186
major_social_work                 0.0182
major_education                   0.0180
```
