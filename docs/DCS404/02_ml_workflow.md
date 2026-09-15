# The Machine Learning Workflow

**Applied Machine Learning Module**

---

Last time we answered *what* machine learning is. Today we tackle a question that sounds simpler but trips up
far more people: once you actually have a problem to solve, *what do you do first, and what do you do next?*

Here's something I've watched happen too many times. A student learns a dozen algorithms, can recite the maths
behind gradient descent, and then sits down in front of a real dataset and freezes — because knowing the tools
is not the same as knowing the process. A project doesn't fail because you picked logistic regression over a
random forest. It fails because nobody defined what "success" meant, or because a crucial feature was missing
and nobody noticed until the very end, or because the model looked brilliant in testing and fell apart the
moment it met the real world.

So today we're going to learn the *process* — a battle-tested recipe called CRISP-DM that professional data
scientists have leaned on since the 1990s — and we're going to walk it end to end on a real dataset, writing
real code at every step, all the way from "here's a business problem" to "here's a saved model ready to
deploy." By the end you'll have built a complete machine learning project, not just a model.

Let's get organised.

## How to work through this

Same deal as last time. Run every code cell (`Shift + Enter`) and actually look at the output before you read
my commentary underneath — the whole notebook is built around *seeing* the result first and then making sense
of it.

A couple of things specific to today. This module uses a real dataset (student records from two Portuguese
schools), and I've bundled it in the `data/` folder next to this notebook, so everything runs offline. And
because this is a *workflow*, the cells build on each other in order — the cleaning step depends on the
loading step, the model depends on the cleaning, and so on. If something errors, the usual cause is running
cells out of order; just run from the top.

As before, watch for the moments where I stop and ask you something or suggest a change to try. They're where
the real learning happens.

## Learning objectives

After completing this module you will be able to:

- Describe the six phases of the CRISP-DM workflow and explain why a structured process matters.
- Turn a vague business goal into a concrete, measurable machine learning problem.
- Explore an unfamiliar dataset: summarise it, visualise its distributions and relationships, and check its
  quality for missing values, outliers, and imbalance.
- Prepare raw data for modelling by cleaning it, encoding categorical variables, and splitting it correctly.
- Train, evaluate, and compare regression models using appropriate metrics.
- Understand what "deployment" involves, including saving and reloading a trained model.

## Setup

Run this once. It loads our libraries, sets a consistent plotting style, and reads in the dataset we'll use
all the way through. If an import fails, install it with `pip install <package>`.


```python
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# A consistent look for every plot in this notebook
plt.rcParams.update({
    "figure.figsize": (7, 4.5),
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})
sns.set_palette("deep")
RANDOM_STATE = 0

# Load the Student Performance dataset. It is semicolon-separated, not comma-separated.
# We look in a couple of likely locations so this runs whether you launched Jupyter
# from the repo root or from inside the notebooks/ folder.
for candidate in [Path("data/student-mat.csv"),
                  Path("notebooks/data/student-mat.csv"),
                  Path("resources/dataset/student-mat.csv")]:
    if candidate.exists():
        df = pd.read_csv(candidate, sep=";")
        print(f"Loaded dataset from: {candidate}")
        break
else:
    raise FileNotFoundError("Could not find student-mat.csv. Expected it in a data/ folder next to the notebook.")

print(f"Shape: {df.shape[0]} students, {df.shape[1]} columns")
df.head()
```

    Loaded dataset from: data/student-mat.csv
    Shape: 395 students, 33 columns





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>school</th>
      <th>sex</th>
      <th>age</th>
      <th>address</th>
      <th>famsize</th>
      <th>Pstatus</th>
      <th>Medu</th>
      <th>Fedu</th>
      <th>Mjob</th>
      <th>Fjob</th>
      <th>...</th>
      <th>famrel</th>
      <th>freetime</th>
      <th>goout</th>
      <th>Dalc</th>
      <th>Walc</th>
      <th>health</th>
      <th>absences</th>
      <th>G1</th>
      <th>G2</th>
      <th>G3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GP</td>
      <td>F</td>
      <td>18</td>
      <td>U</td>
      <td>GT3</td>
      <td>A</td>
      <td>4</td>
      <td>4</td>
      <td>at_home</td>
      <td>teacher</td>
      <td>...</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>5</td>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GP</td>
      <td>F</td>
      <td>17</td>
      <td>U</td>
      <td>GT3</td>
      <td>T</td>
      <td>1</td>
      <td>1</td>
      <td>at_home</td>
      <td>other</td>
      <td>...</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GP</td>
      <td>F</td>
      <td>15</td>
      <td>U</td>
      <td>LE3</td>
      <td>T</td>
      <td>1</td>
      <td>1</td>
      <td>at_home</td>
      <td>other</td>
      <td>...</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>10</td>
      <td>7</td>
      <td>8</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GP</td>
      <td>F</td>
      <td>15</td>
      <td>U</td>
      <td>GT3</td>
      <td>T</td>
      <td>4</td>
      <td>2</td>
      <td>health</td>
      <td>services</td>
      <td>...</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>2</td>
      <td>15</td>
      <td>14</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>GP</td>
      <td>F</td>
      <td>16</td>
      <td>U</td>
      <td>GT3</td>
      <td>T</td>
      <td>3</td>
      <td>3</td>
      <td>other</td>
      <td>other</td>
      <td>...</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>4</td>
      <td>6</td>
      <td>10</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>



---
## 1. The map: CRISP-DM

Before we touch the data, let me give you the map we'll be following. It's called **CRISP-DM** — the
Cross-Industry Standard Process for Data Mining — and despite the mouthful of a name, the idea is
down-to-earth. It's simply a checklist of the six stages that almost every data project moves through, dreamt
up back in the late 1990s and still the default playbook in industry today.

The six phases are:

1. **Business Understanding** — what are we actually trying to achieve, and how will we know if we succeeded?

2. **Data Understanding** — what data do we have, and what shape is it in?

3. **Data Preparation** — cleaning and reshaping the raw data into something a model can eat.

4. **Modelling** — the part everyone thinks of as "machine learning": training the actual model.

5. **Evaluation** — did it work? Is it good enough to trust?

6. **Deployment** — putting the model to work in the real world.

Why bother with a formal process at all? Because, as the old saying goes, *a problem well defined is a problem
half solved.* You can be brilliant at algorithms and still sink a project through plain disorganisation.
Imagine getting all the way to evaluation only to discover the one feature that actually predicts your target
was never collected — weeks of work, gone. CRISP-DM exists to catch that kind of thing early, by making sure
you do the right things in the right order.

And one crucial detail before we start: this is a **loop, not a line.** You'll often reach evaluation, realise
your data wasn't good enough, and circle right back to preparation — or even back to redefining the problem.
That's not failure; that's the process working as intended. Let me draw it.

<center>

<img src ="../assets/crispdm.png" width = "900" height ="700">


Fig: CRISP-DM Process flow 
</center>


---
## 2. Phase 1 — Business Understanding

This is the phase everyone is tempted to skip, and it's the one that most often decides whether a project
lives or dies. Before a single line of code, we ask: *what are we trying to accomplish, and why?* Concretely,
this phase is about defining the problem, understanding who the result is for, taking stock of the data and
resources we have, and — critically — deciding up front what "success" will look like.

Let me make it real with the scenario we'll use all module. Picture a school that wants to **spot students at
risk of a poor final grade early enough to help them.** A counsellor can only give extra attention to a
handful of students, so a vague "some kids are struggling" isn't actionable. They need something specific.

So we translate that wish into a concrete machine learning problem:

- **The goal.** Predict a student's final grade (`G3`, on a 0–20 scale) from information available about them.

- **The task type.** The target is a number, so this is *regression* — the flavour of supervised learning we
  met in Module 1 with the temperature example.

- **Who it's for.** School counsellors, who will use the predictions to prioritise who to support.

- **What success looks like.** On average, our predictions should land within about two grade points of the
  truth. (We'll define exactly how we measure that when we reach evaluation.)

Notice we haven't mentioned an algorithm yet. That's deliberate. The problem definition comes first; the tool
comes later. Now that we know what we're chasing, let's go and meet the data.

---
## 3. Phase 2 — Data Understanding

Now we get to know our data. This is where **exploratory data analysis (EDA)** lives — the detective work of
summarising a dataset and letting it surprise you before you commit to any modelling. Rush this phase and
you'll pay for it later, every time.

### 3.1 Where data comes from

Our data arrived as a tidy CSV file, but that's the easy case. In real projects data comes from all over:
plain text and delimited files, relational and NoSQL databases, web scraping and APIs, live streams. And it
comes in two broad shapes. **Structured** data is the neat, tabular kind — rows and columns, numbers and
categories — the sort you can drop straight into a spreadsheet, like ours. **Unstructured** data is
everything messier: free text, images, audio, video. Most of the world's data is unstructured, which is
exactly why it's so valuable and so hard to work with. Today we're firmly in structured-data territory, which
is the right place to learn the workflow.

This phase is primarily divided into the following sub-steps:

- Data Acquisition: Gathering data

  - Plain text files, delimited files, JSON files

  - Databases (Relational or NOSQL)

  - Web Scraping and APIs

  - URLs, Streaming data


- Data Exploration with verification of its quality

  - Quick Summary of data

  - Attribute Information

  - Descriptive Statistics

  - Histogram, bar plots

  - Correlation Matrix
  
  - Data Quality(DQ) Verification


You should develop the habit of creating a simple document with at least a description of the ideal data needed to test a hypothesis. This will help you:

- Streamline the modeling process.
- Ensure that all the future data come in an improved form.



Data needs to be there to start any data science project or data analysis project. The diagram below represents the common methods of gathering or collecting data.

<center>

<img src ="../assets/datacollection.png" width = "650" height ="450">

Fig: Different ways of Data collection.
</center>

There are different sites in internet such as: _Kaggle_ and _UCI Machine learning Repository_ , from where we can collect the data for machine learning projects.




### 3.2 The dataset: student performance

We're using the **Student Performance** dataset from the UCI Machine Learning Repository (Cortez & Silva,
2008): real records from two Portuguese secondary schools, gathered from school reports and questionnaires.
It has 395 students and 33 columns — 32 input features plus the target.

The features paint a rounded picture of each student. A few worth knowing about now:

| Feature | Meaning |
|---|---|
| `sex`, `age`, `address` | basic demographics |
| `studytime`, `failures`, `absences` | study habits and history |
| `Medu`, `Fedu`, `Mjob`, `Fjob` | parents' education and jobs |
| `goout`, `Dalc`, `Walc`, `health` | lifestyle (social life, alcohol, health) |
| `G1`, `G2` | grades from the first and second periods |
| **`G3`** | **the final grade — this is our target** |

That last group is worth a mental flag already: `G1` and `G2` are earlier grades for the same student, so
we'd expect them to be very strong predictors of the final grade `G3`. Hold that thought — it'll come back
to bite us in an interesting way.

### 3.3 A quick summary

The very first thing I do with any new dataset is ask pandas for a quick summary: how many rows and columns,
what the data types are, and whether anything is missing. `df.info()` gives all of that at a glance.


```python
df.info()
```

    <class 'pandas.DataFrame'>
    RangeIndex: 395 entries, 0 to 394
    Data columns (total 33 columns):
     #   Column      Non-Null Count  Dtype
    ---  ------      --------------  -----
     0   school      395 non-null    str  
     1   sex         395 non-null    str  
     2   age         395 non-null    int64
     3   address     395 non-null    str  
     4   famsize     395 non-null    str  
     5   Pstatus     395 non-null    str  
     6   Medu        395 non-null    int64
     7   Fedu        395 non-null    int64
     8   Mjob        395 non-null    str  
     9   Fjob        395 non-null    str  
     10  reason      395 non-null    str  
     11  guardian    395 non-null    str  
     12  traveltime  395 non-null    int64
     13  studytime   395 non-null    int64
     14  failures    395 non-null    int64
     15  schoolsup   395 non-null    str  
     16  famsup      395 non-null    str  
     17  paid        395 non-null    str  
     18  activities  395 non-null    str  
     19  nursery     395 non-null    str  
     20  higher      395 non-null    str  
     21  internet    395 non-null    str  
     22  romantic    395 non-null    str  
     23  famrel      395 non-null    int64
     24  freetime    395 non-null    int64
     25  goout       395 non-null    int64
     26  Dalc        395 non-null    int64
     27  Walc        395 non-null    int64
     28  health      395 non-null    int64
     29  absences    395 non-null    int64
     30  G1          395 non-null    int64
     31  G2          395 non-null    int64
     32  G3          395 non-null    int64
    dtypes: int64(16), str(17)
    memory usage: 122.6 KB



```python
# How many columns are text (categorical) versus numeric? This shapes our whole preparation plan later.
n_categorical = df.select_dtypes(include="object").shape[1]
n_numeric = df.select_dtypes(include=np.number).shape[1]
print(f"Categorical (text) features : {n_categorical}")
print(f"Numeric features            : {n_numeric}")
print(f"Total                       : {n_categorical + n_numeric}")
```

    Categorical (text) features : 17
    Numeric features            : 16
    Total                       : 33


    /var/folders/cm/07z8g9cd5r3092d58mr51l3w0000gn/T/ipykernel_91886/3426734064.py:2: Pandas4Warning: For backward compatibility, 'str' dtypes are included by select_dtypes when 'object' dtype is specified. This behavior is deprecated and will be removed in a future version. Explicitly pass 'str' to `include` to select them, or to `exclude` to remove them and silence this warning.
    See https://pandas.pydata.org/docs/user_guide/migration-3-strings.html#string-migration-select-dtypes for details on how to write code that works with pandas 2 and 3.
      n_categorical = df.select_dtypes(include="object").shape[1]


So we've got a mix: some columns are numbers already, but a good chunk are text like `"yes"`/`"no"` or
`"teacher"`. File that away — machine learning models only speak numbers, so every one of those text columns
will need converting before we can model. That's a job for Phase 3.

### 3.4 Descriptive statistics

Numbers and text want summarising differently. For the numeric columns we ask for the classic five-number
summary — mean, spread, min, max, quartiles. For the text columns those don't make sense, so instead we ask
how many unique values each has and which is most common.


```python
# Summary of the numeric columns
df.describe(include=[np.number])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>Medu</th>
      <th>Fedu</th>
      <th>traveltime</th>
      <th>studytime</th>
      <th>failures</th>
      <th>famrel</th>
      <th>freetime</th>
      <th>goout</th>
      <th>Dalc</th>
      <th>Walc</th>
      <th>health</th>
      <th>absences</th>
      <th>G1</th>
      <th>G2</th>
      <th>G3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>16.696203</td>
      <td>2.749367</td>
      <td>2.521519</td>
      <td>1.448101</td>
      <td>2.035443</td>
      <td>0.334177</td>
      <td>3.944304</td>
      <td>3.235443</td>
      <td>3.108861</td>
      <td>1.481013</td>
      <td>2.291139</td>
      <td>3.554430</td>
      <td>5.708861</td>
      <td>10.908861</td>
      <td>10.713924</td>
      <td>10.415190</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.276043</td>
      <td>1.094735</td>
      <td>1.088201</td>
      <td>0.697505</td>
      <td>0.839240</td>
      <td>0.743651</td>
      <td>0.896659</td>
      <td>0.998862</td>
      <td>1.113278</td>
      <td>0.890741</td>
      <td>1.287897</td>
      <td>1.390303</td>
      <td>8.003096</td>
      <td>3.319195</td>
      <td>3.761505</td>
      <td>4.581443</td>
    </tr>
    <tr>
      <th>min</th>
      <td>15.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>16.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>9.000000</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>17.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>11.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>18.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>8.000000</td>
      <td>13.000000</td>
      <td>13.000000</td>
      <td>14.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>22.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>75.000000</td>
      <td>19.000000</td>
      <td>19.000000</td>
      <td>20.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Summary of the categorical (text) columns
df.describe(include=["object"])
```

    /var/folders/cm/07z8g9cd5r3092d58mr51l3w0000gn/T/ipykernel_91886/1805765710.py:2: Pandas4Warning: For backward compatibility, 'str' dtypes are included by select_dtypes when 'object' dtype is specified. This behavior is deprecated and will be removed in a future version. Explicitly pass 'str' to `include` to select them, or to `exclude` to remove them and silence this warning.
    See https://pandas.pydata.org/docs/user_guide/migration-3-strings.html#string-migration-select-dtypes for details on how to write code that works with pandas 2 and 3.
      df.describe(include=["object"])





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>school</th>
      <th>sex</th>
      <th>address</th>
      <th>famsize</th>
      <th>Pstatus</th>
      <th>Mjob</th>
      <th>Fjob</th>
      <th>reason</th>
      <th>guardian</th>
      <th>schoolsup</th>
      <th>famsup</th>
      <th>paid</th>
      <th>activities</th>
      <th>nursery</th>
      <th>higher</th>
      <th>internet</th>
      <th>romantic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>395</td>
      <td>395</td>
      <td>395</td>
      <td>395</td>
      <td>395</td>
      <td>395</td>
      <td>395</td>
      <td>395</td>
      <td>395</td>
      <td>395</td>
      <td>395</td>
      <td>395</td>
      <td>395</td>
      <td>395</td>
      <td>395</td>
      <td>395</td>
      <td>395</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>GP</td>
      <td>F</td>
      <td>U</td>
      <td>GT3</td>
      <td>T</td>
      <td>other</td>
      <td>other</td>
      <td>course</td>
      <td>mother</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>yes</td>
      <td>yes</td>
      <td>yes</td>
      <td>yes</td>
      <td>no</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>349</td>
      <td>208</td>
      <td>307</td>
      <td>281</td>
      <td>354</td>
      <td>141</td>
      <td>217</td>
      <td>145</td>
      <td>273</td>
      <td>344</td>
      <td>242</td>
      <td>214</td>
      <td>201</td>
      <td>314</td>
      <td>375</td>
      <td>329</td>
      <td>263</td>
    </tr>
  </tbody>
</table>
</div>



Already there are things to notice. Look at `absences`: the average is a handful of days, but the maximum is
enormous compared to the typical value — a hint that a few students skew things badly, which we'll confirm
when we hunt for outliers. And in the categorical summary you can see, for instance, which school most
students attend. This is the kind of quiet familiarity with the data that pays off later.

### 3.5 Distributions: histograms

Summary numbers hide the *shape* of the data. A histogram fixes that by bucketing each feature into ranges and
counting how many students fall in each — turning a column of numbers into a picture you can read in a second.


```python
# One histogram per numeric feature
axes = df.select_dtypes(include=np.number).hist(figsize=(15, 12), edgecolor="white")
plt.suptitle("Distribution of every numeric feature", y=0.92, fontsize=13)
plt.show()
```


    
![png](02_ml_workflow_files/02_ml_workflow_18_0.png)
    


These little pictures are full of stories. `absences` is heavily skewed — most students miss few classes, a
handful miss many. `failures` is lopsided too: the vast majority have zero past failures. And crucially, look
at `G3`, our target: it's roughly bell-shaped but with a suspicious spike at zero — students who scored 0 on
the final. Those zeros are real and they'll matter. Reading distributions like this is a skill worth
practising; it tells you what you're dealing with before any model does.

### 3.6 Relationships: the correlation matrix

Histograms show one feature at a time. But we also care how features move *together* — and especially how each
one relates to our target. A correlation matrix measures that for every pair of numeric features at once, on a
scale from -1 (move in opposite directions) through 0 (unrelated) to +1 (move together). A heatmap makes it
readable at a glance.


```python
plt.figure(figsize=(11, 9))
corr = df.select_dtypes(include=np.number).corr()
sns.heatmap(corr, vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(220, 20, n=200), annot=False, square=True)
plt.title("Correlation between numeric features")
plt.tight_layout()
plt.show()

# Which features are most correlated with our target, G3?
print("Strongest correlations with the final grade G3:")
print(corr["G3"].drop("G3").sort_values(ascending=False).head(6).round(2))
```


    
![png](02_ml_workflow_files/02_ml_workflow_20_0.png)
    


    Strongest correlations with the final grade G3:
    G2           0.90
    G1           0.80
    Medu         0.22
    Fedu         0.15
    studytime    0.10
    famrel       0.05
    Name: G3, dtype: float64


And there's the payoff, in the printout: `G2` and `G1` are by far the strongest predictors of `G3`, which is
no surprise — a student's earlier grades tell you a lot about their final one. This is genuinely useful to
know before modelling. But it also plants a flag we'll return to: if our model leans almost entirely on `G1`
and `G2`, is it really doing anything clever, and would it even be usable *early* in the year before those
grades exist? Good questions. Park them for now.

### 3.7 The categorical features

Correlation is a numeric idea, so our text columns sat out that last plot. To get a feel for them we count how
often each category appears. Rather than drown you in all 17, here are two representative ones.


```python
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, col in zip(axes, ["school", "Mjob"]):
    df[col].value_counts().plot(kind="bar", ax=ax, color="#60a5fa", edgecolor="white")
    ax.set_title(f"Counts for '{col}'")
    ax.set_xlabel(""); ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.show()
```


    
![png](02_ml_workflow_files/02_ml_workflow_22_0.png)
    


The heights tell you the balance of each category — most students come from one of the two schools, mothers'
jobs cluster in a few categories, and so on. Imbalances like these are worth noting because they can quietly
bias a model toward whatever it saw most of.

### 3.8 Data quality checks

Last stop in getting to know the data, and arguably the most important: is it any *good*? Three quick checks
catch most problems — missing values, outliers, and an imbalanced target.


```python
# 1) Missing values: what fraction of each column is empty?
missing = df.isna().mean().sort_values(ascending=False)
print("Columns with the most missing values:")
print(missing.head().round(3))
print(f"\nTotal missing values in the entire dataset: {int(df.isna().sum().sum())}")
```

    Columns with the most missing values:
    school      0.0
    paid        0.0
    G2          0.0
    G1          0.0
    absences    0.0
    dtype: float64
    
    Total missing values in the entire dataset: 0


Good news — not a single missing value. That's rare in the real world (usually you're patching holes
everywhere), but it means we get to skip missing-data handling this time. We'll still talk about how you'd
handle it, because next time you won't be so lucky.

Now outliers — those extreme values that sit far from everything else. A boxplot is the classic way to spot
them: anything drawn as a dot beyond the "whiskers" is a candidate outlier.


```python
plt.figure(figsize=(15, 6))
df.select_dtypes(include=np.number).boxplot(rot=90)
plt.title("Boxplots of numeric features — dots beyond the whiskers are candidate outliers")
plt.tight_layout()
plt.show()
```


    
![png](02_ml_workflow_files/02_ml_workflow_26_0.png)
    


`absences` jumps out immediately, with a long tail of students way above the norm. A handful of features have
a few outliers too. We'll deal with `absences` in the preparation phase.

Finally, is our target balanced? For a regression target we just look at its distribution — if some grade
values are rare, the model will see few examples of them and struggle there.


```python
plt.figure(figsize=(13, 5))
sns.countplot(x="G3", data=df, color="#60a5fa", edgecolor="white")
plt.title("How many students earned each final grade (G3)")
plt.xlabel("Final grade G3 (0-20)"); plt.ylabel("Number of students")
plt.tight_layout()
plt.show()
```


    
![png](02_ml_workflow_files/02_ml_workflow_28_0.png)
    


This is imbalanced: some grades are common, others (the very low and very high ones) are represented by only a
few students. That spike at 0 we spotted earlier is clearly visible. It doesn't mean those students don't
count — quite the opposite — but it does mean the model will have less to learn from at the extremes, which is
worth remembering when we judge its performance.

That's Phase 2 done. We now know our data's shape, its distributions, its relationships, and its flaws. Time to
roll up our sleeves and prepare it.

---
## 4. Phase 3 — Data Preparation

Here's a statistic that surprises every new data scientist: this phase — cleaning and reshaping the data — is
where you'll spend the *majority* of your time on most real projects. It's unglamorous, nobody puts it in the
demo, and it matters enormously. A brilliant algorithm fed messy data produces messy predictions. Garbage in,
garbage out, exactly as we said last module.

We'll do three things: clean the data, encode the text columns into numbers, and split it for training and
testing.

### 4.1 Cleaning the data

Cleaning covers a lot of ground — handling missing values, taming outliers, removing duplicates, fixing typos.
Let me touch the two that matter here.

**Missing values.** We're lucky today; there are none. But you won't always be, so know your options. You can
*delete* rows or columns with missing data (simple, but you throw away information), or you can *impute* —
fill the gaps with the column's mean or median, or its most frequent value. Which one is right depends on how
much is missing and why.

**Outliers.** These we do have, in `absences`. Rather than delete those students (they're real, and deleting
data is a big decision), we'll *cap* the feature: any absence count above a sensible threshold gets pulled
down to that threshold. This keeps the student in the dataset while stopping a few extreme values from
dominating. Let me cap at 15 days.


```python
# Cap the 'absences' outliers at 15 so a few extreme students don't dominate the model.
before_max = df["absences"].max()
df["absences"] = np.where(df["absences"] > 15, 15, df["absences"])
print(f"'absences' maximum before capping: {before_max}")
print(f"'absences' maximum after capping : {df['absences'].max()}")
```

    'absences' maximum before capping: 75
    'absences' maximum after capping : 15


### 4.2 Encoding the text columns

Remember our earlier flag: models only understand numbers, but 17 of our columns are text. We have to convert
them, and *how* we convert them depends on the kind of category.

**Label encoding** simply assigns each category a number: `no` becomes 0, `yes` becomes 1. This is perfect for
two-valued columns, and for columns with a real order to them. But it has a trap — it implies an *ordering*.
If we label-encoded `Mjob` as teacher=0, health=1, services=2, the model would wrongly assume services is
"more" than teacher, which is nonsense; job is not a quantity.

**One-hot encoding** solves that. For a column with no natural order, it creates a separate 0/1 column for
each category — a `Mjob_teacher` column, a `Mjob_health` column, and so on — so no false ordering sneaks in.
The cost is extra columns, but it's the honest choice for unordered ("nominal") categories.

So our plan: label-encode the yes/no binary columns, and one-hot encode the genuinely unordered ones.


```python
from sklearn.preprocessing import LabelEncoder

# Genuinely unordered ("nominal") categories -> one-hot encode
cat_nominal = ["Mjob", "Fjob", "reason", "guardian"]
# Two-valued ("binary") categories -> label encode to 0/1
cat_binary = ["school", "sex", "address", "famsize", "Pstatus", "schoolsup",
              "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"]

# Label-encode each binary column to 0/1
for col in cat_binary:
    df[col] = LabelEncoder().fit_transform(df[col])

# One-hot encode the nominal columns (dtype=int keeps them as clean 0/1 rather than True/False)
df = pd.get_dummies(df, columns=cat_nominal, prefix=cat_nominal, dtype=int)

print(f"After encoding, the dataset has {df.shape[1]} columns (it grew because of one-hot encoding).")
print("Every column is now numeric:", df.select_dtypes(include="object").empty)
df.head()
```

    After encoding, the dataset has 46 columns (it grew because of one-hot encoding).
    Every column is now numeric: True





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>school</th>
      <th>sex</th>
      <th>age</th>
      <th>address</th>
      <th>famsize</th>
      <th>Pstatus</th>
      <th>Medu</th>
      <th>Fedu</th>
      <th>traveltime</th>
      <th>studytime</th>
      <th>...</th>
      <th>Fjob_other</th>
      <th>Fjob_services</th>
      <th>Fjob_teacher</th>
      <th>reason_course</th>
      <th>reason_home</th>
      <th>reason_other</th>
      <th>reason_reputation</th>
      <th>guardian_father</th>
      <th>guardian_mother</th>
      <th>guardian_other</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>18</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>17</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>16</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 46 columns</p>
</div>



Every column is a number now, and the dataset grew a little because one-hot encoding split each nominal column
into several. The data is finally in a form a model can consume.

### 4.3 Splitting into training and test sets

This last step is small in code and huge in importance, so let me be emphatic about it. We are going to split
our data into two piles. The model gets to learn from the **training set** only. The **test set** we lock in a
drawer and never let the model see during training — it's the final exam, our only honest measure of how the
model will do on students it has never encountered.

Why so strict? Because if you let a model study the test before grading it on the test, of course it scores
well — and that score is a lie. A model that has secretly seen the answers tells you nothing about the real
world. The whole point of holding data back is to keep ourselves honest. We'll use an 80/20 split: 80% to
learn from, 20% to be judged on.

<center>




<p><img src="data:image/png;base64,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AAlE2QAAAAAAAAAAAAAACUTZAAAAAAAAAAAAAAAJRNkAAAAAAAAAAAAAAAlE2QAAAAAAAAAAAAAACUTZAAAAAAAAAAAAAAAJRNkAAAAAAAAAAAAAAAlE2QAAAAAAAAAAAAAACUTZAAAAAAAAAAAAAAAJRNkAAAAAAAAAAAAAAAlE2QAAAAAAAAAAAAAACUTZAAAAAAAAAAAAAAAJRNkAAAAAAAAAAAAAAAlE2QAAAAAAAAAAAAAACUTZAAAAAAAAAAAAAAAJRNkAAAAAAAAAAAAAAAlE2QAAAAAAAAAAAAAACUTZAAAAAAAAAAAAAAAJRNkAAAAAAAAAAAAAAAlE2QAAAAAAAAAAAAAACUTZAAAAAAAAAAAAAAAJRNkAAAAAAAAAAAAAAAlE2QAAAAAAAAAAAAAACUTZAAAAAAAAAAAAAAAJRNkAAAAAAAAAAAAAAAlE2QAAAAAAAAAAAAAACUTZAAAAAAAAAAAAAAAJRNkAAAAAAAAAAAAAAAlE2QAAAAAAAAAAAAAACUTZAAAAAAAAAAAAAAAJRNkAAAAAAAAAAAAAAAlE2QAAAAAAAAAAAAAACUTZAAAAAAAAAAAAAAAJRNkAAAAAAAAAAAAAAAlE2QAAAAAAAAAAAAAACUTZAAAAAAAAAAAAAAAJRNkAAAAAAAAAAAAAAAlE2QAAAAAAAAAAAAAACUTZAAAAAAAAAAAAAAAJRNkAAAAAAAAAAAAAAAlE2QAAAAAAAAAAAAAACUTZAAAAAAAAAAAAAAAJRNkAAAAAAAAAAAAAAAlE2QAAAAAAAAAAAAAACUTZAAAAAAAAAAAAAAAJRNkAAAAAAAAAAAAAAAlE2QAAAAAAAAAAAAAACUTZAAAAAAAAAAAAAAAJRNkAAAAAAAAAAAAAAAlE2QAAAAAAAAAAAAAACUTZAAAAAAAAAAAAAAAJRNkAAAAAAAAAAAAAAAlE2QAAAAAAAAAAAAAACUTZAAAAAAAAAAAAAAAJRNkAAAAAAAAAAAAAAAlE2QAAAAAAAAAAAAAACUTZAAAAAAAAAAAAAAAJRNkAAAAAAAAAAAAAAAlE2QAAAAAAAAAAAAAACUTZAAAAAAAAAAAAAAAJRNkAAAAAAAAAAAAAAAlE2QAAAAAAAAAAAAAACUTZAAAAAAAAAAAAAAAJRNkAAAAAAAAAAAAAAAlE2QAAAAAAAAAAAAAACUTZ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omwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADKJsAAAAAAAAAAAAAIAMomwAAAAAAAAAAAAAgAyibAAAAAAAAAAAAACADPMsHk3sZ24OAAAAAAAAQG39v46uv0gp/ckyAwDQIva6UTSjjvHxcTcGAAAAAAAAAAAAAKCkTgsHAAAAAAAAAAAAAFCeKBsAAAAAAAAAAAAAIIMoGwAAAAAAAAAAAAAggygbAAAAAAAAAAAAACCDKBsAAAAAAAAAAAAAIIMoGwAAAAAAAAAAAAAggygbAAAAAAAAAAAAACCDKBsAAAAAAAAAAAAAIIMoGwAAAAAAAAAAAAAggygbAAAAAAAAAAAAACCDKBsAAAAAAAAAAAAAIIMoGwAAAAAAAAAAAAAggygbAAAAAAAAAAAAACCDKBsAAAAAAAAAAAAAIIMoGwAAAAAAAAAAAAAggygbAAAAAAAAAAAAACCDKBsAAAAAAAAAAAAAIIMoGwAAAAAAAAAAAAAggygbAAAAAAAAAAAAACCDKBsAAAAAAAAAAAAAIIMoGwAAAAAAAAAAAAAggygbAAAAAAAAAAAAACCDKBsAAAAAAAAAAAAAIIMoGwAAAAAAAAAAAAAggygbAAAAAAAAAAAAACCDKBsAAAAAAAAAAAAAIIMoGwAAAAAAAAAAAAAggygbAAAAAAAAAAAAACCDKBsAAAAAAAAAAAAAIIMoGwAAAAAAAAAAAAAgg3cxp1oAAARFSURBVCgbAAAAAAAAAAAAACCDKBsAAAAAAAAAAAAAIIMoGwAAAAAAAAAAAAAggygbAAAAAAAAAAAAACCDKBsAAAAAAAAAAAAAIIMoGwAAAAAAAAAAAAAggygbAAAAAAAAAAAAACCDKBsAAAAAAAAAAAAAIIMoGwAAAAAAAAAAAAAggygbAAAAAAAAAAAAACCDKBsAAAAAAAAAAAAAIIMoGwAAAAAAAAAAAAAggygbAAAAAAAAAAAAACCDKBsAAAAAAAAAAAAAIIMoGwAAAAAAAAAAAAAggygbAAAAAAAAAAAAACCDKBsAAAAAAAAAAAAAIIMoGwAAAAAAAAAAAAAggygbAAAAAAAAAAAAACCDKBsAAAAAAAAAAAAAIIMoGwAAAAAAAAAAAAAggygbAAAAAAAAAAAAACCDKBsAAAAAAAAAAAAAIIMoGwAAAAAAAAAAAAAggygbAAAAAAAAAAAAACCDKBsAAAAAAAAAAAAAIIMoGwAAAAAAAAAAAAAggygbAAAAAAAAAAAAACCDKBsAAAAAAAAAAAAAIIMoGwAAAAAAAAAAAAAggygbAAAAAAAAAAAAACCDKBsAAAAAAAAAAAAAIIMoGwAAAAAAAAAAAAAggygbAAAAAAAAAAAAACCDKBsAAAAAAAAAAAAAIIMoGwAAAAAAAAAAAAAggygbAAAAAAAAAAAAACCDKBsAAAAAAAAAAAAAIIMoGwAAAAAAAAAAAAAggygbAAAAAAAAAAAAACCDKBsAAAAAAAAAAAAAIIMoGwAAAAAAAAAAAAAggygbAAAAAAAAAAAAACCDKBsAAAAAAAAAAAAAIIMoGwAAAAAAAAAAAAAggygbAAAAAAAAAAAAACCDKBsAAAAAAAAAAAAAIIMoGwAAAAAAAAAAAAAggygbAAAAAAAAAAAAACCDKBsAAAAAAAAAAAAAIIMoGwAAAAAAAAAAAAAggygbAAAAAAAAAAAAACCDKBsAAAAAAAAAAAAAIIMoGwAAAAAAAAAAAAAggygbAAAAAAAAAAAAACCDKBsAAAAAAAAAAAAAIIMoGwAAAAAAAAAAAAAggygbAAAAAAAAAAAAACCDKBsAAAAAgP/frh0TAADAMAyqf9XzsLygAwAAAAAAIJCyAQAAAAAAAAAAAAACKRsAAAAAAAAAAAAAIJCyAQAAAAAAAAAAAAACKRsAAAAAAAAAAAAAIJCyAQAAAAAAAAAAAAACKRsAAAAAAAAAAAAAIJCyAQAAAAAAAAAAAAACKRsAAAAAAAAAAAAAIJCyAQAAAAAAAAAAAAACKRsAAAAAAAAAAAAAIJCyAQAAAAAAAAAAAAACKRsAAAAAAAAAAAAAIJCyAQAAAAAAAAAAAAACKRsAAAAAAAAAAAAAIJCyAQAAAAAAAAAAAAACKRsAAAAAAAAAAAAAIJCyAQAAAAAAAAAAAAACKRsAAAAAAAAAAAAA4GvbAfa1yxNVaEpzAAAAAElFTkSuQmCC"  ></p>

Fig: Splitting Dataset
</center>




```python
from sklearn.model_selection import train_test_split

y = df["G3"]                 # what we want to predict: the final grade
X = df.drop(columns=["G3"])  # everything else: the features

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE)

print(f"Training set: {X_train.shape[0]} students  (the model learns from these)")
print(f"Test set    : {X_test.shape[0]} students  (locked away until evaluation)")
```

    Training set: 316 students  (the model learns from these)
    Test set    : 79 students  (locked away until evaluation)


One more idea to mention, because you'll meet it constantly: **feature scaling.** Some algorithms care a lot
about the *scale* of features — an `age` of 18 and an `absences` of 15 live on totally different number
ranges, and certain models get confused by that. Scaling squeezes every feature into a comparable range so no
single one dominates just because its numbers happen to be bigger. The model we're about to use, plain linear
regression, doesn't strictly need it, so we'll keep things simple here — but tuck the concept away, because
for algorithms like SVMs and KNN it's essential.

And that's data preparation. Cleaned, encoded, split. The unglamorous majority of the work is behind us — now
for the fun part.

---
## 5. Phase 4 — Modelling

This is the phase everyone shows up for. But notice how much had to happen before we got here — that's the
real lesson of the workflow. With clean, prepared data in hand, the modelling itself is almost anticlimactic.

Choosing a model means weighing a few things: the kind of problem (ours is regression), how much data we have,
how fast it needs to run, and how important it is that we can *explain* the model's decisions. We'll start with
the natural first choice for a regression problem: **linear regression**, the very model that rediscovered the
temperature formula last module. It's fast, and its simplicity makes it a great baseline — you always want a
simple model to measure fancier ones against.

Then, to see whether something more powerful helps, we'll also train a **random forest** — a model that
combines many decision trees (we'll unpack how it works in a later module). For now, treat it as a stronger,
more flexible competitor and let's see if it earns its keep.


```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Baseline: a simple, fast, interpretable linear model
linreg = LinearRegression()
linreg.fit(X_train, y_train)   # this line is where the learning happens

# Challenger: a more flexible ensemble of decision trees
forest = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)
forest.fit(X_train, y_train)

print("Both models trained.")
print("Remember: they have only seen the training set. The test set is still locked away.")
```

    Both models trained.
    Remember: they have only seen the training set. The test set is still locked away.


That's it — two trained models. The `.fit()` call is where all the objective-and-optimization machinery from
Module 1 quietly ran. Now comes the moment of truth: are they any good?

---
## 6. Phase 5 — Evaluation

Training a model is easy. Knowing whether to *trust* it is the hard, essential part — and it's where we
finally open that locked drawer and test on data the models have never seen.

For a regression problem like ours, three metrics tell the story:

- **MAE (Mean Absolute Error)** — on average, how many grade points off are we? The most intuitive one: an
  MAE of 1.5 means our typical prediction misses by about 1.5 grades. Lower is better.
- **RMSE (Root Mean Squared Error)** — similar, but it squares the errors before averaging, so it punishes
  big misses much harder than small ones. Lower is better. If RMSE is much larger than MAE, you have some
  badly-wrong predictions dragging things up.
- **R² (R-squared)** — what fraction of the variation in grades the model actually explains, from 0 (useless,
  no better than always guessing the average) to 1 (perfect). Higher is better.

Let's compute all three, for both models, on the held-out test set.


```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate(model, name):
    preds = model.predict(X_test)
    return {
        "Model": name,
        "MAE":  mean_absolute_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "R2":   r2_score(y_test, preds),
    }

results = pd.DataFrame([evaluate(linreg, "Linear Regression"),
                        evaluate(forest, "Random Forest")]).set_index("Model")
results.round(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MAE</th>
      <th>RMSE</th>
      <th>R2</th>
    </tr>
    <tr>
      <th>Model</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Linear Regression</th>
      <td>1.583</td>
      <td>2.339</td>
      <td>0.802</td>
    </tr>
    <tr>
      <th>Random Forest</th>
      <td>1.133</td>
      <td>1.894</td>
      <td>0.870</td>
    </tr>
  </tbody>
</table>
</div>



Read that table like a scorecard. The MAE tells you, in plain grade points, how far off a typical prediction
is — and for both models it should be comfortably within the two-point target we set back in Business
Understanding. That's the number I'd actually report to the school counsellor, because it's the one they can
feel. Compare the two models and you'll see whether the random forest's extra complexity bought us anything;
often on tidy tabular data like this, the humble linear model is startlingly hard to beat, which is a lesson
in itself.

Numbers in a table are abstract, though. Let me *show* you how good the predictions are by plotting each
predicted grade against the student's actual grade. Perfect predictions would sit exactly on the diagonal
line; the tighter the cloud hugs that line, the better.


```python
# Predicted vs actual for the linear model. The closer to the diagonal, the better.
preds = linreg.predict(X_test)

plt.figure(figsize=(6.5, 6.5))
plt.scatter(y_test, preds, alpha=0.6, edgecolor="white", color="#3b82f6")
lims = [-1, 21]
plt.plot(lims, lims, "--", color="gray", label="perfect prediction")
plt.xlim(lims); plt.ylim(lims)
plt.xlabel("Actual final grade")
plt.ylabel("Predicted final grade")
plt.title("Predicted vs. actual grades (linear regression)")
plt.legend()
plt.tight_layout()
plt.show()
```


    
![png](02_ml_workflow_files/02_ml_workflow_44_0.png)
    


Most points hug the diagonal nicely, which is what we want to see. But look at the bottom-left: the students
who actually scored 0 are predicted to have positive grades — the model refuses to believe in those zeros,
because they were rare and unusual (exactly the imbalance we flagged in Phase 2). This is the value of
plotting your results rather than trusting a single summary number: the picture shows you *where* the model
struggles, not just how much.

One more question every practitioner asks: *which features is the model actually leaning on?* A random forest
can tell us how important each feature was to its predictions. Let's look at the top few.


```python
# The random forest's view of which features mattered most
importances = pd.Series(forest.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(8, 5))
importances.head(10).iloc[::-1].plot(kind="barh", color="#60a5fa", edgecolor="white")
plt.title("Top 10 most important features (random forest)")
plt.xlabel("Relative importance")
plt.tight_layout()
plt.show()
```


    
![png](02_ml_workflow_files/02_ml_workflow_46_0.png)
    


And there's the flag we planted way back in Phase 2, now fully raised: `G2` and `G1` dominate everything else.
The model is essentially predicting the final grade from the two grades just before it, which is accurate but
maybe not very *insightful* — and useless if a counsellor wants to intervene early in the year, before those
grades even exist. This is a genuine, real-world modelling dilemma, and it's a perfect example of why
evaluation loops back to the earlier phases: a school might well ask us to *drop* `G1` and `G2` and rebuild,
trading accuracy for a model that's actually usable when it matters. (That's one of your exercises.)

This is also the phase where we'd normally check for **overfitting** — a model that aced the training data but
flops on the test data, having memorised rather than learned. Comparing training and test scores is the usual
tell, and it's a theme we'll develop properly in later modules.

---
## 7. Phase 6 — Deployment

A model sitting in a notebook helps no one. Deployment is the final phase, where the model actually goes to
work — and it's the step students most often forget exists, even though in industry it's often the hardest
part.

Deployment usually means three things: writing up what you built and how well it performs (so stakeholders
trust it), wrapping the model in something the rest of the world can call, and setting up monitoring to catch
the model quietly degrading over time as the world changes underneath it.

The technical heart of it is being able to *save* a trained model and *reload* it later somewhere else —
inside a web server, say — without retraining from scratch. Let me show you that, because it's satisfyingly
simple.


```python
import joblib
import tempfile, os

# Save the trained model to a file. In a real project this path would live on a server;
# here we use a temporary file just to demonstrate the round trip.
model_path = os.path.join(tempfile.gettempdir(), "student_grade_model.joblib")
joblib.dump(linreg, model_path)
print(f"Model saved to: {model_path}")

# ... later, in a completely separate program, we load it back and use it. No retraining needed.
loaded_model = joblib.load(model_path)

# Predict the final grade for the first student in our test set
one_student = X_test.iloc[[0]]
predicted = loaded_model.predict(one_student)[0]
actual = y_test.iloc[0]
print(f"\nReloaded model predicts this student's final grade: {predicted:.1f}")
print(f"Their actual final grade was                     : {actual}")
```

    Model saved to: /var/folders/cm/07z8g9cd5r3092d58mr51l3w0000gn/T/student_grade_model.joblib
    
    Reloaded model predicts this student's final grade: 14.4
    Their actual final grade was                     : 14


That round trip — train once, save, then load and predict anywhere — is the backbone of real deployment. The
saved file *is* the model, ready to be dropped onto a server.

To turn it into something a website or app can call, you'd typically wrap it in a small web service. Here's the
shape of it (a Flask app), just so you recognise it when you meet it for real in the final module — don't run
this here, it's for illustration:

```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("student_grade_model.joblib")   # load the saved model once, at startup

@app.route("/predict", methods=["POST"])
def predict():
    features = request.get_json()                   # receive a student's features as JSON
    grade = model.predict([features])[0]            # run the model
    return jsonify({"predicted_grade": round(float(grade), 1)})

# A counsellor's dashboard could now POST a student's data to /predict and get a grade back.
```

That's the whole journey — business question to a live, callable model. The full deployment story (Flask,
packaging, serving) is the finale of this course, in Module 13.

---
## 8. Putting it all together

Step back and look at the ground we covered. We started with a fuzzy wish — "help students before they fail" —
and walked it, phase by phase, all the way to a saved model that can predict a final grade on demand. Here's
the whole journey on one page:

| Phase | What it asks | What we actually did |
|---|---|---|
| 1. Business Understanding | What are we solving, and what is success? | Predict final grade `G3`; success = within ~2 points |
| 2. Data Understanding | What have we got? | Explored 395 students, saw skew, outliers, and that `G1`/`G2` dominate |
| 3. Data Preparation | Get it model-ready | Capped `absences`, encoded 17 text columns, split 80/20 |
| 4. Modelling | Build the thing | Trained linear regression and a random forest |
| 5. Evaluation | Is it any good? | Measured MAE/RMSE/R², plotted predictions, found the `G1`/`G2` dependency |
| 6. Deployment | Put it to work | Saved and reloaded the model; sketched a Flask API |

And remember the single most important feature of this whole map: **it's a loop.** Our evaluation surfaced a
real problem — the model relies on grades that don't exist early in the year — and the honest next move is to
walk right back to Business Understanding and Data Preparation and try again with a more useful set of
features. That circling back isn't a detour; it *is* data science. You rarely get it perfect on the first
lap.


<center>
<img src = "../assets/crispdminfographics.png" width="900" height="700">

Fig: CRISP-DM Infographics
</center>


---
## 9. Your turn

The only way this workflow becomes second nature is by walking it yourself. Add cells under each exercise and
have a go — several of these deliberately send you back to an earlier phase, which is exactly the mindset we're
building.

**Exercise 1 — Build the "early warning" model.**
This is the big one, and it follows directly from our evaluation. Rebuild the whole modelling and evaluation
step but *drop `G1` and `G2` from the features* (`X = df.drop(columns=["G3", "G1", "G2"])`). How much worse do
MAE, RMSE, and R² get? Write a sentence for the school: is a model that works early in the year, but less
accurately, worth the trade?

**Exercise 2 — A third model.**
Add a `DecisionTreeRegressor` (from `sklearn.tree`) to the comparison table alongside the linear model and the
forest. Where does it land? Is more complexity always better here?

**Exercise 3 — Explore a feature you're curious about.**
Pick any feature (say `studytime`, `goout`, or `Dalc`) and make a plot of it against `G3` — a boxplot or a
scatter. Does more study time actually track with higher grades in this data? Does going out track with lower?

**Exercise 4 — Change the split.**
Re-run the train/test split with `test_size=0.3` and a different `random_state`. Do the evaluation numbers
move? What does that wobble tell you about trusting a single split? (This is the seed of *cross-validation*,
coming in Module 13.)

**Exercise 5 — Try scaling.**
Standardise the features with `StandardScaler` (from `sklearn.preprocessing`) before fitting the linear model.
Does the score change? Now explain *why* it barely moves for linear regression but would matter a great deal
for a distance-based model like KNN.

**Reflection questions.**
- We spent far more of this notebook on understanding and preparing data than on the model itself. Why do you
  think that ratio holds in real projects too?
- Our model predicts grades well but leans almost entirely on prior grades. Is a model that's accurate but not
  *actionable* a success or a failure? Who decides?
- CRISP-DM is a loop. Describe a realistic situation where evaluating a model would send you all the way back
  to Business Understanding.

---
## 10. If you remember nothing else

1. Machine learning is a *process*, not just a model. CRISP-DM names its six phases: Business Understanding,
   Data Understanding, Data Preparation, Modelling, Evaluation, Deployment.

2. Define the problem and what success means *before* you write any code. A problem well defined is half
   solved.

3. Exploratory data analysis — summaries, distributions, correlations, quality checks — is detective work you
   do before modelling, and it saves you from nasty surprises later.

4. Data preparation (cleaning, encoding text into numbers, splitting) is where most of the real work lives.
   Models only speak numbers, and garbage in means garbage out.

5. Always keep an untouched test set. A model graded on data it trained on tells you a comforting lie.

6. Evaluate with metrics that fit the problem (MAE, RMSE, R² for regression) *and* with plots — the picture
   shows you where the model fails, not just how often.

7. Deployment — saving, serving, and monitoring the model — is a real phase, not an afterthought.

8. It's a loop. Circling back to an earlier phase when something doesn't hold up is the process working, not
   failing.

---
## 11. Further reading and glossary

### Further reading

- The CRISP-DM 1.0 [step-by-step guide](https://www.the-modeling-agency.com/crisp-dm.pdf) — the original,
  surprisingly readable reference.
- P. Cortez and A. Silva, [Using Data Mining to Predict Secondary School Student Performance](http://www3.dsi.uminho.pt/pcortez/student.pdf) — the paper behind our dataset.
- The [UCI Student Performance dataset](https://archive.ics.uci.edu/ml/datasets/Student+Performance) — full
  documentation and the raw files.
- scikit-learn's [Preprocessing guide](https://scikit-learn.org/stable/modules/preprocessing.html) — the
  practical reference for encoding and scaling.

### Glossary

| Term | Meaning |
|---|---|
| CRISP-DM | A six-phase standard process for data mining and ML projects. |
| EDA | Exploratory Data Analysis — summarising and visualising data to understand it. |
| Structured / unstructured data | Tabular data vs. free-form data like text, images, audio. |
| Outlier | An extreme value far from the rest of the observations. |
| Imbalanced data | When some target values or classes are far rarer than others. |
| Label encoding | Turning categories into single integers (good for binary or ordered categories). |
| One-hot encoding | Turning each category into its own 0/1 column (good for unordered categories). |
| Training / test set | Data the model learns from vs. held-out data used to judge it honestly. |
| Feature scaling | Rescaling features to a comparable range so none dominates by magnitude alone. |
| MAE / RMSE / R² | Regression metrics: average error, error that punishes big misses, variance explained. |
| Deployment | Putting a trained model into real use, including saving, serving, and monitoring it. |

---

That's the workflow, start to finish. You've now built a complete machine learning project — not just a model,
but the whole disciplined journey around it. From here on, every algorithm we learn slots into Phase 4 of this
same map, and you already know how to wrap a proper project around it. Next up, Module 2: we open up linear
regression and finally see, gear by gear, how it learns.

*DCS 404 — Data Science and Machine Learning*
