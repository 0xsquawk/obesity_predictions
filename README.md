# Estimation of Obesity Levels Based on Eating Habits and Physical Condition

This project looks at whether we can predict a person's obesity level just from their eating habits, activity levels, and a few demographic details. The data covers people from Mexico, Peru, and Colombia.

## Dataset

- **Source:** [UCI Machine Learning Repository - Estimation of Obesity Levels](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)
- People are grouped into seven obesity categories, from Insufficient Weight up to Obesity Type III, based on things like how often they eat vegetables, how much water they drink, how they get around, and family history.

## What's in the notebook

The notebook is split into three parts:

1. **EDA** - looking at the data before touching any models. Are the classes balanced? Do any features stand out?
2. **Feature engineering and preprocessing** - scaling numbers, encoding categories, and trying out a BMI feature to see if it helps.
3. **Model selection and training** - trying a few different models, picking the best one, tuning it, and checking how it does on data it hasn't seen.

## What we found in the EDA

- The seven obesity classes are fairly balanced, so there was no need to do any resampling and accuracy is a fine metric to judge models by.
- People with a family history of being overweight are much more likely to fall into the higher obesity categories.
- Weight climbs in a pretty obvious step pattern as obesity level goes up, which makes sense since weight is part of how obesity is measured in the first place.
- None of the numerical features are strongly correlated with each other, so they're each adding something useful rather than repeating the same information.

## First attempt: adding a BMI feature (and why it was a mistake)

We calculated BMI from weight and height and dropped weight and height so the model wouldn't have redundant info. Gradient Boosting came out on top with 98% accuracy, which sounds great, but when we checked feature importance, BMI alone was responsible for about 90% of the model's decisions. That's a problem: BMI is just weight and height combined, and those are basically how obesity is defined in this dataset. So the model wasn't really learning from eating habits or lifestyle, it was just doing the obesity math backwards. This is a classic case of target leakage.

## Second attempt: dropping BMI and using real lifestyle features

Once BMI was removed, the models had to work with actual habits and lifestyle info instead of a shortcut. This time Random Forest came out ahead, with 85% accuracy on the test set. That number is lower than the first attempt, but it's a much more honest one, since the model is now actually learning from things like diet, activity, and transport rather than reverse-engineering the answer.

## How the models were compared

Four models were tested side by side: Logistic Regression, Random Forest, SVM, and Gradient Boosting. To keep the comparison fair, all four were run under the exact same conditions:

- same training data
- same 5-fold cross-validation setup (data split into 5 chunks, each model trained on 4 and tested on the 1 left out, repeated 5 times)
- same scoring metric (accuracy)

Instead of judging each model on a single run, we looked at the average accuracy and how much it varied across the 5 folds. That way one lucky or unlucky split doesn't throw off the whole comparison. It's basically the same idea as running an A/B test: test everyone under identical conditions, look at the average result rather than a single sample, and only then pick a winner.

This whole comparison was run twice:

- **Round 1** (with the BMI feature): Gradient Boosting won with 98% average accuracy.
- **Round 2** (without BMI): Random Forest won with 85% average accuracy.

Whichever model won each round was then tuned further using grid search, which itself uses 5-fold cross-validation to try out different parameter combinations and find the best one. After tuning, the winning model was tested exactly once on the untouched test set, just to confirm it actually holds up outside of cross-validation.

## Tools used

- **Language:** Python
- **Data handling:** pandas, numpy
- **Plotting:** matplotlib, seaborn
- **Modeling:** scikit-learn (Logistic Regression, Random Forest, SVM, Gradient Boosting, ColumnTransformer, GridSearchCV)
- **Getting the data:** ucimlrepo

## Running it yourself

Install what you need:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn ucimlrepo itables
```

Then open `py_obesity_analysis.ipynb` in Jupyter and run the cells in order. The dataset pulls straight from UCI, so there's nothing extra to download.

## Notebook layout

| Section | What's there |
|---|---|
| Setup & data import | Loading libraries, pulling the dataset from UCI |
| 1. EDA | Class balance, feature distributions, correlation heatmap |
| 2. Feature engineering | BMI feature, encoding, scaling, train/test split |
| 3. Model selection & training | Comparing models, tuning the winner, feature importance, spotting the leakage, and the final lifestyle-based model |

## Contributor

**Charchit (0xSquawk)** - [github.com/0xsquawk](https://github.com/0xsquawk)
