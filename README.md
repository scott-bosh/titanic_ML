# Titanic Survival Prediction
## Machine Learning from Disaster

### Overview
This project tackles the legendary Titanic Machine Learning competition from Kaggle - a perfect introduction to ML competitions and predictive modeling. The goal is to predict which passengers survived the Titanic shipwreck using machine learning techniques.

### The Challenge
On April 15, 1912, the widely considered "unsinkable" RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren't enough lifeboats for everyone on board, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others. This project aims to build a predictive model that answers the question: "What sorts of people were more likely to survive?" using passenger data (name, age, gender, socio-economic class, etc.).

### Dataset Description

#### Files
- `train.csv` - Training set
- `test.csv` - Test set


#### Features
| Variable | Definition | Notes |
|----------|------------|-------|
| survival | Survival | 0 = No, 1 = Yes |
| pclass | Ticket class | 1 = 1st (Upper), 2 = 2nd (Middle), 3 = 3rd (Lower) |
| sex | Sex | |
| age | Age in years | - Fractional if less than 1 year<br>- Estimated ages end in .5 |
| sibsp | Number of siblings/spouses aboard | Includes:<br>- Siblings: brother, sister, stepbrother, stepsister<br>- Spouse: husband, wife (excludes mistresses/fiancés) |
| parch | Number of parents/children aboard | Includes:<br>- Parents: mother, father<br>- Children: daughter, son, stepdaughter, stepson<br>Note: Children with nanny only have parch=0 |
| ticket | Ticket number | |
| fare | Passenger fare | |
| cabin | Cabin number | |
| embarked | Port of Embarkation | C = Cherbourg, Q = Queenstown, S = Southampton |
