# DatathonFME 2025 - SmadeX challenge

## Team ShermaTech

In this project we were tasked with training a regressor on app usage and advertisement data of app users, with the objective to predict the revenue that a placed advertisement will provide.

### Exploratory analysis and preprocessing

In the exploratory analysis (files `exploratory-v1.ipynb` and `exploratory-v2.ipynb`) we found that the data needs to be preprocessed heavily before we train our models. Furthermore, the data is separated by parquets due to its large size, which makes using it and processing it more difficult.

This is a recopilation of the problems with the data and the preprocessing proposed:
- In the train test, a lot of rows have no user related data. They are not useful for training models, so they are removed (not in the test set, of course). Otherwise, null values are replaced depending on the meaning of the variable; either by its mean or by its maximum value (for example, the attribute "day_avg_ins" gets replaced with the maximum value of 28 days if it is null).
- A lot of attributes are not normalized (they are either lists or dictionaries of variable size, where keys are usually a type of application).
- Our objective was to normalize the dictionaries by separating them into different attributes (one per unique key). This has some problems because it is possible that not all parquets have the same keys in those dictionaries. Furthermore, the number of unique keys is large and some of them are variations of others (example: "Games/Sports", "Games/Games/Sports", "Sports/Games", etc.). The solution we have implemented is to use a first parquet to read and save keys of these dictionaries; then for the other parquets we use those keys, and if a not-seen-before key appears, we map that value to an "Other" category. We also simplify the categories of apps by picking the first word of their names (example: "Games/Sports" can get simplified to "Games", but "Sports/Games" gets simplified to "Sports").
- For the attributes with lists as values, we apply one-hot encoding (but using the first parquet to get the categories, and using a "other" category for not-seen-before categories).
- Other attributes are also no normalized and contain dictionaries, but the keys the application IDs. We think that application IDs are especially useful since new apps will not be able to be read by our classifier if it is trained on app IDs. If we have access to equivalent variables with more global categorizations (like the previous example of "Games/Sports", "Sports/Games", ...), then we do not use this variable. But if there is no alternative, we use the mean or the sum of each dictionary instead.

All of our preprocessing needs were handled in `preprocessing.ipynb`, where we define a Transformer to fit with the first parquet and transform the next ones. We have two preprocessed train and test sets as well. The second preprocessed trainset has not finished preprocessing because we ran out of time, and we would have liked to use it as it contained more variables with one-hot encodings. Our intention was to create and train a modest neural network with this second trainset but it has not been possible due to time constraints.

### Models

The model we have implemented (done in `model_training.py`) is a XGBoost regressor that is composed of two parts:
- The first one predicts whether the user will spend money on the application that the advertisement advertises.
- The second one predicts, only for those users predicted to spend money, how much money they will spend.

The model performs better than just predicting no revenue, but it does not perform very well because we fitted it with only a small set of variables as we did not have much time.

### Future improvements

If we had more time we would have liked to finish the second preprocessing of the data, which would allow us to fit models with more variables. Then we would have liked to design a modest neural netowrk with which to improve our results found with XGBoost. The advantage of the neural network is that the NN can find relations in the data that we have not found.

An idea we had was to fit different Neural Networks with different parquets each and then join them together as an Ensemble Method classifier. This makes sense as the data is partitioned already.

