## Machine Learning Project

My project was to change a basic hyperparameters function to be able to apply to multiple labels from scratch, despite there are different functions that exist that already do it. After setting up the encoder, I fit_transform y and transform yhat, then y is turned into a numpy array and raveled along with yhat. Then the mean absolute value is taken from the ravel y and ravel yhat. I also changed the get_features_and_label_names so that it looks for more than one label. The changes were also added to the UI so that a person could add multiple labels or get the test score of the fits by using --test-score. But I didn’t have enough time to apply a random grid search, so I removed it from the entire project. You will notice that the MAE results for any project using this will be unnaturally high, this is because, as I was building it from scratch and with no external help, it may not be the most proficient at what it is meant to do.

## Getting Started
### 1. Clone the Repository
```
git@github.com:Ethan-Bock/Machine-Learning-Project.git
cd Machine-Learning-Project
```

### 2. Running script
To find out more about how to use the functions of it type ```python3 hyper_parameters.py -h``` into the terminal

![Linkedin-MachineLearningpng](https://github.com/user-attachments/assets/7c344d98-1231-44ea-9564-5965cc99d741)
