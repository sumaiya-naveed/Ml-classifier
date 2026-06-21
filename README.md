This Ml classifier is trained on the churned dataset to predict whether the customer will be churned or not based on its diff parameters in the dataset.
the folder structure containes the csv dataset file a main.py and a test file called requirement.txt 
For this classifier the two models are trained called logistic regression and random to predict the customer to be churned or not.
Before training we preprocessed the data by adjusting the wrong datatypes of the columns to perform an error free mathmatical calculations.
We encoded the categorical values using the label encoder and numeric ones by one-hot encoding.
Then the dataset is split into the train and test sections having 80% train data and 20% test data.
After splitting the dataset we did the scaling of dataset using the standard scaler from sk-learn which resolves the outliers and bring all the values in the columns of dataset in the same range.
This sclaing was only done on training data not on the test data so that the model can also work on the values outside the training data.

Both of these models include the classification report and ROC_AUC curve data for their comparison 
Since the random forest model had more accuracy in prediction as compared to logistic regression so we moved forward with it.
We plot the confusion matrix on random forest model that showed the accuracy of the predictions by the model 
Then moving on we choose the best parameters to be predicted on from the dataset by the GridsearchCV.

After trainig of both of these models they are saved as .pkl files using joblib 
then in the code section we created a file called main.py to to wrap the code in the fastapi.
we used the /PREDICT end point of fasapi to get the prediction using the app.post() command.
To run the api code we use the command of uvicorn http://127.0.0.1:8000/docs
On moving to the docs we click n the predict end poin and enter the values of all the required parameters to predict whether the customer will churn or not 

Then we deployed the whole ML model onto the Railway platform.
The public URL for this task is ml-classifier-production.up.railway.app
