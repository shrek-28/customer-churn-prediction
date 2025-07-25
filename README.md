# Customer Churn Prediction using Boosted Trees

## Introduction 
Customer churn is a major concern in the telecommunications industry, where retaining existing users is far more cost-effective than acquiring new ones. The Telco Customer Churn dataset provides detailed information on customer demographics, account details, service usage, and billing patterns. The goal of this project is to develop a machine learning model that can predict whether a customer is likely to churn (leave the service) or not, based on their historical behavior and service profile.

This classification task helps telecom companies identify at-risk customers and implement targeted retention strategies, improving business performance and customer satisfaction.

## Dataset Features 
| Feature Name       | Description                                                                    |
| ------------------ | ------------------------------------------------------------------------------ |
| `customerID`       | Unique ID assigned to each customer.                                           |
| `gender`           | Gender of the customer (`Male`/`Female`).                                      |
| `SeniorCitizen`    | Whether the customer is a senior citizen (`1`) or not (`0`).                   |
| `Partner`          | Whether the customer has a partner (`Yes`/`No`).                               |
| `Dependents`       | Whether the customer has dependents (`Yes`/`No`).                              |
| `tenure`           | Number of months the customer has stayed with the company.                     |
| `PhoneService`     | Whether the customer has a phone service (`Yes`/`No`).                         |
| `MultipleLines`    | Whether the customer has multiple phone lines (`Yes`/`No`/`No phone service`). |
| `InternetService`  | Type of internet connection (`DSL`, `Fiber optic`, or `No`).                   |
| `OnlineSecurity`   | Whether the customer has online security add-on (`Yes`/`No`/`No internet`).    |
| `OnlineBackup`     | Whether the customer has online backup add-on.                                 |
| `DeviceProtection` | Whether the customer has device protection service.                            |
| `TechSupport`      | Whether the customer has technical support service.                            |
| `StreamingTV`      | Whether the customer streams TV content.                                       |
| `StreamingMovies`  | Whether the customer streams movies.                                           |
| `Contract`         | Type of contract (`Month-to-month`, `One year`, `Two year`).                   |
| `PaperlessBilling` | Whether the customer uses paperless billing (`Yes`/`No`).                      |
| `PaymentMethod`    | Method of payment (e.g., `Electronic check`, `Mailed check`, etc.).            |
| `MonthlyCharges`   | The amount charged to the customer monthly.                                    |
| `TotalCharges`     | The total amount charged to the customer till date.                            |
| `Churn`            | **Target** variable – whether the customer has churned (`Yes`/`No`).           |

## Methodology 
1. Primary analyses of data was done using ```df.describe()```, ```df.info()``` and ```df.head()```, to provide a initial statistical analysis, see an initial view of the data, and different data types.
2. Churn classes were analyzed using a countplot.
3. Bivariate and Trivariate analysis of variable associations with Churn, using violin plots and boxplots.
4. The correlation of each variable with churn was also visualized using a barplot.
5. Tenure was also analyzed in terms of both univariate and bivariate analysis.
6. Different charges, and churn rates were analyzed differentially, and tenures were classified into different cohorts for easier classification.
7. Churn values were analyzed in relationship to different tenure cohorts as well.
8. The data was separated on a 90:10 ratio for training and testing.
9. ```GridSearchCV``` was used to train different hyperparameter combinations for a decision tree classifier.
10. Model was evaluated in terms of confusion matrix and classification report.

## Results 
* The model achieved an overall classification accuracy of 80%, meaning that 8 out of 10 customer outcomes were correctly predicted by the model. While accuracy gives a general sense of performance, it is not sufficient on its own in the presence of class imbalance.
* The model performed strongly in identifying customers who did not churn. With a precision of 0.86, a recall of 0.89, and an F1-score of 0.88, it demonstrated high reliability in predicting stable customers. These metrics indicate the model’s ability to correctly retain and trust the "No Churn" predictions.
* In contrast, the model struggled with identifying customers who were likely to churn. It achieved a precision of only 0.52, meaning just over half of the customers predicted to churn actually did so. The recall was even lower at 0.44, suggesting that the model successfully identified fewer than half of the actual churners. The F1-score for this class, a balance between precision and recall, was 0.47, indicating suboptimal performance in detecting churn.
* According to the confusion matrix, 64 churners were correctly identified (true positives), but 83 actual churners were incorrectly classified as non-churners (false negatives). Conversely, 498 non-churners were correctly identified, while 59 were incorrectly labeled as churners (false positives). This highlights a significant imbalance in error types, with the model being more prone to missing churners than falsely flagging loyal customers.
* The macro-averaged metrics — precision (0.69), recall (0.66), and F1-score (0.67) — suggest moderate balanced performance across both classes, but these are artificially boosted by the dominance of the majority class. The weighted averages are similar to the overall accuracy (around 0.79), again skewed in favor of the non-churn class.
* The model’s primary weakness lies in its inability to effectively detect customers at risk of churn, which limits its utility in proactive retention strategies. From a business standpoint, missing a churner is costlier than falsely targeting a non-churner, making this a critical issue to address.
