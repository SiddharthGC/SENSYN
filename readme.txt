Run the following steps to run the project.
-----------------------------------------------------------------
Use the following steps to generate the following directories if not present
-> Navigate to the project folder in command prompt
-> mkdir data
-> mkdir data\\csv
-> mkdir data\\predictions
-> mkdir data\\tmp
train-set.txt,dev-set.txt,test-set.txt files present in data folder.
----------------------------------------------------------------
1. Import "ProjectFinalSubmission-SENSYN" project into PyCharm
2. Mark the src/ folder as "Sources Root"
3. Execute Task2.py
    - i/p: sentence
    - o/p: Results asked to compute in Task2
4 .Model Training and Evaluation
    4.1 Execute sts_features.py to generate the features(X) and labels(y) for train and dev partitions.
        - These numpy arrays are saved to disk on data/tmp folder.
    4.2 Execute sts_train.py to train a gradient boosting model
        - Train and Dev data is loaded from data/tmp folder
        - model file is saved to data/models/gb-model.sav file.
        - predictions on dev data is written to data/predictions/dev-set-predicted-answers.txt file.
            - They are in a format as expected by the evauation.py script
    4.3 Execute sts_infer.py to test the trained model on test data
        - Predictions are written to data/predictions/test-set-predicted-answers.txt