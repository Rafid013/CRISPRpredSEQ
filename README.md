# CRISPRpred(SEQ)
## Dependencies
Install anaconda3, open a new conda environment and then install the dependencies.
* python 3.6 or above
* scikit-learn 0.20.3 or above
* pandas 0.24.2 or above
* numpy 1.16.2 or above
## Running the Experiments
Run the python files in the following order-
1. split_into_folds.py
2. generate_features_folds.py
3. append_folds.py
You only have to run the codes mentioned above once unless any of the generated files are deleted (No need to run these 3 codes before each experiment)

To reproduce the results of experimental setup A run the following files-
1. experimentA.py
2. test_experiments.py (enter A when prompted)
3. calculate_avg_roc.py (enter A when prompted)

To reproduce the results of experimental setup B run the following files-
1. experimentB.py
2. process_cv_logs.py (enter B when prompted, run this file to get the detailed result of cross validation, the produced result will be found in Results/gridsearch_roc_without_gapped_result.csv)
3. test_experiments.py (enter B when prompted)
4. calculate_avg_roc.py (enter B when prompted)

To reproduce the results of experimental setup C run the following files-
1. experimentC.py
2. process_cv_logs.py (enter C when prompted, run this file to get the detailed result of cross validation, the produced result will be found in Results/gridsearch_roc_result.csv)
3. test_experiments.py (enter C when prompted)
4. calculate_avg_roc.py (enter C when prompted)
