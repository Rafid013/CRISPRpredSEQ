# CRISPRpred(SEQ)
## Dependencies
Download anaconda for python 3.7 version from this [link](https://www.anaconda.com/distribution/#download-section) and install anaconda3. Open a new conda environment (using the command "conda create --name myenv", replace myenv with the preferred environment name). Then activate the environment (using the command "conda activate myenv", replace myenv with the preferred environment name) and install the dependencies-
* pandas 0.24.2 or above
* numpy 1.16.2 or above
* scikit-learn 0.20.3 or above
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
