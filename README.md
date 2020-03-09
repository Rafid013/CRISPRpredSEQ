# CRISPRpred(SEQ)
## Dependencies
Download anaconda for python 3.7 version from this [link](https://www.anaconda.com/distribution/#download-section) and install anaconda3. Open a new conda environment (using the command "conda create --name myenv", replace myenv with the preferred environment name). Then activate the environment (using the command "conda activate myenv", replace myenv with the preferred environment name) and install the dependencies (run the script dependencies.sh (for linux) or dependencies.ps1 (for windows)) or install them separately-
* pandas 0.24.2 or above
* numpy 1.16.2 or above
* scikit-learn 0.20.3 or above
* pytables 3.5.1 or above
## Running the Experiments
Run the python files in the following order-
1. split_into_folds.py
2. generate_features_folds.py
3. generate_feature_folds_without_hek293_loco.py

You only have to run the codes mentioned above once unless any of the generated files are deleted (No need to run these 3 codes before each experiment)

To reproduce the results of experimental setup A, B or C run the following files-
1. experimentA.py or experimentB.py or experimentC.py (depending on the experimental result you want to reproduce)
2. process_cv_logs.py (this is for experiments B and C only, enter B or C when prompted, run this file to get the detailed result of cross validation, the produced result will be found in Results/gridsearch_roc_without_gapped_result.csv)
3. test_experiments_AC.py (enter A, B or C when prompted)
4. calculate_avg_roc_AC.py (enter A, B or C when prompted)

To reproduce the results of experimental setup D, E or F run the following files-
1. experimentD.py or experimentE.py or experimentF.py
2. test_experiment_without_hek293_DF.py (enter D, E or F when prompted)
3. calculate_avg_roc_without_hek293_DF.py (enter D, E or F when prompted)

To reproduce the results of experiments on DeepHF data-
1. Install thudersvm for gpu
2. Run generate_features_deephf.py
3. Run deephf_A.py or deephf_B.py or deephf_C.py
