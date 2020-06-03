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

## Using Pretrained Models for Prediction
Use a csv file to provide the sgRNA sequences as input. Follow the format given in example1.csv. The sgRNA sequence must be 20-mer with NGG PAM sequence at the end resulting in a 23-mer sequence. For prediction-
1. Use the command- **python generate_features.py \<filepath\>**. This will generate the features for the given sequences. For example, if the path to your csv file is **Data/example.csv**, use the command **python generate_features.py Data/example.csv**.
2. To predict using pretrained versions of model A, B or C, run the command **python pretrained_N.py \<filepath\>** (N can be A, B or C in the command). For example, run **python pretrained_A.py \<filepath\>** if you want to predict using model A. The **\<filepath\>** must be provided in this case (this is the **\<filepath\>** to the original input csv file).

## Training Models Using New Data
Use a csv file to provide the sgRNA sequences and labels as input. Follow the format given in example2.csv. The sgRNA sequence must be 20-mer with NGG PAM sequence at the end resulting in a 23-mer sequence. Follow the steps below-
1. Use the command- **python generate_features.py \<filepath\>**. This will generate the features for the given sequences. For example, if the path to your csv file is **Data/example.csv**, use the command **python generate_features.py Data/example.csv**.
2. To train your own versions of model A, B or C using new data, run the command **python train_N.py \<filepath\>** (N can be A, B or C in the command). For example, run **python train_B.py \<filepath\>** if you want to train model B. The **\<filepath\>** must be provided in this case (this is the **\<filepath\>** to the original input csv file).
3. To predict using your own versions of trained model A, B or C, run the command **python predict_N.py \<filepath\>** (N can be A, B or C in the command). For example, run **python predict_B.py \<filepath\>** if you want to predict using the model B you just trained. The **\<filepath\>** must be provided in this case (this is the **\<filepath\>** to the original input csv file).

## Citation
Muhammad Rafid, A.H., Toufikuzzaman, M., Rahman, M.S. et al. CRISPRpred(SEQ): a sequence-based method for sgRNA on target activity prediction using traditional machine learning. BMC Bioinformatics 21, 223 (2020). https://doi.org/10.1186/s12859-020-3531-9

```
﻿@Article{MuhammadRafid2020,
    author={Muhammad Rafid, Ali Haisam
      and Toufikuzzaman, Md.
      and Rahman, Mohammad Saifur
      and Rahman, M. Sohel},
    title={CRISPRpred(SEQ): a sequence-based method for sgRNA on target activity prediction
      using traditional machine learning},
    journal={BMC Bioinformatics},
    year={2020},
    month={Jun},
    day={01},
    volume={21},
    number={1},
    pages={223},
    issn={1471-2105},
    doi={10.1186/s12859-020-3531-9},
    url={https://doi.org/10.1186/s12859-020-3531-9}
 }
```


