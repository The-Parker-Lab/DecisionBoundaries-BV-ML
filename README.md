Welcome to the repository for the journal titled "Machine Learning and Voronoi-Based Decision Boundaries for Bacterial Vaginosis to determine population- specific Microbial Interactions" The following are descriptions of each of the files and what they are used for:

main.py - The main python script. Running this script fully trains and evaluates all 1500 BV classifiers. Each model is scored by balanced accuracy, false positive rate, and false negative rate. The results are compiled and saved into a csv file.

hyperparams.py - Script coding for the "optimizer" function that is utilized in training. This "optimizer" function contains all the code necessary for the tuning of each model's hyperparameters. This includes all things necessary for tuning utilizing Hyperopt (i.e. relevant hyperparameter spaces, loss functions, and evaluation sizes)

bests.py - Script used to explore results output from main.py. This script identifies highest performing model architecture and feature selection method for each population group. These models are then saved using the "pickle" package. The script also generates the boxplots for these models. 

decisionboundary.py - Script used to make voronoi estimations of model decision boundaries. This script utilizes a method described by Migut et al. (2015) that makes variable-separable 2-dimensional decision boundaries from high dimension models.

utils.py - Script contains supporting functions that assist in the operation of the other scripts.

data repository - The data repository contains the normalized 16s rRNA data from the Srinivasan and Ravel studies. This data is used to develop the classifiers discussed in our study.