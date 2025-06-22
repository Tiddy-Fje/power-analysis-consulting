# Power analysis for clinical dietitian

## Project overview  

It is common for practitioners across fields to carry out studies where each participant comes at a cost, and resources are limited. In this context, statistical power analyses provide a principled way to select the number of participants needed for a study to be conclusive. 

In this project, we help a clinical dietitian do a power analysis for their study. The goal of the study is to compare two diets, A and B, for diabetic patients. We take a statistical consulting approach to propose a solution, making sure the explanations are clear and accessible for the dietitian to make an evidence based decision. We also provide ideas to improve the study’s design. 

See report.pdf for a detailed analysis and discussion of the project. It includes a clarification and formalization of the dietician's request, clear and concise explanations of necessary statistical tools, as well as estimates for the number of participants needed in different settings. 

## Running the code 

To manage the `python` packages needed to run the files `conda` was used. The `requirements.yml` file can be used to create the associated environment easily as `conda create --n <env-name> --file <relative-path-to-this-file>` (or using similar non-`conda` commands).

`computations.py` generates the figures and minimal number of subjects per group associated with the analysis. It can be run as `python computations.py` in the command window. 

## Generating the report PDF

To generate a pdf of the report, standard `LaTeX` command line prompts of your local machine apply. 

## Acknowledgements

This project was developed as part of the Applied Statistics course at EPFL. I thank Dr. Linda Mhalla for providing the project statement and initial guidance.

## Note about Git history

This project was initially submitted to a Github classroom repo. This version is a copy of that submission that is posted on my personal Github account. I could unfortunately not recover the original Git history.  