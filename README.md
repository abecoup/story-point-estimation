# Story Point Estimation

This repository aims to provide further information and content relevant to the replication and extension of the following research: [Investigating the Effectiveness of Clustering for Story Point Estimation](https://github.com/SOLAR-group/LHC-SE) by Vali Tawosi et al.

This work serves as my masters capstone research at Rochester Institute of Technology during the Spring 2023 semester.

To install dependencies simply run: `pip install -r requirements.txt`.

# How to train the LDA model:

`python generate_lda_model.py [PATH TO DATASET] [NUMBER OF TOPICS]`

For example: `python generate_lda_model.py .\tawosi_dataset\ 10`

`[PATH TO DATASET]` is required. `[NUMBER OF TOPICS]` is optional but significantly reduces time to completion when provided. If it is not provided, the best t-value will be computed (very time intensive).

The best model produced is ./models/lda_2265.model which can be used to run SP estimations.

# How to run the estimation:

`python run_sp_estimation.py [PATH TO DATASET] [PATH TO LDA MODEL] [PATH TO SAVE RESULTS TO] [Cluster Building Strategy:'MAE', 'MdAE', or 'sil'] [Algorithm Variant: 'LHC-SE' or 'LHC-TC-SE']`

For example: `python .\run_sp_estimation.py .\tawosi_dataset\ .\models\lda_2265.model .\results_new\ MAE LHC-SE`

If `[PATH TO SAVE RESULTS TO]` is not already created then the directory will be created for you.

## Acknowledgements

Advisor: [Zhe Yu](https://zhe-yu.github.io/)

Original Authors:
- [Vali Tawosi](https://vtawosi.github.io/)
- [Afnan Al-Subaihin](https://afnan.ws/)
- [Federica Sarro](http://www0.cs.ucl.ac.uk/staff/F.Sarro/)
