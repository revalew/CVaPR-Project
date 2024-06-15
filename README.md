# Classification of tumour data and metastasis

CVaPR Project - Computer Vision and Pattern Recognition. Classification of tumour data.

## Summary

For this project I decided to create a CNN - Convolutional Neural Network. The trained classifier will be used to determine whether a patient has "metastasis" (class 0) or "no metastasis" (class 1) based on 105 features.

## Important files

- The main Jupyter Notebook's name: `cnn_classification.ipynb`

- As an experiment, I also tried to create a simple DNN - Dense Neural Network, which can be found in the Notebook called `dense_network_classification.ipynb`.

- The trained models were stored in the `./models/` directory.

- The `cvapr_project_instructions.txt` file contains the rough notes on the details of this project.

- Data used to train the model is located in `./data/` directory:

  - `labels_features.csv` was the only file used, and it contains both features and labels,
  - `features.csv` contains only features,
  - `labels.csv` contains only labels,
  - `clinical_radiomics_imported_from_tsv.xlsx` is the original spreadsheet used to generate CSV files,

  - `./aditional_resources/` folder contains my personal resources, where I wrote the code and comments used to study machine learning (notebook `7. Deep Learning for Computer Vision.ipynb`).
