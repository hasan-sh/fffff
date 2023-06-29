# Installation

- pip install -m requirements.txt
- Spacy requires the following:
    - python -m spacy download en_core_web_sm
- Download the main dataset [here](https://drive.google.com/file/d/1_cvSYyxEdfRVTXXihm7TBWEomn7-WOut/view?usp=sharing). Replace it in the `data` directory.
- Finally, you can run `jupyter lab` in order to view the code.

# Data exploration
**NOTE**: make sure to run all scripts (i.e., python files) from their specific path in the terminal.
- First, you need to run the `retrieve_OCM_labels.py` script. It will create two json files that are needed for data exploration.
- The `Data Exploration & Processing` notebook is ready to be executed. This may take some time, since we validate the the rows' language is English. You don't need to run it, since I ran all cells and saved the `en_data.csv` in the `data` directory.
    - In case you want to run it, please make sure to download the [fpsc3](https://drive.google.com/file/d/1czKWGnhS29cvHFb2onZQz7e5S6zRqO57/view) dataset and place it in the `data` directory.
- The `Chosen Categories` and `Cultures Distribution` notebooks explore the categories that would potentiall be chosen in my thesis, and the cultures within the eHRAF database given these categories.
- All images will be saved within the exploration directory.


# Training
**NOTE**: both models are executed and the results are the same as listed in my paper. So, no need to run them and wait for training, however, feel free to do so.
- The `Model 112 (Training102)` notebook contains all needed code to train the models. The dataset specified is the `data/en_data.csv`.
    - The results will be saved as pickle files.
- In the `Model 113 (Analysis, hidden cues, text)` notebook, I analyze the results of the model by reading the results saved during training.
