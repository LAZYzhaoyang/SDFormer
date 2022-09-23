# How to Generate the dataset
Two datasets, plate and sleeper beam, are used in this paper, and the generation steps of these two datasets are the same. Here, we will take the plate dataset as an example to introduce the steps of generating the dataset.

1. Confirm the parameters in "./data/Abaqus_ModelBase/plate/config.py".
2. Running "gen_mask.py" using Python.
3. Open ABAQUS, lock the working directory into "./data/Abaqus_ModelBase/plate" and use 'Run_Script'command to run './data/Abaqus_ModelBase/plate/get_abaqus_data.py'.
4. Wait for ABAQUS to complete the data simulation, and then run './data/Abaqus_ModelBase/plate/gen_data.py' using Python.
