# How to train and test the SDFormer

## 1.Confirm the parameters in "./utils/configs.py".

In the Class Config, 'dataset_index' is used to specify the dataset for model training and 'model_ index' is used to select the model. The selected model refers to the model in the list basic_model_names, in particular, 'model_index=0' refers to the use of comparison model for training. The comparison model supports Unet, PSPnet and DeepLabV3. Backbone supports Resnet series models. 'SDFormer64' and 'SDFormer264' are the parameter settings of the SDFormer used in the plate dataset and the sleeper beam dataset respectively.

Specify the path to save the test results by changing 'test_result_save_path'.
Specify the model save path by changing the 'save_model_path'.
Specify the log file saving path by changing the 'train_log'.
Specify the path of the training dataset by changing 'train_path'.
Optional: specify the path of the test dataset by changing 'val_path'.

Note: the above paths only need to be set once, and the program will create a new folder under the specified path to save the results of different models.

## 2. Training model

Run the 'train.py' script in Python to get the trained model. 

Note: the 'train.py' script will only train the model specified in the './utils/config.py' file, so you need to change the './utils/config.py' file again to train different models.

## 3. Testing model

Run the 'test.py' script in Python to get the testing result of the model.

Note: The 'test.py' script will also only test the models specified in the './utils/config.py' file, so you need to change the parameters of the './utils/config.py' file repeatedly to get the test results of all models.

## 4. Noise Testing

Run the 'noisetest.py' script in Python to get the anti-noise test results of all the trained models. 

---------------------

The remaining steps will be updated soon......
