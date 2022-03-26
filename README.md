# SDFormer
Implementation of "SDFormer: A Novel Transformer Neural Network for Structural Damage Identification by Segmenting The Strain Field Map".

This paper has been published in "Sensors".

DOI: 10.3390/s22062358

# Abstract
Damage identification is a key problem in the field of structural health monitoring, which is of great significance to improve the reliability and safety of engineering structures. In the past, the structural strain damage identification method based on specific damage index needs the designer to have rich experience and background knowledge, and the designed damage index is hard to apply to different structures. In this paper, a U-shaped efficient structural strain damage identification network SDFormer (structural damage transformer) based on self-attention feature is proposed. SDFormer regards the problem of structural strain damage identification as an image segmentation problem, and introduces advanced image segmentation technology for structural damage identification. This network takes the strain field map of the structure as the input, and then outputs the predicted damage location and level. In the SDFormer, the low-level and high-level features are smoothly fused by skip connection, and the self-attention module is used to obtain damage feature information, to effectively improve the performance of the model. SDFormer can directly construct the mapping between strain field map and damage distribution without complex damage index design. While ensuring the accuracy, it improves the identification efficiency. The effectiveness and accuracy of the model are verified by numerical experiments, and the performance of an advanced convolutional neural network is compared. The results show that SDFormer has better performance than the advanced convolutional neural network. Further, an anti-noise experiment is designed to verify the anti-noise and robustness of the model. The anti-noise performance of SDFormer is better than that of the comparison model in the anti-noise experimental results, which proves that the model has good anti-noise and robustness.

# Citation

@article{Li_2022, doi = {10.3390/s22062358}, url = {https://doi.org/10.3390%2Fs22062358}, year = 2022, month = {mar}, publisher = {{MDPI} {AG}}, volume = {22}, number = {6}, pages = {2358}, author = {Zhaoyang Li and Ping Xu and Jie Xing and Chengxing Yang}, title = {{SDFormer}: A Novel Transformer Neural Network for Structural Damage Identification by Segmenting the Strain Field Map}, journal = {Sensors}}

--------------------

# Log
## 2022/03/19

All codes of this paper have been uploaded. 

Updated instructions on how to generate datasets. See 'HowToGenerateTheDataset.md' for details. 

Updated instructions on how to train and test the model. See 'HowToTrainAndTestModel.md' for details. ('HowToTrainAndTestModel.md' is still being updated.)

## 2022/03/04

The demo code GenMaskDemo.py for generating random damage areas was uploaded. 

## 2022/02/14

The code of this paper will be released soon.
