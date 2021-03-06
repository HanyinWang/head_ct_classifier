# head_ct_classifier
Couse project EECS496 Statistical Machine Learning
CT (Computed Topography) is a widely used non-invasive diagnosis approach. Head CT (Computed Topography) is the major radiology imaging diagnositc method for head trauma and corresponding complications. Head trauma including Intraparenchymal hemorrhage, Intraventricular hemorrhage, Subdural hematoma, Extradural hematoma, Subarachnoid hematoma, cranial fracture, midline shift and mass effect.In this project, I used pre-trained and fine tuned image classification models on head CT.

Multiple preprocessing for CT are provided in this study to fit head CT into pretrained pytorch image classification models.
Pre-trained pytorch image classfication models including VGG16, ResNet18, ResNet101, ResNet152, DenseNet121, Densenet161, DenseNe169, DenseNet201 and AlexNet.

## Input & output
Input is a NIfTI file in nii.gz format containing one head CT scan. Output is predicted yes / no for 14 labels, including Intracranial hemorrhage, Intraparenchymal hemorrhage, Intraventricular hemorrhage, Subdural hematoma, Extradural hematoma, Subarachnoid hematoma, bleeding location-left, bleeding location-right, chronic bleeding, cranial fracture, calvarial fracture, other fracture, mass effect and midline shift.

## Modified pretrained framework
All the weights are used from the pretrained pytorch models except for dense layers. Fully connected layers are replaced with untrained fully connected layers of corresponding input and output size.

## Data
Download and unzip the open sourced CQ500 data from [here](http://headctstudy.qure.ai/dataset)

This data includes
```
491 folders with multiple DICOM files in each folder
reads.csv -- labels from 3 readers
```

## Head CT data pre-processing
Code for head CT data pre-processing can be found [here](pre-processing)
Following steps are applied for pre-processing CQ500 data:
* From multiple files of each patient, choose one scan of best quality. Quality is defined by number of slices, slice thickness.
* Convert chosen DICOM file to NIfTI files (.nii.gz).
* Get the affine of each scan and reslice each slice in to 5 mm.
* Generate brain window, bone window and subdural window according to corresponding luminance center and window width.
* Select the 28 slices in the middle.

## Label data pre-processing
Code for label data pre-processing for CQ500 data can be found [here](pre-processing)
The data was labeled by three experience radiologists. When there is discrepancies between each radiologist, the majority vote would be taken as the final label.

## Training
### Pre-trained models
10 pre-trained models are implemented in this project with all the fully connected layers removed and re-trained. The training code for each model can be found as following:

* [VGG16](code/run_pretrained_vgg16.py)
* [AlexNet](code/run_pretrained_alexnet.py)
* [ResNet18](code/run_pretrained_resnet18.py)
* [ResNet34](code/run_pretrained_resnet34.py)
* [ResNet50](code/run_pretrained_resnet50.py)
* [ResNet101](code/run_pretrained_resnet101.py)
* [ResNet152](code/run_pretrained_resnet152.py)
* [DenseNet161](code/run_pretrained_densenet161.py)
* [DenseNet169](code/run_pretrained_densenet169.py)
* [DenseNet201](code/run_pretrained_densenet201.py)

### Fine-tuning codes
All 10 models are also fine-tuned, the training code can be found a following:

* [VGG16](code/run_pretrained_vgg16_fine_tune.py)
* [AlexNet](code/run_pretrained_alexnet_fine_tune.py)
* [ResNet18](code/run_pretrained_resnet18_fine_tune.py)
* [ResNet34](code/run_pretrained_resnet34_fine_tune.py)
* [ResNet50](code/run_pretrained_resnet50_fine_tune.py)
* [ResNet101](code/run_pretrained_resnet101_fine_tune.py)
* [ResNet152](code/run_pretrained_resnet152_fine_tune.py)
* [DenseNet161](code/run_pretrained_densenet161_fine_tune.py)
* [DenseNet169](code/run_pretrained_densenet169_fine_tune.py)
* [DenseNet201](code/run_pretrained_densenet201_fine_tune.py)

### Sample training code
- Clone this repo and enter the repo
```
git clone https://github.com/HanyinWang/head_ct_classifier.git
cd head_ct_classifier
```
- Create a virtual environment and install requirements
```
conda create -n ct-env
conda activate ct-env
pip install -r requirements.txt
```
- Before running any of the training code, modified the *input_dir* and *output_dir* at the beginning of the code to where you put the processed data and where you want the output to be stored. Running fine tuning DenseNet169 as an example:
```
cd code
python3 run_pretrained_densenet169_fine_tune.py
```

## To use a fine-tuned model
Download the docker image
```
docker pull hanyinwang/head_ct_classifier
```
Run the docker image
```
docker run -p 5000:5000 hanyinwang/head_ct_classifier
```
The model should be running at
```
http://0.0.0.0:5000/
```
Open a browser and type the url above. [A sample processed CT scan can be found here](sample_data). Download the pre-processed data and upload according to the direction. Wait a while....(like 5 mins), then the prediction result for 14 critical finding with corresponding score will pop up.

![](img/demo.gif)

## Deliverables
* [Dockerfile](Dockerfile)
* [Docker image](https://hub.docker.com/r/hanyinwang/head_ct_classifier?utm_source=docker4mac_2.2.0.3&utm_medium=repo_open&utm_campaign=referral)
* [Fine tuned model](sample_fine_tuned_model)
* [Sample processed head CT](sample_data)
