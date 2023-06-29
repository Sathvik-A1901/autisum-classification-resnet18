# Autism Classification using ResNet-18

This repository contains the code and resources for an Autism Classification project utilizing transfer learning with ResNet-18. The goal of this project is to develop a machine learning model that can accurately predict autism based on given input data.

## Dataset
The dataset used for this project consists of images categorized into two classes: autism-positive and autism-negative. The dataset was collected from various sources and manually labeled to ensure accuracy. The images are preprocessed and formatted to be compatible with the ResNet-18 architecture.

## Model Architecture
The ResNet-18 architecture is employed for this project, which is a deep convolutional neural network widely used for image classification tasks. The pre-trained weights of ResNet-18, trained on the ImageNet dataset, are utilized as a starting point for transfer learning. The model is fine-tuned on the autism classification dataset to adapt to the specific task.

## Usage
To use the code in this repository, follow these steps:

1. Clone the repository:

```
git clone https://github.com/Sathvik-A1901/autism-classification-resnet18.git
```

2. Install the necessary dependencies:

```
pip install -r requirements.txt
```

3. Prepare the dataset:
   - Place the autism classification dataset in the `data` directory.
   - Ensure the dataset is organized into the appropriate class folders (autism-positive and autism-negative).

4. Train the model:
   - Run the `autismcnn.ipynb` script to train the model.
   - Adjust the hyperparameters as needed.

5. Evaluate the model:
   - Run the `autismcnn.ipynb` script to evaluate the trained model on test data.
   - View the classification metrics and accuracy.

6. Predict using the model:
   - Utilize the `predictions.py` script to make predictions on new, unseen data.
   - Pass the path to the input image as a command-line argument.

## Results
The trained model achieves impressive accuracy and performance on the autism classification task. The README file will be updated with more details on the model's performance as soon as the results are available.

## Contributions
Contributions to this project are welcome. If you discover any bugs or have suggestions for improvement, please feel free to submit a pull request or open an issue.


