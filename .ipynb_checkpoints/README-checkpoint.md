# Capstone Project: Inventory Monitoring Using Object Count Estimation from Bin Images

## Project Overview

This project leverages AWS SageMaker to train a pre-trained model for object count estimation in bin images. The solution utilizes SageMaker's hyperparameter tuning, debugger, and other ML improvement tools to create a complete machine learning pipeline for inventory monitoring applications.

## Dataset

### Overview

The dataset is sourced from the [Amazon Bin Image Dataset](https://registry.opendata.aws/amazon-bin-imagery/), using a subset based on the `file_list.json` file. This dataset contains bin images with varying numbers of objects, making it ideal for training object counting models.

### Data Access and Preparation

The data is accessed and downloaded from AWS S3 using the following approach:

```python
def download_and_arrange_data():
    s3_client = boto3.client('s3')

    with open('file_list.json', 'r') as f:
        d = json.load(f)

    for k, v in d.items():
        print(f"Downloading Images with {k} objects")
        directory = os.path.join('train_data', k)
        if not os.path.exists(directory):
            os.makedirs(directory)
        for file_path in tqdm(v):
            file_name = os.path.basename(file_path).split('.')[0] + '.jpg'
            s3_client.download_file('aft-vbi-pds', os.path.join('bin-images', file_name),
                                  os.path.join(directory, file_name))

download_and_arrange_data()
```

### Dataset Statistics

- **Total files downloaded**: 10,443 JPG images
- **Distribution by object count**:
  - 1 object: 1,228 files
  - 2 objects: 2,299 files  
  - 3 objects: 2,666 files
  - 4 objects: 2,372 files
  - 5 objects: 1,875 files

### Data Splitting

The dataset was split into:
- **Training set**: 75%
- **Test set**: 15%
- **Validation set**: 15%

Data upload to AWS S3:
```bash
aws s3 cp Inventory_Image s3://capstone2025inventory/ --recursive --quiet
```

## Model Training

### Model Selection

**ResNet50** was chosen as the pre-trained model due to its proven effectiveness in computer vision tasks and strong performance on image classification problems.

### Training Progress

#### Initial Training
- **Parameters**: Learning rate = 0.001, Batch size = 32
- **Test Accuracy**: 29.3%
- **Results**: Only one class (class 3) showed majority predictions falling in the correct class
- **Confusion Matrix**: ![Initial Results](confusion_matrix1.png)

#### Refined Training
After analyzing the outputs and making adjustments (detailed in `sagemaker.ipynb`):
- **Test Accuracy**: 37.9% (improved from 29.3%)
- **Results**: Two classes now show majority predictions in correct classes
- **Confusion Matrix**: ![Refined Results](confusion_matrix2.png)

#### Hyperparameter Tuning
The following hyperparameters were tuned to optimize model performance:

```python
hyperparameter_ranges = {
    "lr": ContinuousParameter(0.00005, 0.002),  
    "batch-size": CategoricalParameter([32, 64, 128]),
    "weight-decay": ContinuousParameter(0.0001, 0.01),  
    "dropout-rate": ContinuousParameter(0.3, 0.6)  
}
```

**Best hyperparameters found**:
```python
hyperparameters = {
    'batch-size': '32',
    'dropout-rate': '0.3690470843213408',
    'lr': '0.00019386339021814795',
    'weight-decay': '0.00030516592316427005'
}
```

#### Final Model Performance
- **Test Accuracy**: 39.5% (improved from 37.9%)
- **Results**: Three classes now show majority predictions in correct classes
- **Confusion Matrix**: ![Final Results](confusion_matrix3.png)
- **Hyperparameter Tuning Results**: ![Tuning Results](hyperparameter_tuning.png)

### Model Profiling

The profiler report for the final model can be found here: [Profiler Report](./ProfilerReport/profiler-output/profiler-report.html)

## Machine Learning Pipeline

The complete ML pipeline follows these steps:

1. **Initial Training**: Train the model with random parameters to establish baseline performance
2. **Analysis & Refinement**: Analyze outputs to identify potential issues and retrain with fixes
3. **Hyperparameter Tuning**: Use SageMaker's hyperparameter tuning to find optimal parameters
4. **Model Deployment**: Deploy the trained model to an endpoint for inference

### Deployment Configuration

The deployment pipeline includes:
- **Custom ImagePredictor class** for handling JPEG inputs and JSON outputs
- **PyTorch model deployment** to a single `ml.m5.large` instance endpoint
- **Inference script** (`inference.py`) with:
  - Model loading from saved `.pth` file
  - JPEG image preprocessing with standard normalization
  - Model prediction with `torch.no_grad()`
  - JSON response formatting

### Endpoint Usage

To query the deployed endpoint:
!()[endpoint.png]
```python
from sagemaker.predictor import Predictor

predictor = ImagePredictor(
    endpoint_name='pytorch-inference-2025-07-10-04-19-01-577',
    sagemaker_session=sagemaker.Session()
)

image_name = 'path_to_your_image.jpg'  # Specify your image path
with open(image_name, "rb") as image:
    f = image.read()
    img_bytes = bytearray(f)

response = predictor.predict(img_bytes, initial_args={"ContentType": "image/jpeg"})
print(response)  # Returns predicted class and probability
```

## Results Summary

The project successfully demonstrated progressive improvement in model performance:
- **Baseline**: 29.3% accuracy
- **After refinement**: 37.9% accuracy  
- **After hyperparameter tuning**: 39.5% accuracy

The final model shows reasonable performance for object counting across multiple classes, with successful deployment to a SageMaker endpoint for real-time inference.

## Files and Resources

- `sagemaker.ipynb` - Main notebook with detailed implementation
- `file_list.json` - Dataset file list
- `inference.py` - Deployment inference script
- `ProfilerReport/` - Model profiling results
- Confusion matrices and tuning results (PNG files)