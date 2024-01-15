# Disease-Prediction-Using-NLP


![image](https://github.com/Abhishek-AMK/Disease-Prediction-Using-NLP/assets/113782190/41a808c3-203d-4927-af9d-0d9bd18d431c )


## Overview
This project implements a Multi-Label Text Classification model using BERT (Bidirectional Encoder Representations from Transformers) Preprocessor and BERT Encoders. The model predicts the disease that an individual may have based on the symptoms entered in text form.

## Libraries Used
- TensorFlow: An open-source machine learning library providing tools for building and deploying machine learning models.
- TensorFlow Hub: A library for reusable machine learning modules, allowing the sharing and reuse of pre-trained models or model components.
- TensorFlow Text: An extension of TensorFlow providing text processing capabilities, including preprocessing and tokenization for natural language processing (NLP) tasks.
- Pandas: A powerful data manipulation and analysis library for Python, providing data structures like DataFrames for handling and analyzing structured data.
- Scikit-learn Preprocessing: A part of Scikit-learn, providing functions for preprocessing data, such as scaling and encoding.
- JSON: A lightweight data interchange format for encoding and decoding data.
- Warnings: A module used to handle warning messages in Python.

```python
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
import sklearn.preprocessing
import json
import warnings
warnings.filterwarnings('ignore')
```

## Hyperparameters
Define the number of epochs and batch size for model training.

```python
EPOCHS = 8
BATCH_SIZE = 64
```

## Importing and Cleaning Data
Read the CSV file into a Pandas DataFrame and perform data cleaning.

```python
df = pd.read_csv('/path/to/Symptom2Disease.csv')
df.drop('Unnamed: 0', axis='columns', inplace=True)
df['label'] = df['label'].apply(lambda x: x.title())
df.head(2)
```

## One-Hot Encoding Target Labels
Convert the categorical labels into a binary matrix (one-hot encoding).

```python
label_binarizer = sklearn.preprocessing.LabelBinarizer()
df = df.join(pd.DataFrame(label_binarizer.fit_transform(df['label']),
                         columns=label_binarizer.classes_,
                         index=df.index))
df.drop('label', axis='columns', inplace=True)
df.head(2)
```

## Preparing Training and Validation Datasets
Split the data into training and validation sets, convert them to TensorFlow Datasets, and apply batching, caching, and prefetching for faster training.

```python
val_df = df.sample(frac=0.2)
train_df = df.drop(val_df.index)

print(f'Training Dataset Size: {len(train_df)}', f'Validation Dataset Size: {len(val_df)}', sep='\n')

def dataframe_to_tf_dataset(dataframe):
    dataframe = dataframe.copy()
    feature = dataframe.pop('text')
    ds = tf.data.Dataset.from_tensor_slices((feature, dataframe))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

train_ds = dataframe_to_tf_dataset(train_df).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
val_ds = dataframe_to_tf_dataset(val_df).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
```

## Building the Model
Define the layers of the model using the functional API.

```python
text_input = tf.keras.Input(shape=(), name='text', dtype='string')
preprocessor = hub.KerasLayer('/path/to/bert_preprocessor', name='bert_preprocessor')
encoder_inputs = preprocessor(text_input)
encoder = hub.KerasLayer('/path/to/bert_encoder', trainable=True, name='bert_encoder')
outputs = encoder(encoder_inputs)
pooled_output = outputs['pooled_output']
x = tf.keras.layers.Dropout(0.20, name='dropout')(pooled_output)
outputs = tf.keras.layers.Dense(24, activation='softmax', name='output')(x)
model = tf.keras.Model(text_input, outputs, name='disease_prediction_model')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

## Training the Model
Train the model using the training dataset.

```python
model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)
```

## Saving the Model and Classes
Save the model in Keras format and the classes dictionary in JSON format.

```python
model.save('/path/to/model.keras')
classes = label_binarizer.classes_
classes_dict = {i: v for i, v in enumerate(classes)}
with open('/path/to/classes.json', 'w') as file:
    json.dump(classes_dict, file)
```

## Testing the Model
Test the model with custom inputs to check its predictions.

```python
predict('I inadvertently lose weight and have a hard time gaining it back. I use antacids to get rid of the pain and discomfort I experience. It aches so much in my mouth.')
predict('My vision is foggy, and it appears to be growing worse. I feel exhausted and worn out all the time. I also have severe dizziness and lightheadedness on occasion.')
predict('I get wheezing and breathing difficulties, which are asthma symptoms. I frequently have headaches and fever. I am continuously exhausted.')
```

# Graphical User Interface (GUI) 

## Overview
The Graphical User Interface (GUI) is designed to provide a user-friendly way to interact with the trained disease prediction model. Users can input symptoms, and the GUI will display the top predicted diseases based on the model's predictions.

## Prerequisites
Before using the GUI, ensure that you have the necessary dependencies installed. The following dependencies are required:
- TensorFlow
- TensorFlow Hub
- TensorFlow Text
- Pandas
- Scikit-learn
- JSON
- Warnings
- IPython Widgets (Jupyter Notebook)

Install the required dependencies using the following command:
```bash
pip install tensorflow tensorflow-hub tensorflow-text pandas scikit-learn ipywidgets
```

## Loading the Model
The GUI loads the pre-trained disease prediction model saved in the SavedModel format. The model is loaded using TensorFlow's `saved_model.load` function. Ensure that the model file is located in the specified path.

```python
# Load the saved model
loaded_model = tf.saved_model.load('/path/to/model_saved_model')
```

## Defining the Keras Model
The loaded SavedModel is converted into a Keras Sequential model using the `Lambda` layer. The input shape and dtype are adjusted according to the model's input requirements.

```python
# Get the inference function from the model's signatures
inference_function = loaded_model.signatures["serving_default"]

# Define a new Keras model using the loaded function
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(), dtype=tf.string, name='text'),
    tf.keras.layers.Lambda(inference_function, input_shape=(), dtype=tf.float32)
])
```

## Loading Classes
The classes dictionary, mapping class indices to class names, is loaded from the JSON file. This dictionary is used to interpret the model's predictions.

```python
# Load the classes
with open('/path/to/classes.json', 'r') as file:
    classes_dict = json.load(file)
    classes = {int(k): v for k, v in classes_dict.items()}
```

## Creating Widgets
The GUI includes three widgets:
1. Text Input: A textarea for entering symptoms.
2. Predict Button: A button for triggering predictions.
3. Output: An output widget for displaying predictions.

```python
# Text input widget
text_input = widgets.Textarea(
    value='',
    placeholder='Enter your symptoms...',
    description='Symptoms:',
    disabled=False
)

# Button widget
predict_button = widgets.Button(description='Predict')

# Output widget
output = widgets.Output()
```

## Predicting Diseases
The `on_predict_button_click` function is called when the Predict button is clicked. It retrieves the entered symptoms, gets predictions using the `get_prediction` function, and prints the top predictions.

```python
# Function to be called when the button is clicked
def on_predict_button_click(b):
    with output:
        output.clear_output()
        text = text_input.value
        prediction = get_prediction(text)
        print(f"Entered symptoms: {text}")
        print("\nTop Predictions:")
        for pred in prediction:
            print(f"{pred[0]}: {pred[1]:.2f}%")
```

## Getting Predictions
The `get_prediction` function uses the model's `predict` method to obtain predictions for the entered symptoms. It converts predictions to a dictionary, sorts them by probability in descending order, and returns the top three predictions.

```python
# Function to get predictions
def get_prediction(text):
    predictions_dict = model.predict(tf.constant([text]))
    predictions_dict = {classes[i]: predictions_dict['output'][0][i] * 100 for i in range(len(classes))}
    predictions_dict = {k: v for k, v in sorted(predictions_dict.items(), key=lambda item: item[1], reverse=True)}
    top_predictions = list(predictions_dict.items())[:3]
    return top_predictions
```

## Using the GUI
To use the GUI:
1. Run the code cells to load the model, define the Keras model, and create widgets.
2. Enter symptoms in the Text Input widget.
3. Click the Predict button to see the top predicted diseases in the Output widget.

Make sure to replace the placeholder paths in the code with the actual paths to your model and classes files.

Enjoy using the Disease Prediction GUI!
