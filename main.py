import numpy as np
import pandas as pd
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Function to load datasets
def load_datasets(training_file, new_data_file):
    training_data = pd.read_csv(training_file, delimiter="\t")
    new_data = pd.read_csv(new_data_file, delimiter="\t")
    return training_data, new_data

# Function to preprocess the data
def preprocess_data(training_data, new_data):
    X = training_data.iloc[:, :-1].values
    y = training_data.iloc[:, -1].values
    X_new = new_data.values
    
    # Define preprocessing for numerical features
    numerical_features = slice(0, 9)  # Columns 0 to 8
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant')),
        ('scaler', StandardScaler())])
    
    # Define preprocessing for categorical features
    categorical_features = slice(9, 12)  # Columns 9 to 11
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])
    
    # Preprocessing data
    X = preprocessor.fit_transform(X)
    X_new = preprocessor.transform(X_new)
    
    return X, y, X_new

# Function to create neural network models
def create_models(input_dim):
    models = []

    # Model 1: Simple model with different activation function
    model1 = Sequential()
    model1.add(Dense(12, input_dim=input_dim, activation='tanh'))
    model1.add(Dense(8, activation='tanh'))
    model1.add(Dense(1, activation='sigmoid'))
    model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    models.append(model1)

    # Model 2: Adding Dropout and different learning rate
    model2 = Sequential()
    model2.add(Dense(12, input_dim=input_dim, activation='relu'))
    model2.add(Dropout(0.2))
    model2.add(Dense(8, activation='relu'))
    model2.add(Dropout(0.2))
    model2.add(Dense(1, activation='sigmoid'))
    model2.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
    models.append(model2)

    # Model 3: Adding Batch Normalization
    model3 = Sequential()
    model3.add(Dense(24, input_dim=input_dim, activation='relu'))
    model3.add(BatchNormalization())
    model3.add(Dense(12, activation='relu'))
    model3.add(BatchNormalization())
    model3.add(Dense(1, activation='sigmoid'))
    model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    models.append(model3)

    # Model 4: Improved model with ReLU activation, Batch Normalization, and Dropout
    model4 = Sequential()
    model4.add(Dense(16, input_dim=input_dim, activation='relu'))
    model4.add(BatchNormalization())
    model4.add(Dropout(0.3))
    model4.add(Dense(8, activation='relu'))
    model4.add(BatchNormalization())
    model4.add(Dropout(0.3))
    model4.add(Dense(1, activation='sigmoid'))
    model4.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    models.append(model4)

    # Model 5: More neurons, dropout, and different learning rate
    model5 = Sequential()
    model5.add(Dense(30, input_dim=input_dim, activation='relu'))
    model5.add(Dropout(0.3))
    model5.add(Dense(15, activation='relu'))
    model5.add(Dropout(0.3))
    model5.add(Dense(1, activation='sigmoid'))
    model5.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.005), metrics=['accuracy'])
    models.append(model5)

    return models

# Function to train models
def train_models(models, X_train, y_train, X_val, y_val, epochs=150, batch_size=10):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    histories = []
    
    for model in models:
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                            validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)
        histories.append(history)
    
    return histories

# Function to save a model
def save_model(model, model_filename_prefix):
    # Save model structure
    structure_filename = model_filename_prefix + ".json"
    model_json = model.to_json()
    with open(structure_filename, "w") as f:
        f.write(model_json)
    
    # Save model weights
    weight_filename = model_filename_prefix + ".weights.h5"
    model.save_weights(weight_filename)

# Function to load a model
def load_model(model_filename_prefix):
    # Load model structure
    structure_filename = model_filename_prefix + ".json"
    with open(structure_filename, "r") as f:
        model_json = f.read()
    model = model_from_json(model_json)
    
    # Load model weights
    weight_filename = model_filename_prefix + ".weights.h5"
    model.load_weights(weight_filename)
    
    return model

# Function to save predictions
def save_predictions(predictions, filename):
    np.savetxt(filename, predictions, delimiter=',', fmt='%d')

# Function to predict the target attribute of a new dataset based on a model
def predict_new_dataset_from_model(new_input_attribute_data, model):
    predictions = model.predict(new_input_attribute_data)
    # Convert probabilities to binary predictions (0 or 1)
    binary_predictions = (predictions > 0.5).astype(int)
    return binary_predictions

# Function to evaluate the model
def evaluate_model(model, X_val, y_val):
    y_pred_prob = model.predict(X_val)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    f1 = f1_score(y_val, y_pred)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return accuracy, precision, recall, f1

# Function to plot evaluation metrics
def plot_evaluation_metrics(metrics, model_names):
    metrics_df = pd.DataFrame(metrics, columns=['Accuracy', 'Precision', 'Recall', 'F1 Score'], index=model_names)
    metrics_df.plot(kind='bar', figsize=(12, 8))
    plt.title('Evaluation Metrics for Different Models')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# Function to plot training history
def plot_training_history(histories, model_names):
    plt.figure(figsize=(14, 8))
    
    for i, history in enumerate(histories):
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label=f'Model {i+1} Train Loss')
        plt.plot(history.history['val_loss'], label=f'Model {i+1} Val Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label=f'Model {i+1} Train Accuracy')
        plt.plot(history.history['val_accuracy'], label=f'Model {i+1} Val Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

# Main function
def main():
    training_file = 'firstData.txt'
    new_data_file = 'secondData.txt'
    model_filename_prefix = "neuralNetworkModel"
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    training_data, new_data = load_datasets(training_file, new_data_file)
    X, y, X_new = preprocess_data(training_data, new_data)
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create models
    print("Creating models...")
    models = create_models(X_train.shape[1])
    
    # Train models
    print("Training models...")
    histories = train_models(models, X_train, y_train, X_val, y_val)
    
    # Evaluate models and collect metrics
    metrics = []
    model_names = []
    
    for i, model in enumerate(models):
        print(f"Evaluating model {i+1}...")
        accuracy, precision, recall, f1 = evaluate_model(model, X_val, y_val)
        metrics.append([accuracy, precision, recall, f1])
        model_names.append(f"Model {i+1}")
        
        # Save the model
        model_prefix = f"{model_filename_prefix}_{i+1}"
        print(f"Saving model {i+1}...")
        save_model(model, model_prefix)
        
        # Load the model to ensure saving and loading works
        print(f"Loading model {i+1}...")
        loaded_model = load_model(model_prefix)
        
        # Predict new data
        print(f"Predicting with model {i+1}...")
        predictions = predict_new_dataset_from_model(X_new, loaded_model)
        
        # Save predictions
        prediction_filename = f'predicted{i+1}.txt'
        print(f"Saving predictions to {prediction_filename}...")
        save_predictions(predictions, prediction_filename)
    
    # Plot evaluation metrics
    plot_evaluation_metrics(metrics, model_names)
    
    # Plot training history
    plot_training_history(histories, model_names)

if __name__ == '__main__':
    main()
