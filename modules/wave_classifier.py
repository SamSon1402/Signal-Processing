import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib

class WaveClassifier(nn.Module):
    """
    A 1D Convolutional Neural Network for classifying wave signals.
    Designed for tissue type classification in medical imaging applications.
    """
    def __init__(self, input_length=1000):
        super(WaveClassifier, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First convolutional block
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Second convolutional block
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Third convolutional block
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Calculate the size after convolutional layers
        self.fc_input_size = 64 * (input_length // 8)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 4)  # 4 classes: normal, abnormal, tumor, cyst
        )
        
    def forward(self, x):
        # x shape: [batch, input_length]
        # Reshape for 1D convolution: [batch, channels, length]
        x = x.unsqueeze(1)
        
        # Apply convolutional layers
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = self.fc_layers(x)
        
        return x

def train_classifier(model, X_train, y_train, X_val=None, y_val=None, 
                     batch_size=32, epochs=20, learning_rate=0.001):
    """
    Train the wave classifier model.
    
    Parameters:
    -----------
    model : WaveClassifier
        The model to train
    X_train : torch.Tensor
        Training data
    y_train : torch.Tensor
        Training labels
    X_val : torch.Tensor, optional
        Validation data
    y_val : torch.Tensor, optional
        Validation labels
    batch_size : int
        Batch size for training
    epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate for optimization
        
    Returns:
    --------
    model : WaveClassifier
        Trained model
    history : dict
        Training history
    """
    # Create a dataset and dataloader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create validation dataset if provided
    if X_val is not None and y_val is not None:
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Average training loss for this epoch
        epoch_loss = running_loss / len(train_loader)
        history['train_loss'].append(epoch_loss)
        
        # Validation
        if X_val is not None and y_val is not None:
            val_loss, val_accuracy = evaluate_classifier(model, X_val, y_val, criterion)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        else:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}')
    
    return model, history

def evaluate_classifier(model, X_test, y_test, criterion=None):
    """
    Evaluate the classifier on test data.
    
    Parameters:
    -----------
    model : WaveClassifier
        The model to evaluate
    X_test : torch.Tensor
        Test data
    y_test : torch.Tensor
        Test labels
    criterion : torch.nn.Module, optional
        Loss function
        
    Returns:
    --------
    metrics : dict
        Evaluation metrics
    """
    model.eval()
    
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        # Forward pass
        outputs = model(X_test)
        loss = criterion(outputs, y_test).item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == y_test).sum().item()
        accuracy = correct / y_test.size(0)
    
    return {
        'loss': loss,
        'accuracy': accuracy
    }

def save_model(model, model_path):
    """
    Save the trained model.
    
    Parameters:
    -----------
    model : WaveClassifier
        The trained model to save
    model_path : str
        Path where the model will be saved
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def load_model(model_path, input_length=1000):
    """
    Load a trained model.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model
    input_length : int
        Length of the input signal
        
    Returns:
    --------
    model : WaveClassifier
        The loaded model
    """
    model = WaveClassifier(input_length)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Function to train and save a random forest classifier on extracted features
def train_random_forest(X_train, y_train, X_val=None, y_val=None, model_path=None):
    """
    Train a random forest classifier on extracted features.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features
    y_train : numpy.ndarray
        Training labels
    X_val : numpy.ndarray, optional
        Validation features
    y_val : numpy.ndarray, optional
        Validation labels
    model_path : str, optional
        Path to save the model
        
    Returns:
    --------
    model : RandomForestClassifier
        Trained model
    metrics : dict
        Evaluation metrics
    """
    # Create and train the model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate on validation set if provided
    metrics = {}
    if X_val is not None and y_val is not None:
        accuracy = model.score(X_val, y_val)
        metrics['accuracy'] = accuracy
        print(f"Validation accuracy: {accuracy:.4f}")
    
    # Save the model if a path is provided
    if model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
    
    return model, metrics

# Function to train and save an SVM classifier on extracted features
def train_svm(X_train, y_train, X_val=None, y_val=None, model_path=None):
    """
    Train an SVM classifier on extracted features.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features
    y_train : numpy.ndarray
        Training labels
    X_val : numpy.ndarray, optional
        Validation features
    y_val : numpy.ndarray, optional
        Validation labels
    model_path : str, optional
        Path to save the model
        
    Returns:
    --------
    model : SVC
        Trained model
    metrics : dict
        Evaluation metrics
    """
    # Create and train the model
    model = SVC(
        C=1.0,
        kernel='rbf',
        probability=True,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate on validation set if provided
    metrics = {}
    if X_val is not None and y_val is not None:
        accuracy = model.score(X_val, y_val)
        metrics['accuracy'] = accuracy
        print(f"Validation accuracy: {accuracy:.4f}")
    
    # Save the model if a path is provided
    if model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
    
    return model, metrics

def load_pretrained_model(model_type="CNN", input_length=1000):
    """
    Load a pretrained model for inference.
    For this demo, we simulate pretrained models.
    
    Parameters:
    -----------
    model_type : str
        Type of model to load ("CNN", "Random Forest", "SVM")
    input_length : int
        Length of input signals for CNN model
        
    Returns:
    --------
    model : object
        Loaded model
    """
    # Create a directory for pretrained models
    os.makedirs("models/pretrained", exist_ok=True)
    
    if model_type == "CNN":
        # Create a new CNN model
        model = WaveClassifier(input_length)
        
        # Generate synthetic data for training
        from modules.wave_extractor import WaveFeatureExtractor
        
        extractor = WaveFeatureExtractor(sampling_rate=input_length)
        
        X_train = []
        y_train = []
        
        # Generate training examples
        tissue_types = ["normal", "abnormal", "tumor", "cyst"]
        for idx, tissue_type in enumerate(tissue_types):
            for i in range(100):  # 100 examples per class
                abnormality = np.random.uniform(0, 1) if tissue_type != "normal" else 0
                _, signal = extractor.generate_simulated_medical_wave(
                    tissue_type=tissue_type, 
                    abnormality_level=abnormality
                )
                X_train.append(signal)
                y_train.append(idx)
        
        # Convert to tensors
        X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
        y_train = torch.tensor(np.array(y_train), dtype=torch.long)
        
        # Create a dataset and dataloader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Train for a few epochs
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(2):  # Just 2 epochs for demo purposes
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        
        model.eval()
        return model
    
    elif model_type == "Random Forest":
        # Create a random forest classifier
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Here we would normally load a pretrained model
        # For this demo, we'll use a dummy model
        # We would use: model = joblib.load("models/pretrained/rf_model.joblib")
        
        # Instead, simulate a trained model with dummy data
        X_dummy = np.random.rand(400, 50)  # 50 features
        y_dummy = np.random.randint(0, 4, 400)  # 4 classes
        model.fit(X_dummy, y_dummy)
        
        return model
    
    elif model_type == "SVM":
        # Create an SVM classifier
        model = SVC(
            C=1.0,
            kernel='rbf',
            probability=True,
            random_state=42
        )
        
        # Here we would normally load a pretrained model
        # For this demo, we'll use a dummy model
        # We would use: model = joblib.load("models/pretrained/svm_model.joblib")
        
        # Instead, simulate a trained model with dummy data
        X_dummy = np.random.rand(400, 50)  # 50 features
        y_dummy = np.random.randint(0, 4, 400)  # 4 classes
        model.fit(X_dummy, y_dummy)
        
        return model
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")