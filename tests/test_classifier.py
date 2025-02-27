import unittest
import numpy as np
import torch
import sys
import os

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.wave_classifier import WaveClassifier, train_classifier, evaluate_classifier

class TestWaveClassifier(unittest.TestCase):
    
    def setUp(self):
        # Create a small model for testing
        self.input_length = 100
        self.model = WaveClassifier(input_length=self.input_length)
        
        # Create dummy data for testing
        self.batch_size = 4
        self.dummy_inputs = torch.randn(self.batch_size, self.input_length)
        self.dummy_labels = torch.randint(0, 4, (self.batch_size,))
        
    def test_model_initialization(self):
        # Test that the model initializes correctly
        self.assertIsInstance(self.model, WaveClassifier)
        
        # Test that model has expected layers
        self.assertTrue(hasattr(self.model, 'conv_layers'))
        self.assertTrue(hasattr(self.model, 'fc_layers'))
        
    def test_forward_pass(self):
        # Test the forward pass works and returns expected shape
        outputs = self.model(self.dummy_inputs)
        
        # Check output shape: [batch_size, num_classes]
        self.assertEqual(outputs.shape, (self.batch_size, 4))
        
    def test_model_training(self):
        # Create dummy dataset
        X_train = torch.randn(20, self.input_length)
        y_train = torch.randint(0, 4, (20,))
        X_val = torch.randn(5, self.input_length)
        y_val = torch.randint(0, 4, (5,))
        
        # Test training function
        trained_model, history = train_classifier(
            self.model, 
            X_train, 
            y_train, 
            X_val, 
            y_val, 
            batch_size=2, 
            epochs=2,
            learning_rate=0.01
        )
        
        # Check that history contains expected metrics
        self.assertTrue('train_loss' in history)
        self.assertTrue('val_loss' in history)
        self.assertTrue('val_accuracy' in history)
        
        # Check that training loss decreased
        self.assertLessEqual(history['train_loss'][-1], history['train_loss'][0])
        
    def test_model_evaluation(self):
        # Create dummy test data
        X_test = torch.randn(10, self.input_length)
        y_test = torch.randint(0, 4, (10,))
        
        # Evaluate model
        metrics = evaluate_classifier(self.model, X_test, y_test)
        
        # Check that metrics include accuracy and loss
        self.assertTrue('accuracy' in metrics)
        self.assertTrue('loss' in metrics)
        
        # Check that accuracy is a reasonable value
        self.assertGreaterEqual(metrics['accuracy'], 0.0)
        self.assertLessEqual(metrics['accuracy'], 1.0)
        
    def test_model_prediction(self):
        # Get a single input
        single_input = torch.randn(1, self.input_length)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(single_input)
            probs = torch.nn.functional.softmax(outputs, dim=1)
        
        # Check that probabilities sum to 1
        self.assertAlmostEqual(probs.sum().item(), 1.0, places=5)
        
        # Check that all probabilities are between 0 and 1
        self.assertTrue(all((probs >= 0) & (probs <= 1)).item())
    
    def test_model_save_load(self):
        # Test saving and loading the model
        model_path = "test_model.pth"
        
        # Save model state
        torch.save(self.model.state_dict(), model_path)
        
        # Create a new model and load the state
        new_model = WaveClassifier(input_length=self.input_length)
        new_model.load_state_dict(torch.load(model_path))
        
        # Ensure the models produce the same outputs
        self.model.eval()
        new_model.eval()
        
        with torch.no_grad():
            orig_output = self.model(self.dummy_inputs)
            new_output = new_model(self.dummy_inputs)
        
        # Check that outputs are identical
        torch.testing.assert_close(orig_output, new_output)
        
        # Clean up
        os.remove(model_path)

if __name__ == '__main__':
    unittest.main()