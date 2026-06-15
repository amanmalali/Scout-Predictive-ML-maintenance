from abc import ABC, abstractmethod
import numpy as np

class ScoutTask(ABC):
    """
    The interface that any user-defined task must implement to work 
    with the Scout Pipeline.
    """

    @abstractmethod
    def get_data(self):
        """
        Returns a dictionary containing the dataset splits.
        """
        pass

    @abstractmethod
    def build_model(self):
        """
        Returns a fresh instance of the base model.
        """
        pass

    @abstractmethod
    def train_base_model(self, model, data, config):
        """
        User-defined training loop.
        """
        pass

    @abstractmethod
    def inference(self, model, x_data):
        """
        Standardized inference wrapper.
        """
        pass

    @abstractmethod
    def calculate_loss(self, y_pred, y_target):
        """
        Calculates the task-specific scalar loss metric (e.g., MSE)
        for reporting purposes.
        """
        pass
    
    @abstractmethod
    def predict_with_loss(self, model, x, y):
        """
        Runs inference and returns predictions AND per-sample losses.
        
        Args:
            model: Trained model
            x: Input features
            y: Ground truth targets
            
        Returns:
            preds: np.array of predictions
            losses: np.array of loss values per sample (same length as preds).
                    Do NOT return a scalar here.
        """
        pass

    def custom_pipeline_step(self, context):
        """
        Optional Hook for custom logic.
        """
        pass