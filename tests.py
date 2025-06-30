"""
Description: Test cases for the Pytorch implementation of the topic classifier.
"""

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_data_loading():
    try:
        import pandas as pd
        df = pd.read_csv("data_topic/train.csv")
        assert not df.empty, "Train CSV is empty."
        logger.info("Data loading test passed.")
    except Exception as e:
        logger.error(f"Data loading test failed: {e}")

def test_model_shapes():
    try:
        import torch
        from model.architecture import TopicClassifier

        input_size = 10
        hidden_size = 16
        num_classes = 3
        dummy_input = torch.randn(5, input_size)

        model = TopicClassifier(input_size, hidden_size, num_classes)
        output = model(dummy_input)

        assert output.shape == (5, num_classes), "Incorrect output shape."
        logger.info("Model shape test passed.")
    except Exception as e:
        logger.error(f"Model shape test failed: {e}")

if __name__ == "__main__":
    test_data_loading()
    test_model_shapes()
