import logging

def test_logging():
    """Tests the basic logging functionality."""
    logging.info("Hello World")

# Configure basic logging to see the output
logging.basicConfig(level=logging.INFO)

test_logging()