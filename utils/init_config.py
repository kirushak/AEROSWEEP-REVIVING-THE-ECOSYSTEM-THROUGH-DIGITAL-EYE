import os
import json
import logging
from pathlib import Path
from utils import glo

def load_twilio_config():
    """Load Twilio configuration from file and set environment variables."""
    config_path = Path("config/twilio_config.json")
    
    if not config_path.exists():
        logging.warning(f"Twilio config file not found: {config_path}")
        return
        
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Only set environment variables if enabled
        if config.get("enabled", False):
            # Set environment variables for Twilio
            if config.get("account_sid"):
                os.environ["TWILIO_ACCOUNT_SID"] = config["account_sid"]
                
            if config.get("auth_token"):
                os.environ["TWILIO_AUTH_TOKEN"] = config["auth_token"]
                
            logging.info("Twilio configuration loaded successfully.")
            
            # Store in global variables
            glo.set_value("twilio_enabled", True)
            glo.set_value("twilio_config", config)
        else:
            logging.info("Twilio is disabled in configuration.")
            glo.set_value("twilio_enabled", False)
            
    except Exception as e:
        logging.error(f"Error loading Twilio config: {e}")
        glo.set_value("twilio_enabled", False)

def initialize_configs():
    """Initialize all configuration settings."""
    # Initialize Twilio
    load_twilio_config()
    
    # Add other configuration initializations here as needed 