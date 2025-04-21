from twilio.rest import Client
import os
import logging
import json
from typing import Optional, Dict, Any
from pathlib import Path

# Global variable to track if an SMS has been sent in the last minute
# to prevent sending too many messages
last_sms_time = 0
SMS_COOLDOWN = 5  # seconds

# Global config
twilio_config = None

# Import improved logging for better debug messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler('twilio_debug.log')  # Also log to file
    ]
)
logger = logging.getLogger('twilio')

def load_twilio_config(config_path="config/twilio_config.json") -> Dict[str, Any]:
    """Load Twilio configuration from JSON file."""
    global twilio_config, SMS_COOLDOWN
    
    if twilio_config is not None:
        return twilio_config
        
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Update cooldown from config
        if 'cooldown_seconds' in config:
            SMS_COOLDOWN = config['cooldown_seconds']
            
        twilio_config = config
        return config
    except Exception as e:
        logging.error(f"Error loading Twilio config: {e}")
        return {
            "enabled": False,
            "account_sid": "",
            "auth_token": "",
            "from_number": "",
            "to_number": "",
            "cooldown_seconds": 60,
            "trash_keywords": []
        }

def initialize_twilio_client() -> Optional[Client]:
    """Initialize Twilio client using configuration."""
    config = load_twilio_config()
    
    if not config.get("enabled", False):
        return None
        
    try:
        account_sid = config.get("account_sid") or os.environ.get('TWILIO_ACCOUNT_SID') or 'ACdde184f1c2c14071237531ab5c734fb1'
        auth_token = config.get("auth_token") or os.environ.get('TWILIO_AUTH_TOKEN')
        
        if not account_sid or not auth_token:
            logging.warning("Twilio credentials not found in config or environment variables.")
            return None
            
        return Client(account_sid, auth_token)
    except ImportError:
        logging.warning("Twilio package not installed. Run 'pip install twilio' to use SMS features.")
        return None
    except Exception as e:
        logging.error(f"Error initializing Twilio client: {e}")
        return None

def send_trash_detection_sms(
    trash_classes: list,
    location: str = "Unknown",
    client: Optional[Client] = None
) -> bool:
    """
    Send SMS notification about detected trash.
    
    Args:
        trash_classes: List of trash classes detected
        location: Location where trash was detected
        client: Optional Twilio client (if None, will initialize new client)
    
    Returns:
        bool: True if SMS was sent successfully, False otherwise
    """
    import time
    
    global last_sms_time
    
    logger.info(f"Attempting to send SMS about trash classes: {trash_classes} at {location}")
    
    config = load_twilio_config()
    
    # Skip if not enabled
    if not config.get("enabled", False):
        logger.info("SMS notifications are disabled in config")
        return False
    
    # Check if auth token is missing
    if not config.get("auth_token"):
        logger.error("AUTH TOKEN MISSING: Cannot send SMS without a valid auth token")
        logger.info("Please update config/twilio_config.json with your Twilio auth token")
        return False
    
    # Check if we're in the cooldown period
    current_time = time.time()
    if current_time - last_sms_time < SMS_COOLDOWN:
        logger.info(f"SMS cooldown active. Last SMS sent {current_time - last_sms_time:.1f}s ago. Cooldown: {SMS_COOLDOWN}s")
        return False
    
    # Get phone numbers from config
    to_number = config.get("to_number") or "+919150526354"
    from_number = config.get("from_number") or "+18066022074"
    
    logger.info(f"SMS will be sent from {from_number} to {to_number}")
    
    if not client:
        logger.info("Initializing Twilio client")
        client = initialize_twilio_client()
    
    if not client:
        logger.error("Failed to initialize Twilio client")
        return False
    
    try:
        # Format message with detected trash types
        trash_types = ", ".join(trash_classes)
        message = f"ALERT: Trash detected! Types: {trash_types}. Location: {location}"
        
        logger.info(f"Sending message: {message}")
        
        # Send the message
        message = client.messages.create(
            body=message,
            from_=from_number,
            to=to_number
        )
        
        # Update last SMS time
        last_sms_time = current_time
        
        logger.info(f"SMS notification sent successfully! SID: {message.sid}")
        return True
    except Exception as e:
        logger.error(f"Failed to send SMS notification: {str(e)}")
        # Print more detailed error information
        import traceback
        logger.error(traceback.format_exc())
        return False

def is_trash_class(class_name: str) -> bool:
    """Check if a detected class is considered trash."""
    config = load_twilio_config()
    trash_keywords = config.get("trash_keywords", [])
    
    # Always treat all classes from trash detection models as trash
    # These models are specifically trained for trash detection
    current_model = os.environ.get("CURRENT_MODEL_PATH", "")
    if current_model:
        model_name = os.path.basename(current_model).lower()
        if "trash" in model_name or model_name == "yolov10.pt" or model_name == "mobilenet_trash_detector.pt":
            return True
    
    # If no keywords defined, assume it's a trash detection model already
    if not trash_keywords:
        return True
        
    # Check if class name contains any of the trash keywords
    class_name = class_name.lower()
    return any(keyword.lower() in class_name for keyword in trash_keywords)

def send_direct_sms(message_text: str) -> bool:
    """
    Send an SMS directly using Twilio without cooldown or additional checks.
    Used for direct user actions like clicking the webcam button.
    
    Args:
        message_text: The text message to send
    
    Returns:
        bool: True if SMS was sent successfully, False otherwise
    """
    logger.info(f"Attempting to send direct SMS: {message_text}")
    
    config = load_twilio_config()
    
    # Skip if not enabled
    if not config.get("enabled", False):
        logger.warning("SMS notifications are disabled in config")
        return False
    
    # Check if auth token is missing
    if not config.get("auth_token"):
        logger.error("AUTH TOKEN MISSING: Cannot send SMS without a valid auth token")
        return False
    
    # Get phone numbers from config
    to_number = config.get("to_number") or "+918056316600"
    from_number = config.get("from_number") or "+18066022074"
    
    # Initialize client directly from configuration
    try:
        account_sid = config.get("account_sid") or 'ACdde184f1c2c14071237531ab5c734fb1'
        auth_token = config.get("auth_token")
        
        if not account_sid or not auth_token:
            logger.error("Twilio credentials not found in config")
            return False
            
        client = Client(account_sid, auth_token)
    except Exception as e:
        logger.error(f"Error initializing Twilio client: {e}")
        return False
    
    try:
        # Send the message
        message = client.messages.create(
            body=message_text,
            from_=from_number,
            to=to_number
        )
        
        logger.info(f"Direct SMS notification sent successfully! SID: {message.sid}")
        return True
    except Exception as e:
        logger.error(f"Failed to send direct SMS notification: {str(e)}")
        # Print more detailed error information
        import traceback
        logger.error(traceback.format_exc())
        return False 