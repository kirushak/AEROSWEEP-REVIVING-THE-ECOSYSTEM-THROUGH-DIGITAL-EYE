import os
import json
import logging
from pathlib import Path
from twilio.rest import Client

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('twilio_test')

def test_twilio_connection():
    """Test Twilio connection and configuration."""
    logger.info("Starting Twilio connection test")
    
    # Check if twilio is installed
    try:
        import twilio
        logger.info(f"Twilio package is installed (version: {twilio.__version__})")
    except ImportError:
        logger.error("Twilio package is not installed. Run 'pip install twilio'")
        return False
    
    # Check config file
    config_path = Path("config/twilio_config.json")
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return False
    
    # Load config
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info("Successfully loaded config file")
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        return False
    
    # Check enabled status
    if not config.get("enabled", False):
        logger.error("Twilio is disabled in config (enabled: false)")
        return False
    
    # Check credentials
    account_sid = config.get("account_sid")
    auth_token = config.get("auth_token")
    from_number = config.get("from_number")
    to_number = config.get("to_number")
    
    if not account_sid:
        logger.error("Missing account_sid in config")
        return False
    
    if not auth_token:
        logger.error("Missing auth_token in config. You need to obtain this from your Twilio account.")
        logger.info("Go to https://www.twilio.com/console to find your auth token")
        return False
    
    if not from_number:
        logger.error("Missing from_number in config")
        return False
    
    if not to_number:
        logger.error("Missing to_number in config")
        return False
    
    # Try to initialize client
    try:
        client = Client(account_sid, auth_token)
        logger.info("Successfully initialized Twilio client")
    except Exception as e:
        logger.error(f"Error initializing Twilio client: {e}")
        return False
    
    # Try to send test message
    try:
        logger.info("Attempting to send test SMS...")
        message = client.messages.create(
            body="YOLOSHOW Test Message - This is a test from your trash detection system",
            from_=from_number,
            to=to_number
        )
        logger.info(f"Test SMS sent successfully! SID: {message.sid}")
        return True
    except Exception as e:
        logger.error(f"Error sending test SMS: {e}")
        return False

if __name__ == "__main__":
    success = test_twilio_connection()
    if success:
        print("\nTwilio test completed successfully! ✅")
    else:
        print("\nTwilio test failed. Check the logs above for details. ❌")
        print("\nTo fix this issue:")
        print("1. Make sure you have a valid Twilio account")
        print("2. Update config/twilio_config.json with your actual auth token")
        print("3. Verify that your Twilio phone number is properly set up")
        print("4. Ensure the to_number is verified in your Twilio account (for trial accounts)") 