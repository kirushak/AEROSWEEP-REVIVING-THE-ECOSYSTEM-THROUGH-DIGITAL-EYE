#!/usr/bin/env python3
"""
Test script for Twilio messaging functionality.
Run with: python test_twilio.py
"""

from utils.twilio_utils import send_direct_sms

def main():
    """Test sending a direct SMS message via Twilio."""
    print("Testing Twilio SMS sending...")
    
    # Attempt to send a test message
    success = send_direct_sms("Test message from Twilio")
    
    if success:
        print("✅ SUCCESS: SMS message sent successfully!")
    else:
        print("❌ FAILED: Could not send SMS message. Check the twilio_debug.log file for details.")
        print("Common issues:")
        print("  1. Auth token not set or incorrect in config/twilio_config.json")
        print("  2. Account suspended or no credits available")
        print("  3. From number not verified or authorized in your Twilio account")
        print("  4. To number not verified (required for trial accounts)")

if __name__ == "__main__":
    main() 