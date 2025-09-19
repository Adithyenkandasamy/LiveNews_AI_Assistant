#!/usr/bin/env python3
"""
Test script to diagnose Gemini API connection issues
"""

import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai

def test_gemini_connection():
    print("🔍 Testing Gemini API Connection...")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Check if API key exists
    api_key = os.getenv('GEMINI_API_KEY')
    print(f"1. API Key present: {'✅ Yes' if api_key else '❌ No'}")
    
    if api_key:
        print(f"   API Key length: {len(api_key)} characters")
        print(f"   API Key starts with: {api_key[:8]}..." if len(api_key) > 8 else f"   API Key: {api_key}")
    else:
        print("❌ GEMINI_API_KEY not found in environment variables")
        print("   Make sure your .env file contains:")
        print("   GEMINI_API_KEY=your_actual_api_key_here")
        return False
    
    # Test API configuration
    try:
        genai.configure(api_key=api_key)
        print("2. API Configuration: ✅ Success")
    except Exception as e:
        print(f"2. API Configuration: ❌ Failed - {e}")
        return False
    
    # Test model initialization
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        print("3. Model Initialization: ✅ Success")
    except Exception as e:
        print(f"3. Model Initialization: ❌ Failed - {e}")
        print("   Trying alternative model name: gemini-1.5-flash")
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            print("   Alternative model: ✅ Success")
        except Exception as e2:
            print(f"   Alternative model: ❌ Failed - {e2}")
            return False
    
    # Test actual API call
    try:
        response = model.generate_content("Say hello")
        if response.text:
            print("4. API Call Test: ✅ Success")
            print(f"   Response: {response.text.strip()}")
            return True
        else:
            print("4. API Call Test: ❌ Empty response")
            return False
    except Exception as e:
        print(f"4. API Call Test: ❌ Failed - {e}")
        return False

if __name__ == "__main__":
    success = test_gemini_connection()
    print("\n" + "=" * 50)
    if success:
        print("🎉 Gemini API is working correctly!")
    else:
        print("💥 Gemini API connection failed. Check the errors above.")
        sys.exit(1)
