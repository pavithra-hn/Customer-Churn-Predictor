#!/usr/bin/env python3
"""
Quick setup script for Customer Churn Predictor
Run this to set up everything in one go!
"""

import os
import subprocess
import sys

def run_command(command):
    """Run a command and return success status"""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {command}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🚀 Setting up Customer Churn Predictor...")
    print("=" * 50)
    
    # Step 1: Install requirements
    print("\n📦 Installing dependencies...")
    if not run_command("pip install -r requirements.txt"):
        print("Failed to install dependencies. Please install manually.")
        return
    
    # Step 2: Train model
    print("\n🤖 Training machine learning model...")
    if not run_command("python model_training.py"):
        print("Failed to train model. Please check the script.")
        return
    
    # Step 3: Check if model file exists
    if os.path.exists("churn_model.pkl"):
        print("✅ Model file created successfully!")
    else:
        print("❌ Model file not found. Please run model_training.py manually.")
        return
    
    print("\n🎉 Setup complete!")
    print("\n🌟 To run the application:")
    print("   streamlit run app.py")
    print("\n📱 Your app will open at: http://localhost:8501")
    
    # Ask if user wants to run the app now
    response = input("\nWould you like to run the app now? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        print("\n🚀 Starting Streamlit app...")
        os.system("streamlit run app.py")

if __name__ == "__main__":
    main()