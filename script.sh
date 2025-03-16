#!/bin/bash

# Define available models and their corresponding GitHub repos & Hugging Face models
declare -A MODELS
MODELS["spark"]="https://huggingface.co/SparkAudio/Spark-TTS-0.5B pretrained_models/Spark-TTS-0.5B"

START_COMMAND="python api-server.py"


# Create and activate Conda environment
echo "Setting up Conda environment..."
conda create -y -n "spark" python="3.10"
source activate "spark" || conda activate "spark"

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Downloading models..."
git lfs install
git clone $MODELS["spark"]
cd ..

# Start the model server
echo "Starting Spark tts server..."
eval "$START_COMMAND"