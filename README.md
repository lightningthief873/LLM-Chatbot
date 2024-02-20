# LLM-Chatbot

## Description
The chatbot is an LLM (Large  Language Model) based conversational chatbot with Natural Language capabilities and not just a predefined query to answer models. Now this chatbot gets its data from any number of PDFs and phrases its answers like a human would do. 

It has a Zephyr 7Billion sharded Model as its backbone and relies on Langchain and Transformers as the technology used for processing NLP models. The model is hosted on Hugging Face, hence it uses Huggingface pipelines to process the string of words.

Now the chatbot requires up to 50GB internal memory to train along with 6+ GB RAM and 8+ GB Nvidia GPU. As it is working on the CUDA platform of Nvidia, the GPU is a must for faster processing.

The chatbot can also process audio from a distant script that translates audio to PDF. It can also process and answer in Spanish and the default English language.

It gives the answers to queries in about 30s and takes about 5 minutes for training.

The base technology used is Pytorch library integrated with Cuda and Python as the scripting language.

## Specifications needed:
1) 8+ GB RAM
2) 8+ GB GPU (NVIDIA)
3) 50 GB SSD

## Steps to run the Chatbot on a local machine:

Step 1: Install Python 3.10+.

Step 2: Install CUDA from the documentation from this link and connect to Python and Pytorch: https://www.freecodecamp.org/news/how-to-setup-windows-machine-for-ml-dl-using-nvidia-graphics-card-cuda/

Step 3: The current code is given for Google collab in both, model_6.py file as well as model_6.ipynb file. Remove unnecessary lines which you don't require. (I have tried my best to remove unnecessary code from the .py file, but the Jupyter notebook is for sole Colab purposes, but in case some further developments are to be made.

Step 4: In place of the Anakin/Zepher model name, give the path of the model folder. If you want to change the model, download your model and place it in that folder.

Step 5: Do the same for the translation model. Both models are available on the Huggingface repository. The full names of the models used are: a) anakin87/zephyr-7b-alpha-sharded b) Helsinki-NLP/opus-mt-en-es c) sentence-transformers/all-mpnet-base-v2.

Step 6: Define your own app.py file for connection to the front end. Though a rudimentary file is given, it is advised to make changes.

Step 7: Change the index.html file for UI.

## Functions to remove from the model_6.py file. 

1) Change the drive directory to the directory which you have as the pdf data.
2) Remove the instances I have created of the chatbot. Use them as references and make new ones in the app.py file per connection.
3) The function to display widgets (Used in Colab for testing purposes UI).

## Basic installations

#install required packages
pip install transformers peft  accelerate bitsandbytes safetensors sentencepiece streamlit chromadb langchain sentence-transformers sacremoses pypdf

### fixing unicode error in google colab
import locale
locale.getpreferredencoding = lambda: "UTF-8"
