# About This Project  
This is an attempt to build a Language Model from Scratch using PyTorch  
Pretrained model weights and pickle file is saved in pretrained folder  
Pretrained Model were trained on Tales of Shakespeare Text  

This model is supposed to predict the next characters of prompt and mimic shakespeare writing  

# Files  
[bigram.ipynd](bigram.ipynb) - A straightforward prediction of next character without considering the context of prompt  
[gpt-v1.ipynb](gpt-v1.ipynb) - A transformer model to predict the next character while considering the context of prompt  
[chat.py](chat.py) - Run this python file to use the pretrained model  
[shakespeare.txt](shakespeare.txt) - Dataset consisting of several Shakespeare novels
