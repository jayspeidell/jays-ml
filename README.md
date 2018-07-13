This is a collection of helper files I use for machine learning projects. It's mostly for my own backup, and as I use it for Kaggle it's more intended for copying and pasting than importing. Feel free to use and copy anything you find here, just keep in mind the unrefined state. It's currently out of date, I have a much more robust version of the MultiLabel class I'll upload in the not-too-distant future and I'm working on a fully functional model stacking class. 

# Contents 

### 1. model_training/MultiLabel

A class that builds a multi-label model from any regular classification model that has standard fit and predict methods. 

### 2. model_training/stack_predictions

A function for model stacking that builds new features based on predictions from input models. Barebones and messy, it's not useful for model validation in it's current state. 
