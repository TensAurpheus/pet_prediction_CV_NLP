# PetFinder Adoption Speed Prediction (Deep Learning Final Project)

This project tackles a multiclass prediction task from the PetFinder.my dataset: **predicting how quickly a pet gets adopted** based on profile information.  
The target metric is **Quadratic Weighted Kappa (QWK)**, which emphasizes how close predictions are to the true adoption class.

## What I Built
- **Text pipeline**: transformer-based modeling for pet descriptions
- **Image pipeline**: CNN-based classifier (ResNet/EfficientNet experiments) on pet photos
- **Multimodal experiments**: validation-time fusion of text + image predictions
- End-to-end notebook flow: data prep, training loops, validation, and model checkpointing

## Key Result
- Best single-branch validation performance in this notebook reached approximately **QWK = 0.3865** (image model branch).
- Additional fusion/multimodal experiments were implemented to compare combined vs. single-modality behavior.

## Tech Stack
- Python, PyTorch, torchvision
- Hugging Face Transformers
- pandas, NumPy, scikit-learn, matplotlib
- Jupyter Notebook

## Project File
- `DL_FinalProject_Рудий_АІ.ipynb` — complete workflow and experiments

## Notes
This is an educational final project focused on practical deep learning experimentation, reproducible training structure, and metric-driven model comparison.
