import streamlit as st
from PIL import Image, ImageDraw, ImageFont

import torch
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.utils import make_grid
import torchvision.transforms.functional as F
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import seaborn as sns
import pandas as pd

# Step 1: Initialize model with the best available weights
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()

preprocess = weights.transforms()

st.header("Image classifier")
st.write("hello")

uploaded_file = st.file_uploader("Choose an image...")


if uploaded_file is not None:
    #src_image = load_image(uploaded_file)
    img = Image.open(uploaded_file).convert('RGB')

    st.image(uploaded_file, caption='Input Image', use_column_width=True)

    batch = preprocess(img).unsqueeze(0)

    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"{category_name}: {100 * score:.1f}%")

    st.markdown(f"Predicted **{category_name}**: {100 * score:.1f}%")
