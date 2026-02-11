import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Neural Style Transfer", layout="centered")
st.title("🎨 Neural Style Transfer using Deep Learning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Image Loader
# -------------------------------
def load_image(image, max_size=400):
    image = image.convert("RGB")
    size = max(image.size)
    if size > max_size:
        scale = max_size / size
        image = image.resize((int(image.size[0]*scale), int(image.size[1]*scale)))

    transform = transforms.Compose([transforms.ToTensor()])
    image = transform(image).unsqueeze(0)
    return image.to(device)

# -------------------------------
# Gram Matrix
# -------------------------------
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    gram = torch.mm(features, features.t())
    return gram

# -------------------------------
# Feature Extraction
# -------------------------------
vgg = models.vgg19(pretrained=True).features.to(device).eval()

layers = {
    '0': 'conv1',
    '5': 'conv2',
    '10': 'conv3',
    '19': 'conv4',
    '28': 'conv5'
}

def get_features(image, model):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

# -------------------------------
# UI Inputs
# -------------------------------
content_file = st.file_uploader("Upload Content Image", type=["jpg", "png"])
style_file = st.file_uploader("Upload Style Image", type=["jpg", "png"])

if content_file and style_file:
    content_img = Image.open(content_file)
    style_img = Image.open(style_file)

    col1, col2 = st.columns(2)
    col1.image(content_img, caption="Content Image", use_container_width=True)
    col2.image(style_img, caption="Style Image", use_container_width=True)

    if st.button("✨ Generate Stylized Image"):
        with st.spinner("Applying style transfer... Please wait"):
            content = load_image(content_img)
            style = load_image(style_img)

            content_features = get_features(content, vgg)
            style_features = get_features(style, vgg)

            style_grams = {layer: gram_matrix(style_features[layer])
                           for layer in style_features}

            generated = content.clone().requires_grad_(True).to(device)

            optimizer = optim.Adam([generated], lr=0.003)
            content_weight = 1e4
            style_weight = 1e2

            # -------------------------------
            # Training Loop
            # -------------------------------
            for step in range(300):
                gen_features = get_features(generated, vgg)

                content_loss = torch.mean(
                    (gen_features['conv4'] - content_features['conv4'])**2
                )

                style_loss = 0
                for layer in style_grams:
                    gen_gram = gram_matrix(gen_features[layer])
                    style_gram = style_grams[layer]
                    style_loss += torch.mean((gen_gram - style_gram)**2)

                total_loss = content_weight * content_loss + style_weight * style_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            # -------------------------------
            # Show Output
            # -------------------------------
            output = generated.squeeze(0).detach().cpu()
            output = transforms.ToPILImage()(output)

            st.subheader("🎉 Stylized Output Image")
            st.image(output, use_container_width=True)

            st.success("Style transfer completed successfully!")
