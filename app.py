import streamlit as st
import torch
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image

st.set_page_config(page_title="Auto Style Transfer", layout="centered")
st.title("🎨 Automatic Neural Style Transfer")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Load Fixed Style Image
# -------------------------------
STYLE_IMAGE_PATH = "style.jpg"

def load_image(image, max_size=400):
    image = image.convert("RGB")
    size = max(image.size)
    if size > max_size:
        scale = max_size / size
        image = image.resize((int(image.size[0]*scale), int(image.size[1]*scale)))
    transform = transforms.ToTensor()
    return transform(image).unsqueeze(0).to(device)

# -------------------------------
# Gram Matrix
# -------------------------------
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    return torch.mm(features, features.t())

# -------------------------------
# VGG Model
# -------------------------------
vgg = models.vgg19(pretrained=True).features.to(device).eval()

layers = {'0':'conv1','5':'conv2','10':'conv3','19':'conv4','28':'conv5'}

def get_features(image):
    features = {}
    x = image
    for name, layer in vgg._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

# -------------------------------
# UI
# -------------------------------
content_file = st.file_uploader("Upload Content Image", type=["jpg","png"])

if content_file:
    content_img = Image.open(content_file)
    st.image(content_img, caption="Content Image", use_container_width=True)

    if st.button("✨ Apply Artistic Style"):
        with st.spinner("Stylizing image..."):
            content = load_image(content_img)
            style = load_image(Image.open(STYLE_IMAGE_PATH))

            content_features = get_features(content)
            style_features = get_features(style)
            style_grams = {k: gram_matrix(v) for k, v in style_features.items()}

            generated = content.clone().requires_grad_(True)
            optimizer = optim.Adam([generated], lr=0.003)

            for _ in range(300):
                gen_features = get_features(generated)

                content_loss = torch.mean(
                    (gen_features['conv4'] - content_features['conv4'])**2
                )

                style_loss = sum(
                    torch.mean((gram_matrix(gen_features[l]) - style_grams[l])**2)
                    for l in style_grams
                )

                loss = 1e4 * content_loss + 1e2 * style_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            output = generated.squeeze(0).detach().cpu()
            output = transforms.ToPILImage()(output)

            st.subheader("🎉 Stylized Output")
            st.image(output, use_container_width=True)
