import torch
import torch.nn as nn
import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

class MNISTV0(nn.Module):
    def __init__(self) -> None:
        super(MNISTV0, self).__init__()
        self.cnn_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2,2), stride=2),
        )
        self.cnn_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(1,1)),
        )
        self.cnn_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2,2), stride=2),
        )
        self.cnn_block_4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(1,1)),
        )
        self.flatten = nn.Flatten()
        self.dense = nn.Sequential(
            nn.Linear(10*26*26, 10),
        )
    
    def forward(self, x):
        x = self.cnn_block_1(x)
        x = self.cnn_block_2(x)
        x = self.cnn_block_3(x)
        x = self.cnn_block_4(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

model = MNISTV0()
model.load_state_dict(torch.load('model/mnist_v0.pth', weights_only=True))
model.eval()

def recognise_digit(image: np.ndarray):
    # Convert canvas data to grayscale image
    image = np.asarray(image.image_data)
    image = Image.fromarray(image)
    image = image.resize((28, 28))
    image = image.convert('L')
    image = np.asarray(image)
    
    # Convert to float32 and normalize to [0,1]
    image = image.astype(np.float32) / 255.0
    
    # Add batch and channel dimensions, and convert to tensor
    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    
    with torch.inference_mode():
        output = model(image)
        predicted_digit = output.argmax(dim=1).item()
        return f'Model Predicted the image as "{predicted_digit}"'

st.title('WebUI for digit classification (1-9)')
st.write("Draw a digit, and click 'Recognize Digit' to process and display it.")

# Create a canvas to draw on
image_data = st_canvas(
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color='#000000',
    height=300,
    width=300,
    drawing_mode="freedraw"
)

if st.button('Recognise'):
    if image_data is not None and image_data.image_data is not None:
        output = recognise_digit(image_data)
        st.write(output)
