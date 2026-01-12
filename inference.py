import torch
from torchvision import transforms
from PIL import Image
import streamlit as st

CONFIG = {
    'device' : 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_classes' : 4,
    'model_pth_path': 'model/brain_tumor_classifier.pth',
    'output_reference': {0: 'glioma', 1: 'meningioma', 2: 'No tumor', 3: 'pituitary'},
}

class Model(torch.nn.Module):
    def __init__(self, dropout1=0.2, dropout2=0.1, hidden1=1024, hidden2=256, hidden3=128, out1=32, out2=64, dropout3=0.5):
        super().__init__()
        self.layer_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=out1, kernel_size=(3, 3), padding=1, padding_mode='reflect'),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout1)
        )
        self.layer_2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=out1, out_channels=out2, kernel_size=(3, 3), padding=1, padding_mode='reflect'),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout2)
        )
        self.flatten = torch.nn.Flatten()
        self.fc_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=out2 * (out2 - 8) * (out2 - 8), out_features=hidden1),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden1, out_features=hidden2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout3),
            torch.nn.Linear(in_features=hidden2, out_features=hidden3),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden3, out_features=CONFIG['num_classes'])
        )

    def forward(self, x):
        x = self.layer_1(x)  
        x = self.layer_2(x)
        x = self.flatten(x)
        x = self.fc_layer(x)
        return x
    

img_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


@st.cache_resource
def load_model():
    checkpoint = torch.load(
        CONFIG['model_pth_path'],
        map_location=CONFIG['device'],
        weights_only=False
    )

    cfg = checkpoint['config']

    model = Model(
        hidden1=cfg['hidden1'],
        hidden2=cfg['hidden2'],
        hidden3=cfg['hidden3'],
        dropout1=cfg['dropout1'],
        dropout2=cfg['dropout2'],
        dropout3=cfg['dropout3'],
        out1=cfg['out1'],
        out2=cfg['out2']
    ).to(CONFIG['device'])

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def predict(img_input):
    model = load_model()

    try:
        img = Image.open(img_input).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    img = img_transform(img)
    img = img.unsqueeze(0).to(CONFIG['device'])

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = probs.argmax().item()

    label = CONFIG['output_reference'].get(pred_idx, "Unknown Tumor")

    # Optional: return confidence scores
    confidences = {
        CONFIG['output_reference'][i]: probs[i].item() * 100
        for i in range(len(probs))
    }

    return label, confidences