import time
import os
import face_recognition
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import torch.nn.functional as F
import warnings
import gradio as gr

warnings.filterwarnings("ignore")

# ----------------------- Định nghĩa Model -----------------------
class Model(nn.Module):
    def __init__(self, num_classes,latent_dim= 2048, lstm_layers=1 , hidden_dim = 2048, bidirectional = False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT) #Residual Network CNN
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim,hidden_dim, lstm_layers,  bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048,num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        batch_size,seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size,seq_length,2048)
        x_lstm,_ = self.lstm(x,None)
        return fmap,self.dp(self.linear1(torch.mean(x_lstm,dim = 1)))

# ----------------------- Các hàm xử lý -----------------------
def frame_extract(path, start_frame):
    vidObj = cv2.VideoCapture(path)
    vidObj.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    success = True
    while success:
        success, image = vidObj.read()
        if success:
            yield image

def process_data(video_path, sequence_length, transform):
    frames = []
    face_frames = []
    a = int(100 / sequence_length)
    first_frame = np.random.randint(0, a)
    for i, frame in enumerate(frame_extract(video_path, first_frame)):
        faces = face_recognition.face_locations(frame)
        try:
            top, right, bottom, left = faces[0]
            frame = frame[top:bottom, left:right, :]
        except:
            continue
        # Resize frame khuôn mặt về kích thước (112,112)
        face_frame = cv2.resize(frame, (112, 112))
        # Chuyển đổi từ BGR sang RGB để hiển thị đúng màu trong Gradio
        face_frame_rgb = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        face_frames.append(face_frame_rgb)
        
        frames.append(transform(frame))
        if len(frames) == sequence_length:
            break

    frames = torch.stack(frames)
    frames = frames[:sequence_length]
    return frames, face_frames

def predict(model, input_tensor):
    start_time = time.time()
    with torch.no_grad():
        labels = ["Fake", "Real"]
        model.eval()
        _, outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        max_prob, index = torch.max(probabilities, 1)
        y_pred = labels[index.item()]
        y_proba = max_prob.item()
    exec_time = time.time() - start_time
    return y_pred, y_proba, exec_time

# ----------------------- Load Model & Cấu hình Transform -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(2)
path_to_model = "80_frame"
model.load_state_dict(torch.load(path_to_model, map_location=device))
model.eval()

sequence_length = 80
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
video_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# ----------------------- Hàm inference cho Gradio -----------------------
def inference(video_path):
    start_time = time.time()
    frames, face_frames = process_data(video_path, sequence_length, video_transforms)
    frames = frames.unsqueeze(0)  # Thêm batch dimension
    y_pred, y_proba, re_time = predict(model, frames)
    exec_time = time.time() - start_time
    result_text = f"Prediction: {y_pred}\nProbability: {y_proba:.4f}\nEntire Process Time: {exec_time:.2f} seconds\nReference Time: {re_time:.2f} seconds"
    return result_text, face_frames

# ----------------------- Tạo Gradio Interface -----------------------
demo = gr.Interface(
    fn=inference,
    inputs=gr.Video(label="Upload Video"),
    outputs=[
        gr.Textbox(label="📋 Result"),
        gr.Gallery(label="⏳ Considered Frames", columns=5)  # Sử dụng tham số columns nếu được hỗ trợ
    ],
    title="Defake Detection in Videos Demo",
    description="🕵️‍♂️ Upload a video to refer whether the detected face is Real or Fake."
)

demo.launch(share=True)
