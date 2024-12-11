import os
import subprocess
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, HTTPException
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData
from sqlalchemy.orm import sessionmaker
from tempfile import NamedTemporaryFile
import uvicorn

import argparse
import numpy  as np
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import matplotlib.pyplot as plt

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

import cv2
import pandas as pd
import numpy as np

from fastapi import FastAPI, File, UploadFile
from io import BytesIO
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# FastAPI 앱 정의
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 오리진을 허용 (배포 시 구체적인 도메인으로 제한)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메소드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

# 클래스 이름 정의
class_names = [
    'bean_sprouts', 'beef', 'chicken', 'egg', 'fork',
    'garlic', 'green_onion', 'kimchi', 'onion',
    'potato', 'spam'
]

# MySQL 데이터베이스 설정
DATABASE_URL = "mysql+pymysql://{id:password}@localhost:3306/myrefrigerator"
engine = create_engine(DATABASE_URL)
metadata = MetaData()

# 예측 결과를 저장할 테이블 정의
predictions = Table(
    'ingredients', metadata,
    Column('ingredients_id', Integer, primary_key=True),
    Column('name', String(50)),
    Column('url', String(255))
)

# 테이블 생성 (이미 존재하면 생성되지 않음)
metadata.create_all(engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 탐지 기능을 수행하는 함수
def run_single_image_detection(image_path):
    result_dir = "capstone_AI/yolov5/runs/detect/exp8/"
    labels_dir = os.path.join(result_dir, "labels")
    predicted_image_path = os.path.join(result_dir, os.path.basename(image_path))

    # YOLO 모델을 사용하여 탐지 수행
    subprocess.run([
        "python", "capstone_AI/yolov5/detect.py",
        "--weights", "capstone_AI/yolov5/runs/train/exp9/weights/best.pt",
        "--img", "416",
        "--conf", "0.1",
        "--source", image_path,
        "--project", "capstone_AI/yolov5/runs/detect",
        "--name", "exp8",
        "--exist-ok", "--save-txt"
    ], check=True)

    labels_file = os.path.join(labels_dir, os.path.splitext(os.path.basename(image_path))[0] + ".txt")
    detected_classes_set = set()
    
    # 라벨 파일이 있는 경우 클래스 정보를 set에 추가
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                class_id = int(line.split()[0])
                detected_classes_set.add(class_names[class_id])

    return detected_classes_set

print("test")

def get_class_string(class_name):
    class_mapping = {
        'bean_sprouts': "test_bean_sprouts.jpg",
        'beef': "test_beef.jpg",
        'chicken': "test_chicken.jpg",
        'egg': "test_egg.jpg",
        'fork': "test_fork.jpg",
        'garlic': "test_garlic.jpg",
        'green_onion': "test_green_onion.jpg",
        'kimchi': "test_kimchi.jpg",
        'onion': "test_onion.jpg",
        'potato': "test_potato.jpg",  # assuming you meant 'potato'
        'spam': "test_spam.jpg",
    }
    return class_mapping.get(class_name, "unknown_class")

setup=True  #ONLY SET TRUE ONCE THEN SET TO FALSE
data='{path}'

hyperparams=False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs=100
learning_rate=0.005
batch_size=128
train=True

model='{path}'
test_acc=True
cam=False

class Deep_Emotion(nn.Module):
    def __init__(self):
        '''
        Deep_Emotion class contains the network architecture.
        '''
        super(Deep_Emotion,self).__init__()
        self.conv1 = nn.Conv2d(1,10,3)
        self.conv2 = nn.Conv2d(10,10,3)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(10,10,3)
        self.conv4 = nn.Conv2d(10,10,3)
        self.pool4 = nn.MaxPool2d(2,2)

        self.norm = nn.BatchNorm2d(10)

        self.fc1 = nn.Linear(810,50)
        self.fc2 = nn.Linear(50,7)

        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(640, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 640)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self,input):
        out = self.stn(input)

        out = F.relu(self.conv1(out))
        out = self.conv2(out)
        out = F.relu(self.pool2(out))

        out = F.relu(self.conv3(out))
        out = self.norm(self.conv4(out))
        out = F.relu(self.pool4(out))

        out = F.dropout(out)
        out = out.view(-1, 810)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = '{path}'  # 모델 파일 경로

# 모델 로드
net = Deep_Emotion()
net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
net.to(device)
net.eval()

# 감정 클래스 정의
classes = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')

# 이미지 전처리
transformation = transforms.Compose([
    transforms.Grayscale(),  # 흑백 이미지로 변환
    transforms.Resize((48, 48)),  # 48x48 크기로 변환
    transforms.ToTensor(),  # 텐서로 변환
    transforms.Normalize((0.5,), (0.5,))  # 정규화
])

# 감정 예측 함수
def predict_emotion(image_bytes: BytesIO):
    img = Image.open(image_bytes)  # 이미지 로드
    img = transformation(img).unsqueeze(0)  # 배치 차원 추가

    # 이미지를 GPU로 이동 (필요시)
    img = img.to(device)

    # 예측 수행
    with torch.no_grad():
        output = net(img)
        prediction = F.softmax(output, dim=1)
        predicted_class = torch.argmax(prediction, 1).item()

    # 예측된 감정
    predicted_emotion = classes[predicted_class]

    return predicted_emotion

# 이미지 업로드 및 감정 예측을 위한 엔드포인트
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # 이미지 파일 읽기
        img_bytes = await file.read()
        img = BytesIO(img_bytes)
        
        # 감정 예측
        predicted_emotion = predict_emotion(img)

        # 예측된 감정 반환
        return {"predicted_emotion": predicted_emotion}

    except Exception as e:
        return {"error": str(e)}


@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    try:
        # 업로드된 이미지 파일 저장
        with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        try:
            # YOLO 모델을 사용하여 탐지 수행
            detected_classes = run_single_image_detection(tmp_path)

            # 데이터베이스 연결
            db = SessionLocal()
            inserted_classes = []
            existing_classes = []

            for class_name in detected_classes:
                # 매핑된 URL 가져오기
                mapped_url = get_class_string(class_name)

                # 데이터베이스에서 이미 저장된 클래스인지 확인
                existing_entry = db.execute(
                    predictions.select().where(predictions.c.name == class_name)
                ).fetchone()

                if not existing_entry:
                    # 저장되지 않은 경우에만 삽입
                    db.execute(predictions.insert().values(name=class_name, url=mapped_url))
                    inserted_classes.append({"name": class_name, "url": mapped_url})
                else:
                    # 이미 존재하는 클래스
                    existing_classes.append({"name": class_name, "url": existing_entry.url})

            # 변경사항 커밋 및 연결 종료
            db.commit()
            db.close()

            print([{"name": cls, "url": get_class_string(cls)} for cls in detected_classes])

            # 탐지 결과 반환
            return {
                "all_detected_classes": [{"name": cls} for cls in detected_classes],
                "newly_inserted_classes": inserted_classes,
                "already_existing_classes": existing_classes,
                "message": "Detection complete. Check the classified details above."
            }

        finally:
            # 임시 파일 삭제
            os.remove(tmp_path)

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail="YOLO 모델 실행 중 오류가 발생했습니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# uvicorn을 사용하여 서버를 실행
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)