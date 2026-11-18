from pathlib import Path
import json
import torch
from torch import nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT=Path(__file__).resolve().parent
ART=ROOT/'artifacts'
METRICS=json.loads((ART/'metrics.json').read_text(encoding='utf-8'))
LABELS=METRICS['dataset']['labels']
FEATURES=METRICS['dataset']['features']
CHECKPOINT=torch.load(ART/'iris_model.pt',map_location='cpu',weights_only=True)

class IrisNet(nn.Module):
    def __init__(self):
        super().__init__(); self.layers=nn.Sequential(nn.Linear(4,16),nn.ReLU(),nn.Linear(16,12),nn.ReLU(),nn.Linear(12,3))
    def forward(self,x): return self.layers(x)
model=IrisNet(); model.load_state_dict(CHECKPOINT['state_dict']); model.eval()
mean=torch.tensor(CHECKPOINT['feature_mean'],dtype=torch.float32)
scale=torch.tensor(CHECKPOINT['feature_scale'],dtype=torch.float32)
app=FastAPI(title='Iris PyTorch API',version='1.0.0',description='Kaggle Iris veri seti üzerinde eğitilen PyTorch sınıflandırıcısı')
app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=['*'],allow_headers=['*'])
class PredictionInput(BaseModel):
    sepal_length:float=Field(ge=3,le=9,description='Sepal length (cm)')
    sepal_width:float=Field(ge=1,le=6,description='Sepal width (cm)')
    petal_length:float=Field(ge=0,le=8,description='Petal length (cm)')
    petal_width:float=Field(ge=0,le=4,description='Petal width (cm)')
@app.get('/api/health')
def health(): return {'status':'ok','model':'PyTorch MLP','dataset':METRICS['dataset']['name']}
@app.get('/api/metrics')
def metrics(): return METRICS
@app.post('/api/predict')
def predict(row:PredictionInput):
    values=torch.tensor([[row.sepal_length,row.sepal_width,row.petal_length,row.petal_width]],dtype=torch.float32)
    with torch.no_grad():
        x=(values-mean)/scale; p=torch.softmax(model(x),dim=1)[0]
    idx=int(torch.argmax(p))
    return {'prediction':LABELS[idx],'confidence':round(float(p[idx]),5),'probabilities':{LABELS[i]:round(float(p[i]),5) for i in range(len(LABELS))},'input':row.model_dump()}
@app.get('/api/sample')
def sample():
    return METRICS['samples']
