from pathlib import Path
import json, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
CSV = ROOT / 'data' / 'raw' / 'Iris.csv'
ARTIFACT = ROOT / 'artifacts'
ARTIFACT.mkdir(exist_ok=True)
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(1)

FEATURES = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
LABEL_COL = 'Species'
LABELS = ['Iris-setosa','Iris-versicolor','Iris-virginica']
df = pd.read_csv(CSV)
df = df.drop(columns=[c for c in ['Id'] if c in df.columns])
X = df[FEATURES].astype('float32').to_numpy()
y = df[LABEL_COL].map({name:i for i,name in enumerate(LABELS)}).astype('int64').to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,stratify=y,random_state=SEED)
scaler = StandardScaler().fit(X_train)
Xtr = torch.tensor(scaler.transform(X_train), dtype=torch.float32)
Xte = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
ytr = torch.tensor(y_train, dtype=torch.long)
yte = torch.tensor(y_test, dtype=torch.long)

class IrisNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(4,16), nn.ReLU(), nn.Linear(16,12), nn.ReLU(), nn.Linear(12,3))
    def forward(self, x): return self.layers(x)

model = IrisNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.025, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
epochs = 180
history=[]
for epoch in range(1,epochs+1):
    model.train(); optimizer.zero_grad(); logits=model(Xtr); loss=criterion(logits,ytr); loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        train_acc=(model(Xtr).argmax(1)==ytr).float().mean().item()
        val_logits=model(Xte); val_loss=criterion(val_logits,yte).item(); val_acc=(val_logits.argmax(1)==yte).float().mean().item()
    history.append({'epoch':epoch,'loss':round(loss.item(),5),'train_accuracy':round(train_acc,5),'validation_loss':round(val_loss,5),'validation_accuracy':round(val_acc,5)})
model.eval()
with torch.no_grad():
    probs=torch.softmax(model(Xte),dim=1).numpy(); pred=probs.argmax(1)
accuracy=float((pred==y_test).mean())
confusion=[[int(((y_test==i)&(pred==j)).sum()) for j in range(3)] for i in range(3)]
report={
 'dataset':{'name':'Iris Species','source':'https://www.kaggle.com/uciml/iris','rows':int(len(df)),'features':FEATURES,'labels':LABELS,'license':'CC0: Public Domain'},
 'training':{'framework':'PyTorch','architecture':'4 → 16 → 12 → 3 MLP','epochs':epochs,'seed':SEED,'split':'Stratified 80/20','optimizer':'Adam','accuracy':accuracy,'test_rows':int(len(y_test)),'confusion_matrix':confusion,'history':history,
   'feature_mean':scaler.mean_.tolist(),'feature_scale':scaler.scale_.tolist()},
 'feature_summary':{name:{'min':float(df[name].min()),'mean':float(df[name].mean()),'max':float(df[name].max())} for name in FEATURES},
 'samples':df.sample(12,random_state=SEED).to_dict(orient='records')
}
torch.save({'state_dict':model.state_dict(),'feature_mean':scaler.mean_.tolist(),'feature_scale':scaler.scale_.tolist(),'labels':LABELS,'features':FEATURES}, ARTIFACT/'iris_model.pt')
(ARTIFACT/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({'accuracy':accuracy,'confusion_matrix':confusion,'test_rows':len(y_test),'model':str(ARTIFACT/'iris_model.pt')}))

