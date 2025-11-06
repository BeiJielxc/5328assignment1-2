#!/usr/bin/env python
# coding: utf-8

# # Co-Teaching++ (GPU, Forget-Rate Schedule + Dropout/BN)


# In[1]:


# ==== Deterministic setup (reproducibility) ====
import os, random, numpy as np, torch, math
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = bool(torch.cuda.is_available())
NUM_WORKERS = 0  # keep 0 to avoid multi-process RNG issues unless necessary

# Make cuDNN deterministic globally
try:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
except Exception as _e:
    pass

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception as _e:
        pass
print("Device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))



# In[3]:


def set_seed(seed=42):
    import os, random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def _infer_hw_c_from_flat_dim(D:int):
    s = int(np.sqrt(D) + 1e-8)
    if s*s == D: return s, s, 1
    if D % 3 == 0:
        s3 = int(np.sqrt(D//3) + 1e-8)
        if s3*s3*3 == D: return s3, s3, 3
    return None, None, None

def load_npz(path:str):
    d = np.load(path)
    Xtr, Str = d["Xtr"], d["Str"]
    Xts, Yts = d["Xts"], d["Yts"]
    if Xtr.ndim == 4 and Xtr.shape[-1]==3: H,W,C = Xtr.shape[1], Xtr.shape[2], 3
    elif Xtr.ndim == 3: H,W,C = Xtr.shape[1], Xtr.shape[2], 1
    elif Xtr.ndim == 2:
        H,W,C = _infer_hw_c_from_flat_dim(Xtr.shape[1])
        if H is None: raise ValueError("Cannot infer HWC")
    else: raise ValueError(f"Bad shape: {Xtr.shape}")
    num_classes = int(max(Str.max(), Yts.max())+1)
    return (Xtr, Str, Xts, Yts, (H,W,C), num_classes)

class NPZImageDataset(Dataset):
    def __init__(self, X, y, shape_hw_c):
        X = X.astype(np.float32)
        H,W,C = shape_hw_c
        if X.ndim == 2:
            X = X.reshape(-1,H,W) if C==1 else X.reshape(-1,H,W,C)
        if X.max()>1.5: X = X/255.0
        if X.ndim == 3: X = X[:,None,:,:]
        elif X.ndim == 4: X = np.transpose(X,(0,3,1,2))
        X = (X - X.mean())/(X.std()+1e-6)
        self.X, self.y = X, y.astype(np.int64)
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return torch.from_numpy(self.X[i]), int(self.y[i])



# In[5]:


class CNN_BN_Drop(nn.Module):
    def __init__(self, in_ch=1, num_classes=10, p_drop=0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p_drop),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p_drop),
        )
        self.adapt = nn.AdaptiveAvgPool2d((7,7))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*7*7, 256), nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(256, num_classes)
        )
    def forward(self,x):
        x = self.features(x)
        x = self.adapt(x)
        return self.classifier(x)

def make_model(in_ch, C, p_drop=0.4):
    return CNN_BN_Drop(in_ch=in_ch, num_classes=C, p_drop=p_drop)



# In[7]:


def topk_indices_by_small_loss(logits, y, keep_ratio):
    losses = F.cross_entropy(logits, y, reduction='none')
    k = max(1, int(keep_ratio * len(losses)))
    return torch.topk(-losses, k=k).indices

def coteach_epoch(model1, model2, loader, opt1, opt2, keep_ratio):
    model1.train(); model2.train(); total=0.0
    for x,y in loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        l1, l2 = model1(x), model2(x)
        idx1 = topk_indices_by_small_loss(l1, y, keep_ratio)
        idx2 = topk_indices_by_small_loss(l2, y, keep_ratio)
        loss1 = F.cross_entropy(l1[idx2], y[idx2])
        loss2 = F.cross_entropy(l2[idx1], y[idx1])
        opt1.zero_grad(); loss1.backward(); opt1.step()
        opt2.zero_grad(); loss2.backward(); opt2.step()
        total += (loss1.item()+loss2.item())/2
    return total/len(loader)

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); corr=0; tot=0
    for x,y in loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x).argmax(1)
        corr += (pred==y).sum().item(); tot += y.size(0)
    return corr/tot

def linear_schedule(epoch, max_epoch, final_forget):
    return min(final_forget, final_forget * epoch / max(1, max_epoch))

def cosine_schedule(epoch, max_epoch, final_forget):
    t = epoch / max(1, max_epoch)
    return final_forget * (0.5 - 0.5*math.cos(math.pi * t))



# In[9]:


from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def train_config_coteach_scheduled(
    dataset_path, epochs, wd, lr, seed=2025,
    final_forget=0.6, sched='linear', p_drop=0.4,
    warmup_epochs=8,
    tr_idx=None, va_idx=None,
    generator=None
):
   
    set_seed(seed)

    
    Xtr, Str, Xts, Yts, hwc, C = load_npz(dataset_path)

    
    if tr_idx is None or va_idx is None:
        tr_idx, va_idx = train_test_split(
            np.arange(len(Str)), test_size=0.1, stratify=Str, random_state=seed
        )

    
    tr = NPZImageDataset(Xtr[tr_idx], Str[tr_idx], shape_hw_c=hwc)
    va = NPZImageDataset(Xtr[va_idx], Str[va_idx], shape_hw_c=hwc)
    ts = NPZImageDataset(Xts, Yts, shape_hw_c=hwc)

    
    tr_loader = DataLoader(tr, batch_size=256, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                           generator=generator)
    va_loader = DataLoader(va, batch_size=256, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    ts_loader = DataLoader(ts, batch_size=256, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    
    in_ch = hwc[2]
    m1 = make_model(in_ch, C, p_drop).to(DEVICE)
    m2 = make_model(in_ch, C, p_drop).to(DEVICE)
    opt1 = torch.optim.AdamW(m1.parameters(), lr=lr, weight_decay=wd)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=lr, weight_decay=wd)

    
    for ep in range(1, epochs + 1):
        if ep <= warmup_epochs:
            m1.train(); m2.train()
            for x, y in tr_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt1.zero_grad(set_to_none=True)
                F.cross_entropy(m1(x), y).backward()
                opt1.step()
                opt2.zero_grad(set_to_none=True)
                F.cross_entropy(m2(x), y).backward()
                opt2.step()
            continue

        
        t_ep = ep - warmup_epochs
        T_total = max(1, epochs - warmup_epochs)
        if sched == 'cosine':
            forget = cosine_schedule(t_ep, T_total, final_forget)
        else:
            forget = linear_schedule(t_ep, T_total, final_forget)

        keep_ratio = float(min(1.0, max(0.0, 1.0 - forget)))
        coteach_epoch(m1, m2, tr_loader, opt1, opt2, keep_ratio)

    
    va_acc = max(evaluate(m1, va_loader), evaluate(m2, va_loader))
    ts_acc = max(evaluate(m1, ts_loader), evaluate(m2, ts_loader))
    val_loss = 1.0 - va_acc
    # new update ,return model
    return val_loss, ts_acc, m1, m2




# In[16]:


from sklearn.model_selection import KFold

def tune_and_report_coteach_plus_cv(
    dataset_path,
    lr=1e-3,
    it_values=(30,),
    wd_values=(5e-2,),
    seed=42,
    final_forget=0.6,
    sched='linear',
    p_drop=0.5,
    num_folds=10
):
    print(f"==== 10-Fold CV on Dataset: {dataset_path} ====")


    set_seed(seed)


    Xtr, Str, Xts, Yts, hwc, C = load_npz(dataset_path)
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

    fold_accs, fold_losses = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(Xtr)):
        print(f"\n===== Fold {fold + 1}/{num_folds} =====")

        g = torch.Generator(device='cpu').manual_seed(seed + fold)

        best = None
        for it in it_values:
            for wd in wd_values:
                vloss, tacc = train_config_coteach_scheduled(
                    dataset_path=dataset_path,
                    epochs=it, wd=wd, lr=lr,
                    seed=seed + fold,             
                    final_forget=final_forget,
                    sched=sched,
                    p_drop=p_drop,
                    tr_idx=train_idx, va_idx=val_idx,
                    generator=g
                )
                print(f"Fold {fold+1} | wd={wd}, it={it} | val_loss={vloss:.4f}, test_acc={tacc*100:.2f}%")

                if (best is None) or (vloss < best[0] - 1e-12) or (abs(vloss - best[0]) < 1e-12 and tacc > best[1]):
                    best = (vloss, tacc, wd, it)

        fold_losses.append(best[0])
        fold_accs.append(best[1])

    mean_acc = float(np.mean(fold_accs))
    std_acc  = float(np.std(fold_accs))
    mean_loss = float(np.mean(fold_losses))

    print("\n==== 10-Fold Cross Validation Result ====")
    print(f"Mean Val Loss: {mean_loss:.4f}")
    print(f"Mean Test Accuracy: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    return mean_loss, mean_acc




def tune_and_report_coteach_plus(dataset_path, lr=1e-3, it_values=(500,), wd_values=(1e-4,1e-3,5e-2,1e-2,5e-1,1e-1),
                                 seed=42, final_forget=0.6, sched='linear', p_drop=0.4):
    print(f"==== Dataset: {dataset_path} ====")
    print("Tuned configs (both orientations):")
    best=None
    for it in it_values:
        for wd in wd_values:
            vloss,tacc = train_config_coteach_scheduled(dataset_path, epochs=it, wd=wd, lr=lr, seed=seed,
                                                        final_forget=final_forget, sched=sched, p_drop=p_drop)
            print(f"wd={wd}, it={it} | val_loss={vloss:.4f}, test_acc={tacc*100:.2f}%")
            if (best is None) or (vloss<best[0]-1e-12) or (abs(vloss-best[0])<1e-12 and tacc>best[1]):
                best=(vloss,tacc,wd,it)
    print(f"** Best: wd={best[2]}, it={best[3]} | val_loss={best[0]:.4f}, test_acc={best[1]*100:.2f}%")







