#!/usr/bin/env python
# coding: utf-8

# ## 1. Import package

# In[1]:


import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import random
import scipy.optimize as opt
import scipy.linalg as sla
from sklearn.model_selection import StratifiedKFold


# ## 2.Data loading and processing

# In[2]:


@dataclass
class Dataset:
    Xtr: np.ndarray  # noisy train+val 
    Str: np.ndarray  # noisy train+val 
    Xts: np.ndarray  # clean test 
    Yts: np.ndarray  # clean test

def load_npz(path): 
    d=np.load(path); 
    return Dataset(d['Xtr'],d['Str'],d['Xts'],d['Yts'])

#Other pipeline funcion define
def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)

def one_hot(y: np.ndarray, C: int) -> np.ndarray:
    oh = np.zeros((y.shape[0], C), dtype=np.float32)
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh

def softmax(z: np.ndarray) -> np.ndarray:
    # stablize the value
    z = z - z.max(axis=1, keepdims=True)
    expz = np.exp(z)
    return expz / (expz.sum(axis=1, keepdims=True) + 1e-12)

def accuracy(y, yhat): return float((y==yhat).mean())


#retun T
def get_T(name: str) -> np.ndarray:
    name = name.lower()
    if "0.3" in name:
        T = np.array([[0.7, 0.3, 0.0],
                      [0.0, 0.7, 0.3],
                      [0.3, 0.0, 0.7]], dtype=np.float32)
    elif "0.6" in name:
        T = np.array([[0.4, 0.3, 0.3],
                      [0.3, 0.4, 0.3],
                      [0.3, 0.3, 0.4]], dtype=np.float32)
    else:
        raise ValueError("Unknown dataset: only 0.3 and 0.6 are supported here.")
    return T

# split the training set
def split(X,y,ratio=0.2,seed=42):
    set_seed(seed); 
    n=X.shape[0]; 
    idx=np.random.permutation(n); 
    sp=int(n*(1-ratio))
    return X[idx[:sp]],y[idx[:sp]],X[idx[sp:]],y[idx[sp:]]


# ## 3.Model Building

# ### 3.1 Model Prepocess

# In[3]:


@dataclass
class Standardizer:
    mean: np.ndarray; std: np.ndarray
    def transform(self,Xf):
        Xn=(Xf-self.mean)/self.std
        bias=np.ones((Xn.shape[0],1),dtype=np.float64)
        return np.hstack([Xn,bias])

def flatten(X): 
    return X.reshape(X.shape[0],-1).astype(np.float64)
def fit_std(Xtrf): 
    m=Xtrf.mean(0,keepdims=True); 
    s=Xtrf.std(0,keepdims=True)+1e-5; 
    return Standardizer(m,s)


# ### 3.2 Forward part

# **Forward Method:**  
# Multiply the *clean prediction distribution* \( p_{\text{clean}} \) by \( T^\top \),  
# to obtain the predicted distribution of noisy labels:  
# $$
# p_{\tilde{y}} = T^\top \, p_{\text{clean}}
# $$
# Then compute the cross-entropy loss with respect to the noisy labels.

# In[4]:


def forward_loss(p_clean, y, T):
    p_tilde = p_clean @ T         
    p_tilde = np.clip(p_tilde, 1e-12, 1.0)
    return float(-np.log(p_tilde[np.arange(y.shape[0]), y]).mean())


# Return $ \frac{\partial L}{\partial z} $ (the gradient with respect to the logits),  
# which can be combined with the linear layer to compute  
# $ \frac{\partial L}{\partial W} = X^\top \frac{\partial L}{\partial z} $.

# In[5]:


def dLdz_forward(p_clean, y, T):
    N, C = p_clean.shape
    Y = one_hot(y, C)
    p_tilde = p_clean @ T
    p_tilde = np.clip(p_tilde, 1e-12, 1.0)
    dL_dp_tilde = -(Y / p_tilde) / N
    dL_dp_clean = dL_dp_tilde @ T.T
    s = (dL_dp_clean * p_clean).sum(axis=1, keepdims=True)
    dL_dz = p_clean * (dL_dp_clean - s)
    return dL_dz


# ### 3.3 Multiclass Logistic Regression (Softmax Regression)

# This part applies a **softmax** function on the linear outputs (logits) to produce  
# a probability distribution over all classes:
# 
# $ p(y = c \mid x) = \dfrac{\exp(z_c)}{\sum_{k=1}^{C} \exp(z_k)} $,  
# where $ z = XW + b $ are the logits.
# 
# The model is trained by minimizing the **cross-entropy loss** between the predicted  
# distribution $p(y \mid x)$ and the true labels.

# In[6]:


@dataclass
class FwdCfg: wd:float=1e-4; max_iter:int=300; seed:int=42
class SoftmaxFwd:
    def __init__(self, D, C, T, cfg:FwdCfg):
        self.D, self.C, self.T, self.cfg = D, C, T, cfg
        set_seed(cfg.seed)
        self.W = (0.01*np.random.randn(D,C)).astype(np.float64)
    def _fun(self,w,X,y):
        W=w.reshape(self.D,self.C); p=softmax(X@W)
        base=forward_loss(p,y,self.T)
        reg =0.5*self.cfg.wd*(sla.norm(W,'fro')**2)
        dLdz=dLdz_forward(p,y,self.T)
        grad = X.T@dLdz + self.cfg.wd*W
        return base+reg, grad.reshape(-1)
    def fit(self,X,y):
        res=opt.minimize(self._fun,self.W.reshape(-1),args=(X,y),method="L-BFGS-B",jac=True,
                         options={"maxiter":self.cfg.max_iter})  
        self.W=res.x.reshape(self.D,self.C); return res
    def predict_proba(self,X): return softmax(X@self.W)
    def predict(self,X): return self.predict_proba(X).argmax(1)
       


# ## 4. Hyper-parameter Tunning-Cross Validation

# In[13]:


@dataclass
class Result:
    cfg: "FwdCfg"
    val_loss: float
    test_acc: float

#10 fold cross-validation for model selection
def search_with_direction_cv(Xtr, Str, Xts, Yts, T, seed=42, n_splits=10) -> Tuple[Result, List[Result]]:
    # Flatten
    Xtr_f = flatten(Xtr)
    Xts_f = flatten(Xts)
    C = T.shape[0]
    D = Xtr_f.shape[1]

    # Grid serach
    grid = [
        FwdCfg(wd=1e-4, max_iter=500, seed=seed),
        FwdCfg(wd=1e-3, max_iter=500, seed=seed),
        FwdCfg(wd=1e-2, max_iter=500, seed=seed),
        FwdCfg(wd=5e-2, max_iter=500, seed=seed),
        FwdCfg(wd=1e-1, max_iter=500, seed=seed),
        FwdCfg(wd=5e-1, max_iter=500, seed=seed),
    ]

    # StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    cfg_cv_losses = []   # [(cfg, mean_val_loss)]
    for cfg in grid:
        fold_losses = []
        for tr_idx, val_idx in skf.split(Xtr_f, Str):
            X_tr_raw, y_tr = Xtr_f[tr_idx], Str[tr_idx]
            X_val_raw, y_val = Xtr_f[val_idx], Str[val_idx]

            # Fit standardizer on training fold only to avoid leakage
            std = fit_std(X_tr_raw)
            X_tr = std.transform(X_tr_raw)
            X_val = std.transform(X_val_raw)

            # Train and validate on this fold
            model = SoftmaxFwd(D, C, T, cfg)
            model.fit(X_tr, y_tr)

            p_val = model.predict_proba(X_val)
            vloss = forward_loss(p_val, y_val, T) + 0.5 * cfg.wd * (sla.norm(model.W, 'fro') ** 2)
            fold_losses.append(vloss)

        cfg_cv_losses.append((cfg, float(np.mean(fold_losses))))

    # Select the cfg with the lowest mean validation loss
    best_cfg, best_cv_loss = min(cfg_cv_losses, key=lambda x: x[1])

    std_full = fit_std(Xtr_f)
    X_tr_full = std_full.transform(Xtr_f)
    X_te = std_full.transform(Xts_f)

    best_model = SoftmaxFwd(D, C, T, best_cfg)
    best_model.fit(X_tr_full, Str)

    test_acc = accuracy(Yts, best_model.predict(X_te))

    results: List[Result] = [
        Result(cfg=cfg, val_loss=mean_vloss, test_acc=test_acc if cfg is best_cfg else np.nan)
        for cfg, mean_vloss in cfg_cv_losses
    ]

    best = Result(cfg=best_cfg, val_loss=best_cv_loss, test_acc=test_acc)
    return best, results


# ## 5. Model-selection

# ### 5.1  Dataset with known flip rates (FashionMNIST)

# In[14]:


def main():
    set_seed(42)
    datasets = [
        "datasets/FashionMNIST0.3.npz",
        "datasets/FashionMNIST0.6.npz",
    ]
    for path in datasets:
        print(f"\n==== Dataset: {path} ====")
        T = get_T(path)
        data = load_npz(path)
        best, results = search_with_direction(data.Xtr, data.Str, data.Xts, data.Yts, T, seed=42)

        def fmt(r:Result):
            return f"[p@T] wd={r.cfg.wd}, it={r.cfg.max_iter} | val_loss={r.val_loss:.4f}, test_acc={r.test_acc*100:.2f}%"

        print("Tuned configs (both orientations):")
        for r in results: print(" ", fmt(r))
        print("** Best:", fmt(best))

if __name__ == "__main__":
    main()


# ### 5.2. Dataset with unknown flip rates (CIFAR)

# In[10]:


@dataclass
class FwdCfg:
    wd: float = 0.05
    max_iter: int = 500
    seed: int = 42


def run_forward_correction_experiment():
    # ==== CIFAR (ForwardCorrection, softmax) - 10-fold CV on wd (fixed dims) ====
    T_CIFAR = np.array([
        [0.3753, 0.3311, 0.2935],
        [0.3028, 0.3493, 0.3479],
        [0.3277, 0.2998, 0.3725]
    ], dtype=np.float64)
    DATASET_PATH = "datasets/CIFAR.npz"

    # load data
    d = np.load(DATASET_PATH)
    Xtr, Str, Xts, Yts = d["Xtr"], d["Str"], d["Xts"], d["Yts"]

    # flatten
    Xtr_f = flatten(Xtr)
    Xts_f = flatten(Xts)

    C = T_CIFAR.shape[0]
    seed = 42

    # grid search
    grid = [
        FwdCfg(wd=1e-4, max_iter=500, seed=seed),
        FwdCfg(wd=1e-3, max_iter=500, seed=seed),
        FwdCfg(wd=5e-2, max_iter=500, seed=seed),
        FwdCfg(wd=1e-2, max_iter=500, seed=seed),
        FwdCfg(wd=5e-1, max_iter=500, seed=seed),
        FwdCfg(wd=1e-1, max_iter=500, seed=seed),
        FwdCfg(wd=0.2, max_iter=500, seed=seed),
        FwdCfg(wd=0.5, max_iter=500, seed=seed),
    ]

    print(f"\n==== Dataset: {DATASET_PATH} ====")
    print("Tuned configs with 10-fold CV (forward loss):")

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cv_summaries = []

    for cfg in grid:
        fold_losses = []
        fold_test_accs = []

        for tr_idx, val_idx in skf.split(Xtr_f, Str):
            X_tr_raw, y_tr = Xtr_f[tr_idx], Str[tr_idx]
            X_val_raw, y_val = Xtr_f[val_idx], Str[val_idx]

            std = fit_std(X_tr_raw)
            X_tr = std.transform(X_tr_raw)
            X_val = std.transform(X_val_raw)
            X_te = std.transform(Xts_f)

            D_fold = X_tr.shape[1]

            model = SoftmaxFwd(D_fold, C, T_CIFAR, cfg)
            res = model.fit(X_tr, y_tr)

            p_val = model.predict_proba(X_val)
            vloss = forward_loss(p_val, y_val, T_CIFAR) + 0.5 * cfg.wd * (sla.norm(model.W, "fro") ** 2)
            fold_losses.append(float(vloss))

            y_pred_te = model.predict(X_te)
            test_acc = (y_pred_te == Yts).mean()
            fold_test_accs.append(float(test_acc))

        mean_vloss = float(np.mean(fold_losses))
        std_vloss = float(np.std(fold_losses, ddof=1)) if len(fold_losses) > 1 else 0.0
        mean_tacc = np.mean(fold_test_accs)
        std_tacc = np.std(fold_test_accs, ddof=1)

        cv_summaries.append((cfg, mean_vloss, std_vloss, mean_tacc, std_tacc))
        print(f"  [p@T] wd={cfg.wd:g}, it={cfg.max_iter} | mean_val_loss={mean_vloss:.4f} (±{std_vloss:.4f}), "
              f"mean_test_acc={mean_tacc * 100:.2f}% (±{std_tacc * 100:.2f}%)")

    # Select best cfg
    best_cfg, best_mean_vloss, best_std_vloss, best_mean_tacc, best_std_tacc = \
        min(cv_summaries, key=lambda x: x[1])

    std_full = fit_std(Xtr_f)
    X_tr_full = std_full.transform(Xtr_f)
    X_te = std_full.transform(Xts_f)
    D_full = X_tr_full.shape[1]

    best_model = SoftmaxFwd(D_full, C, T_CIFAR, best_cfg)
    res_full = best_model.fit(X_tr_full, Str)
    test_acc = float((best_model.predict(X_te) == Yts).mean())

    print(f"\n** Best (by 10-fold mean val loss): [p@T] wd={best_cfg.wd:g}, it={best_cfg.max_iter} | "
          f"mean_val_loss={best_mean_vloss:.4f} (±{best_std_vloss:.4f}), "
          f"mean_test_acc={best_mean_tacc * 100:.2f}% (±{best_std_tacc * 100:.2f}%)")
    print(f"Full test accuracy: {test_acc * 100:.2f}%")

    return best_cfg, best_model, test_acc


# ====== Entry point ======
if __name__ == "__main__":
    run_forward_correction_experiment()


# ## 6. Main Execution Function

# In[ ]:


def train_config_forward_once(
    dataset_path: str,
    cfg: FwdCfg,
    tr_idx: np.ndarray = None,
    va_idx: np.ndarray = None,
    seed: int = 42,
):

    set_seed(seed)

    # load data
    d = np.load(dataset_path)
    Xtr, Str, Xts, Yts = d["Xtr"], d["Str"], d["Xts"], d["Yts"]

    # import T
    if "FashionMNIST0.3" in dataset_path:
        T = np.array([
            [0.7, 0.3, 0.0],
            [0.0, 0.7, 0.3],
            [0.3, 0.0, 0.7]
        ], dtype=np.float32)
    elif "FashionMNIST0.6" in dataset_path:
        T = np.array([
            [0.4, 0.3, 0.3],
            [0.3, 0.4, 0.3],
            [0.3, 0.3, 0.4]
        ], dtype=np.float32)
    elif "CIFAR" in dataset_path:
        T = np.array([
            [0.3753, 0.3311, 0.2935],
            [0.3028, 0.3493, 0.3479],
            [0.3277, 0.2998, 0.3725]
        ], dtype=np.float64)
    else:
        raise ValueError(f"Unrecognized dataset_path: {dataset_path}")

    C = T.shape[0]

    # split the data if no idx input
    if tr_idx is None or va_idx is None:
        from sklearn.model_selection import train_test_split
        tr_idx, va_idx = train_test_split(
            np.arange(len(Str)), test_size=0.1, stratify=Str, random_state=seed
        )

    # split the data base on idx
    X_train, y_train = Xtr[tr_idx], Str[tr_idx]
    X_val, y_val = Xtr[va_idx], Str[va_idx]

    # ==== flatten & standardization ====
    X_train_f = flatten(X_train)
    X_val_f   = flatten(X_val)
    X_test_f  = flatten(Xts)

    std = fit_std(X_train_f)
    X_train_std = std.transform(X_train_f)
    X_val_std   = std.transform(X_val_f)
    X_test_std  = std.transform(X_test_f)

    D = X_train_std.shape[1]

    # Training
    model = SoftmaxFwd(D, C, T, cfg)
    res = model.fit(X_train_std, y_train)

    # Validation
    p_val = model.predict_proba(X_val_std)
    base_loss = forward_loss(p_val, y_val, T)
    reg_loss  = 0.5 * cfg.wd * (sla.norm(model.W, 'fro') ** 2)
    val_loss  = base_loss + reg_loss
    val_acc   = accuracy(y_val, model.predict(X_val_std))

    # testing
    test_acc  = accuracy(Yts, model.predict(X_test_std))

    return val_loss, val_acc, test_acc, model, std


