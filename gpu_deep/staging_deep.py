"""Optional/confirmatory: end-to-end 3D CNN for WITHIN-MCI staging on the imaged
cohort, mirroring the leakage-safe design of fixed_train_cnn_cv + fixed_fuse_and_report.
Tasks:  amyloid_mci (A+/A-, imaged MCI n~204)  |  traj3 (stable/slow/fast, imaged MCI n~191)
Trains CNN (subject-level 5-fold, OOF embeddings), then leakage-safe fold-aligned fusion
vs clinical / FreeSurfer. Writes staging_final_summary.csv. RUN ON GPU."""
import os, json, numpy as np, pandas as pd, torch, monai
from monai.data import Dataset; from monai.networks.nets import resnet18
from monai.transforms import (Compose,LoadImaged,EnsureChannelFirstd,Resized,ScaleIntensityd,
    ConcatItemsd,DeleteItemsd,RandFlipd,RandAffined,RandGaussianNoised,ToTensord)
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline; from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE); D=os.path.join(ROOT,"data")
IMG=(96,96,96); EPOCHS=40; FOLDS=5; PATIENCE=8; LR=1e-4; SEED=42
OUT=os.path.join(HERE,"staging_results"); os.makedirs(OUT,exist_ok=True)
dev="cuda" if torch.cuda.is_available() else "cpu"
TASKS={"amyloid_mci":(["A-","A+"],"amyloid_mci"),"traj3":(["stable","slow","fast"],"traj3")}

def tf(train):
    k=["mwp1","mwp2"]; it=[LoadImaged(k),EnsureChannelFirstd(k),Resized(k,IMG),ScaleIntensityd(k),
        ConcatItemsd(k,"image",dim=0),DeleteItemsd(k)]
    if train: it+=[RandFlipd("image",0.5,0),RandAffined("image",prob=0.3,rotate_range=(.05,)*3,scale_range=(.05,)*3,mode="bilinear"),RandGaussianNoised("image",0.2,0.0,0.02)]
    return Compose(it+[ToTensord("image")])
def recs(df,cls,col):
    ci={c:i for i,c in enumerate(cls)}
    return [{"mwp1":r.path_mwp1,"mwp2":r.path_mwp2,"label":ci[str(getattr(r,col))],"rid":int(r.RID)} for r in df.itertuples()]
def ld(ds,sh): return DataLoader(ds,batch_size=4,shuffle=sh,num_workers=0,pin_memory=(dev=="cuda"))
def infer(model,loader):
    model.eval(); L,P,E,R={},[],[],[]; cap={}
    h=model.fc.register_forward_hook(lambda m,i,o: cap.__setitem__("e",i[0].detach().cpu()))
    labs=[]
    try:
        with torch.no_grad():
            for b in loader:
                x=b["image"].to(dev)
                with torch.autocast(dev,enabled=(dev=="cuda")): lo=model(x)
                P.append(torch.softmax(lo.float(),1).cpu().numpy()); labs.append(b["label"].numpy())
                E.append(cap["e"].numpy()); R+= [int(v) for v in b["rid"].numpy()]
    finally: h.remove()
    return np.concatenate(labs),np.concatenate(P),np.concatenate(E),np.array(R,int)

master=pd.read_csv(os.path.join(D,"master_features.csv"),low_memory=False)
fam=json.load(open(os.path.join(D,"feature_families.json")))
clin=[c for c in fam["demo"]+fam["cognition"] if c in master.columns]
fs=[c for c in fam["freesurfer"] if c in master.columns]
man=pd.read_csv(os.path.join(HERE,"fixed_deep_manifest_staging.csv"))

def pipe(): return Pipeline([("imp",SimpleImputer(strategy="median",keep_empty_features=True)),("sc",StandardScaler()),("clf",LogisticRegression(max_iter=4000,class_weight="balanced",random_state=SEED))])
def auc(y,P,cls):
    if len(cls)==2: return roc_auc_score((y==cls[1]).astype(int),P[:,1])
    return roc_auc_score(label_binarize(y,classes=cls),P,average="macro")

summary=[]
for task,(cls,col) in TASKS.items():
    d=man[man[col].isin(cls)].reset_index(drop=True); y=d[col].astype(str).values
    print(f"\n=== {task} n={len(d)} {pd.Series(y).value_counts().to_dict()} ===",flush=True)
    skf=StratifiedKFold(FOLDS,shuffle=True,random_state=SEED); preds=[]
    for fold,(tr,te) in enumerate(skf.split(d,y)):
        tro,teo=d.iloc[tr].reset_index(drop=True),d.iloc[te].reset_index(drop=True)
        monai.utils.set_determinism(SEED+fold)
        itr,iva=train_test_split(np.arange(len(tro)),test_size=0.15,stratify=tro[col],random_state=SEED+fold)
        model=resnet18(spatial_dims=3,n_input_channels=2,num_classes=len(cls),shortcut_type="B").to(dev)
        cc=np.bincount([recs(tro.iloc[itr],cls,col)[i]["label"] for i in range(len(itr))],minlength=len(cls))
        w=torch.tensor(cc.sum()/(len(cls)*np.maximum(cc,1)),dtype=torch.float32,device=dev)
        opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
        scaler=torch.amp.GradScaler("cuda",enabled=(dev=="cuda")); lossf=torch.nn.CrossEntropyLoss(weight=w)
        dtr=Dataset(recs(tro.iloc[itr],cls,col),tf(True)); dva=Dataset(recs(tro.iloc[iva],cls,col),tf(False))
        best=-1; bs=None; wait=0
        for ep in range(EPOCHS):
            model.train()
            for b in ld(dtr,True):
                x=b["image"].to(dev); yb=b["label"].to(dev); opt.zero_grad(set_to_none=True)
                with torch.autocast(dev,enabled=(dev=="cuda")): loss=lossf(model(x),yb)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            sch.step()
            yl,pp,_,_=infer(model,ld(dva,False)); sc=balanced_accuracy_score(np.array(cls)[yl],np.array(cls)[pp.argmax(1)])
            if sc>best: best=sc; bs={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; wait=0
            else: wait+=1
            if wait>=PATIENCE: break
        model.load_state_dict(bs)
        for split,dd in (("train",tro),("test",teo)):
            yl,pp,ee,rr=infer(model,ld(Dataset(recs(dd,cls,col),tf(False)),False))
            base=pd.DataFrame({"task":task,"fold":fold,"split":split,"RID":rr,"y_true":np.array(cls)[yl]})
            for i,c in enumerate(cls): base[f"p_{c}"]=pp[:,i]
            embdf=pd.DataFrame(ee,columns=[f"emb_{j:03d}" for j in range(ee.shape[1])])
            fr=pd.concat([base.reset_index(drop=True),embdf.reset_index(drop=True)],axis=1)
            preds.append(fr)
        print(f"  fold{fold} done (val bAcc={best:.3f})",flush=True)
    allp=pd.concat(preds,ignore_index=True); emb=[c for c in allp.columns if c.startswith("emb_")]
    # fusion per fold
    rows=[]
    for fold in range(FOLDS):
        f=allp[allp.fold==fold]; trd=f[f.split=="train"]; ted=f[f.split=="test"]
        tr=master.merge(trd,on="RID"); te=master.merge(ted,on="RID")
        yt=tr[col if col in tr else "y_true"].astype(str).values if False else tr["y_true"].astype(str).values
        yte=te["y_true"].astype(str).values
        sets={"clinical":clin,"freesurfer":fs,"clinical+deep":clin+emb,"all":clin+fs+emb}
        # deep_cnn = CNN probs directly
        rows.append(("deep_cnn",te["RID"].values,yte,te[[f"p_{c}" for c in cls]].values))
        for nm,cols in sets.items():
            m=pipe().fit(tr[cols],yt); pr=m.predict_proba(te[cols]); fc=list(m.named_steps["clf"].classes_)
            pr=pr[:,[fc.index(c) for c in cls]]; rows.append((nm,te["RID"].values,yte,pr))
    # aggregate OOF per featureset
    agg={}
    for nm,rid,yv,pv in rows: agg.setdefault(nm,[]).append((rid,yv,pv))
    for nm,parts in agg.items():
        rid=np.concatenate([p[0] for p in parts]); yv=np.concatenate([p[1] for p in parts]); pv=np.vstack([p[2] for p in parts])
        ba=balanced_accuracy_score(yv,np.array(cls)[pv.argmax(1)]); au=auc(yv,pv,cls)
        summary.append(dict(task=task,featureset=nm,bAcc=round(ba,4),AUC=round(au,4),n=len(yv)))
        print(f"  {task:12} {nm:14} bAcc={ba:.3f} AUC={au:.3f} (n={len(yv)})",flush=True)
    allp.to_csv(os.path.join(OUT,f"staging_oof_{task}.csv"),index=False)
pd.DataFrame(summary).to_csv(os.path.join(HERE,"staging_final_summary.csv"),index=False)
print("\nSaved staging_final_summary.csv")
