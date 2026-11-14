"""آزمایش حذفی: آیا elevation / location / sex به سقف کارایی اضافه می‌کنند؟"""
import numpy as np, pandas as pd, torch, torch.nn as nn
R="data/derm7pt/release_v0"
df=pd.read_csv(f"{R}/meta/meta.csv")
MERGE={"pigment_network":{"absent":"absent","typical":"typical","atypical":"atypical"},
 "streaks":{"absent":"absent","regular":"regular","irregular":"irregular"},
 "pigmentation":{"absent":"absent","diffuse regular":"regular","localized regular":"regular",
   "diffuse irregular":"irregular","localized irregular":"irregular"},
 "regression_structures":{"absent":"absent","blue areas":"present","white areas":"present","combinations":"present"},
 "dots_and_globules":{"absent":"absent","regular":"regular","irregular":"irregular"},
 "blue_whitish_veil":{"absent":"absent","present":"present"},
 "vascular_structures":{"absent":"absent","comma":"benign","hairpin":"benign","arborizing":"benign",
   "wreath":"benign","dotted":"irregular","linear irregular":"irregular","within regression":"irregular"}}
CRIT=list(MERGE)
for c,m in MERGE.items(): df[c+"_m"]=df[c].map(m)
df["y"]=df.diagnosis.str.contains("melanoma").astype(int)
idx={s:pd.read_csv(f"{R}/meta/{s}_indexes.csv").iloc[:,0].values for s in ["train","valid","test"]}

def run(cols, seeds=5):
    X=pd.get_dummies(df[cols]).astype(np.float32)
    Xt={s:torch.tensor(X.iloc[v].values) for s,v in idx.items()}
    yt={s:torch.tensor(df.y.iloc[v].values) for s,v in idx.items()}
    cnt=torch.bincount(yt["train"]).float(); w=cnt.sum()/(2*cnt)
    out=[]
    for sd in range(seeds):
        torch.manual_seed(sd)
        m=nn.Linear(X.shape[1],2); opt=torch.optim.Adam(m.parameters(),lr=0.05)
        lf=nn.CrossEntropyLoss(weight=w); best=(0,None)
        for e in range(301):
            opt.zero_grad(); l=lf(m(Xt["train"]),yt["train"]); l.backward(); opt.step()
            if e%10==0:
                with torch.no_grad(): va=(m(Xt["valid"]).argmax(1)==yt["valid"]).float().mean().item()
                if va>best[0]: best=(va,{k:v.clone() for k,v in m.state_dict().items()})
        m.load_state_dict(best[1])
        with torch.no_grad(): p=m(Xt["test"]).argmax(1)
        t=yt["test"]
        TP=((p==1)&(t==1)).sum().item(); FN=((p==0)&(t==1)).sum().item()
        TN=((p==0)&(t==0)).sum().item(); FP=((p==1)&(t==0)).sum().item()
        out.append(((TP+TN)/len(t), TP/(TP+FN), TN/(TN+FP)))
    a=np.array(out)
    return X.shape[1], a.mean(0), a.std(0)

SETS={
 "7 concepts (current)":            [c+"_m" for c in CRIT],
 "+ elevation":               [c+"_m" for c in CRIT]+["elevation"],
 "+ elevation + location":    [c+"_m" for c in CRIT]+["elevation","location"],
 "+ elevation + location + sex":[c+"_m" for c in CRIT]+["elevation","location","sex"],
 "metadata only":  ["elevation","location","sex"],
}
print(f"{'feature set':<32}{'cols':>6}{'accuracy':>16}{'recall':>16}{'specificity':>16}")
print("-"*88)
for name,cols in SETS.items():
    n,mu,sd=run(cols)
    print(f"{name:<32}{n:>6}   {mu[0]:.3f}±{sd[0]:.3f}   {mu[1]:.3f}±{sd[1]:.3f}   {mu[2]:.3f}±{sd[2]:.3f}")
