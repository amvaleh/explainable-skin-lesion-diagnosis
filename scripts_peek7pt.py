import pandas as pd
R="data/derm7pt/release_v0"
df=pd.read_csv(f"{R}/meta/meta.csv")
print("cases:",len(df))
CRIT=["pigment_network","streaks","pigmentation","regression_structures",
      "dots_and_globules","blue_whitish_veil","vascular_structures"]
print("\n== diagnosis ==\n", df.diagnosis.value_counts().to_string())
print("\n== seven_point_score ==\n", df.seven_point_score.value_counts().sort_index().to_string())
for c in CRIT:
    print(f"\n== {c} ==\n", df[c].value_counts().to_string())
print("\n== management ==\n", df.management.value_counts().to_string())
print("\n== difficulty ==\n", df.level_of_diagnostic_difficulty.value_counts().to_string())
for s in ["train","valid","test"]:
    print(s, len(pd.read_csv(f"{R}/meta/{s}_indexes.csv")))
print("\nmissing clinic/derm paths:", df.clinic.isna().sum(), df.derm.isna().sum())

# contact sheet: clinical | dermoscopic pairs with labels
from PIL import Image, ImageDraw
import numpy as np
rng=np.random.default_rng(1)
TH=190; ROWS=6
pick=[]
for dx_key in ["melanoma","clark nevus","basal cell carcinoma","seborrheic keratosis","blue nevus","vascular lesion"]:
    sub=df[df.diagnosis.str.contains(dx_key)]
    pick.append(sub.iloc[rng.integers(len(sub))])
sheet=Image.new("RGB",(TH*2+330, ROWS*(TH+8)+8),"white"); d=ImageDraw.Draw(sheet)
for r,row in enumerate(pick):
    y=8+r*(TH+8)
    for i,col in enumerate(["clinic","derm"]):
        im=Image.open(f"{R}/images/{row[col]}").resize((TH,TH)); sheet.paste(im,(i*TH,y))
    d.text((TH*2+8,y+4), f"{row.diagnosis}", fill="black")
    d.text((TH*2+8,y+20), f"7pt score: {row.seven_point_score}  ({row.management})", fill="black")
    for k,c in enumerate(CRIT):
        d.text((TH*2+8,y+38+k*15), f"{c.replace('_',' ')}: {row[c]}", fill="black")
sheet.save("data/derm7pt_peek.png"); print("saved", sheet.size)
