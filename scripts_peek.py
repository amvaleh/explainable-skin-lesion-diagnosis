import pandas as pd, numpy as np
from PIL import Image, ImageDraw
R="data/ham10000"
df=pd.read_csv(f"{R}/raw/HAM10000_metadata.tab",sep="\t")
df.columns=[c.strip() for c in df.columns]
for c in df.select_dtypes("object"): df[c]=df[c].str.strip('"')
print(df.shape); print(df.head(3).to_string())
print("\n== class counts ==\n", df.dx.value_counts().to_string())
print("\n== dx_type ==\n", df.dx_type.value_counts().to_string())
print("\n== sex ==\n", df.sex.value_counts().to_string())
print("\n== site (top8) ==\n", df.localization.value_counts().head(8).to_string())
print("\n== age ==\n", df.age.describe().to_string())
print("\nunique lesions:", df.lesion_id.nunique(), "images:", df.image_id.nunique())
print("images per lesion:\n", df.groupby("lesion_id").size().value_counts().sort_index().to_string())

order=["mel","bcc","akiec","nv","bkl","df","vasc"]
NC,TH=6,160; rows=[]
rng=np.random.default_rng(0)
sheet=Image.new("RGB",(NC*TH, len(order)*TH*2+len(order)*26+10),"white")
d=ImageDraw.Draw(sheet); y=0
for dx in order:
    sub=df[df.dx==dx]
    d.text((6,y+7), f"{dx}  (n={len(sub)})", fill="black")
    y+=26
    ids=rng.choice(sub.image_id.values, min(NC,len(sub)), replace=False)
    for i,iid in enumerate(ids):
        im=Image.open(f"{R}/images/{iid}.jpg").resize((TH,TH))
        sheet.paste(im,(i*TH,y))
        mk=Image.open(f"{R}/masks/{iid}_segmentation.png").convert("L").resize((TH,TH))
        ov=Image.composite(Image.new("RGB",(TH,TH),(255,60,60)), im, mk.point(lambda p:120 if p>127 else 0))
        sheet.paste(Image.blend(im,ov,0.55),(i*TH,y+TH))
    y+=TH*2
sheet.save("data/ham10000_peek.png")
print("\nsaved data/ham10000_peek.png", sheet.size)
