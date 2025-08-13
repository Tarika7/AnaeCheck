import pandas as pd
import os
import shutil
from PIL import Image, UnidentifiedImageError

# --- Paths ---
excel_path = r"C:\Users\tarik\Desktop\AnaeCheck\Dataset\data\India\hb.xlsx"
source_root = r"C:\Users\tarik\Desktop\AnaeCheck\Dataset\data\India"
destination_root = r"C:\Users\tarik\Desktop\AnaeCheck\dataset_all"

# --- Load Excel ---
df = pd.read_excel(excel_path)

# --- Create Anemia Category ---
def categorize(hb):
    if hb >= 12:
        return 'Normal'
    elif 10 <= hb < 12:
        return 'Mild'
    else:
        return 'Severe'

df['Anemia_Category'] = df['Hgb'].apply(categorize)

# --- Create destination folders ---
for category in ['Normal','Mild','Severe']:
    os.makedirs(os.path.join(destination_root, category), exist_ok=True)

# --- Copy largest valid image (JPG or PNG) from each patient folder ---
for idx, row in df.iterrows():
    patient_folder = os.path.join(source_root, str(row['Number']))
    if not os.path.exists(patient_folder):
        print(f"Folder not found for patient {row['Number']}")
        continue
    
    # Find all JPG or PNG images
    images = [f for f in os.listdir(patient_folder) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    if not images:
        print(f"No image files found for patient {row['Number']}")
        continue
    
    # Pick the largest valid image by width*height
    largest_img = None
    max_area = 0
    for img_file in images:
        img_path = os.path.join(patient_folder, img_file)
        try:
            with Image.open(img_path) as im:
                area = im.width * im.height
                if area > max_area:
                    max_area = area
                    largest_img = img_file
        except UnidentifiedImageError:
            continue
    
    if largest_img is None:
        print(f"No valid image found for patient {row['Number']}")
        continue
    
    # Copy to destination
    src = os.path.join(patient_folder, largest_img)
    dst = os.path.join(destination_root, row['Anemia_Category'], f"{row['Number']}.jpg")
    shutil.copy(src, dst)
    print(f"Copied patient {row['Number']} to {row['Anemia_Category']}")



