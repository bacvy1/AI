import os
from PIL import Image

def convert_jfif_to_jpg(folder):
    for filename in os.listdir(folder):
        if filename.lower().endswith(".jfif"):
            jfif_path = os.path.join(folder, filename)
            jpg_path = os.path.splitext(jfif_path)[0] + ".jpg"

            try:
                with Image.open(jfif_path) as img:
                    img.convert("RGB").save(jpg_path, "JPEG")
                os.remove(jfif_path)  # xóa file gốc
                print(f"Converted: {filename} -> {os.path.basename(jpg_path)}")
            except Exception as e:
                print(f"Error with {filename}: {e}")

if __name__ == "__main__":
    folder_path = "./dataset/Bac"  # đổi thành thư mục bạn muốn, "." = thư mục hiện tại
    convert_jfif_to_jpg(folder_path)