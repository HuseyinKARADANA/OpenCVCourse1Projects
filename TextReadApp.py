from PIL import Image
import pytesseract


img_to_str=pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"
img=Image.open("files/metin.png")
result=pytesseract.image_to_string(img)


with open("text_result.txt",mode="w") as file:
    file.write(result)
    print("ready")