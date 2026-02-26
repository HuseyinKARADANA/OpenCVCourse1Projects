import cv2  # OpenCV kütüphanesi (görüntü işleme için)
import numpy as np  # Sayısal işlemler için NumPy
import pytesseract  # OCR (metin okuma) işlemleri için
import imutils  # Kontur işlemlerini kolaylaştıran yardımcı kütüphane

# Görüntüyü dosyadan oku
img = cv2.imread("files/licence_plate.jpg")

# Görüntüyü gri tonlamaya çevir (kenar ve şekil tespiti için daha uygun)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Gürültüyü azaltmak için bilateral filter uygula
# (Kenarları koruyarak yumuşatma yapar)
filtered = cv2.bilateralFilter(gray, 6, 250, 250)

# Canny algoritması ile kenar tespiti yap
edged = cv2.Canny(filtered, 30, 200)

# Kenar görüntüsündeki konturları bul
contours = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# OpenCV sürüm farklarından dolayı uygun kontur listesini al
cnts = imutils.grab_contours(contours)

# Konturları alanlarına göre büyükten küçüğe sırala
# En büyük 10 tanesini al
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

screen = None  # Plaka konturunu tutacak değişken

# Konturlar arasında 4 köşeli olanı bulmaya çalış
for c in cnts:
    # Konturun çevresini hesapla
    epsilon = 0.018 * cv2.arcLength(c, True)

    # Konturu yaklaşık bir çokgene dönüştür
    approx = cv2.approxPolyDP(c, epsilon, True)

    # Eğer 4 köşeli ise (dikdörtgen/plaka olabilir)
    if len(approx) == 4:
        screen = approx
        break

# Plaka alanını maskelemek için siyah bir görüntü oluştur
mask = np.zeros(gray.shape, np.uint8)

# Bulunan plaka konturunu beyaz olarak çiz (mask üzerinde)
new_img = cv2.drawContours(mask, [screen], 0, (255, 255, 255), -1)

# Orijinal görüntüyü maske ile birleştir (sadece plaka alanı kalır)
new_img = cv2.bitwise_and(img, img, mask=mask)

# Maskede beyaz olan (plaka) piksel koordinatlarını bul
(x, y) = np.where(mask == 255)

# En üst ve en sol koordinatları al
(topx, topy) = (np.min(x), np.min(y))

# En alt ve en sağ koordinatları al
(bottomx, bottomy) = (np.max(x), np.max(y))

# Plaka alanını kırp (crop)
cropped = gray[topx:bottomx + 1, topy:bottomy + 1]

# OCR ile plakadaki yazıyı oku
text = pytesseract.image_to_string(cropped, lang="eng")

print("Detected Text:", text)

# İşlem adımlarını görsel olarak göster
cv2.imshow("1. Original", img)  # Orijinal görüntü
cv2.imshow("2. Gray", gray)  # Gri tonlama
cv2.imshow("3. Filtered", filtered)  # Filtre uygulanmış hali
cv2.imshow("4. Edges", edged)  # Kenar tespiti
cv2.imshow("5. Contour", new_img)  # Plaka maskelenmiş hali
cv2.imshow("6. Cropped", cropped)  # Kırpılmış plaka

# Bir tuşa basılana kadar bekle
cv2.waitKey(0)

# Tüm pencereleri kapat
cv2.destroyAllWindows()