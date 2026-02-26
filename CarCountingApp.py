import cv2              # OpenCV kütüphanesi (görüntü işleme)
import numpy as np      # NumPy (matematiksel işlemler)


# Video dosyasını aç
vid = cv2.VideoCapture("files/traffic.avi")

# Arka plan çıkarıcı oluştur (KNN algoritması kullanır)
# Parametreler:
# history = 200 → Arka plan modelinin kaç kare geçmişe bakacağı
# dist2Threshold = 200 → Pikselin arka plan olup olmadığını belirleyen eşik
# detectShadows = False → Gölge tespiti yapmasın
backsub = cv2.createBackgroundSubtractorKNN(200, 200, False)

c = 0   # Araç sayacı


# Sonsuz döngü (video kare kare okunur)
while True:
    ret, frame = vid.read()  # Videodan bir kare oku

    if ret:  # Eğer kare başarıyla okunduysa

        # Arka plan çıkarma uygula
        # Hareket eden nesneler beyaz (foreground), arka plan siyah olur
        fgmask = backsub.apply(frame)

        # Sayım için iki adet dikey çizgi çiz
        cv2.line(frame, (50, 0), (50, 300), (0, 255, 0), 2)
        cv2.line(frame, (70, 0), (70, 300), (0, 255, 0), 2)


        # Maskedeki konturları bul
        contours, hierarchy = cv2.findContours(
            fgmask,                 # Kaynak görüntü
            cv2.RETR_TREE,          # Hiyerarşi modunu belirler
            cv2.CHAIN_APPROX_SIMPLE # Kontur noktalarını sadeleştirir
        )

        # Hiyerarşi bazen None dönebilir, hata önleme
        try:
            hierarchy = hierarchy[0]
        except:
            hierarchy = []


        # Her konturu sırayla incele
        for contour, hier in zip(contours, hierarchy):

            # Konturun etrafına dikdörtgen çizmek için bounding box al
            (x, y, w, h) = cv2.boundingRect(contour)

            # Küçük gürültüleri filtrele (çok küçük nesneleri sayma)
            if w > 40 and h > 40:

                # Aracın etrafına mavi dikdörtgen çiz
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

                # Eğer araç sayım çizgileri arasından geçiyorsa sayacı artır
                if x > 50 and x < 70:
                    c += 1


        # Ekrana araç sayısını yaz
        cv2.putText(
            frame,
            "Car: " + str(c),
            (90, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        # Görüntüleri göster
        cv2.imshow("Car Counter", frame)
        cv2.imshow("Mask", fgmask)

        # q tuşuna basılırsa çık
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break


# Video kaynağını serbest bırak
vid.release()

# Açık pencereleri kapat
cv2.destroyAllWindows()