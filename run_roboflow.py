from roboflow import Roboflow
import cv2

# Roboflow API anahtarı
api_key = "ZoWplqEklSJHIhSCouku"

# Roboflow Universe Model
model_id = "merged-project-3-mvzae"
version_number = 1

test_image_path = "images/smoking.jpg"
scale_percent = 75

# Kodun Çalışma Kısmı

print("Roboflow'a bağlanılıyor...")
# API anahtarınla Roboflow'a bağlan
rf = Roboflow(api_key=api_key)

# Proje bilgilerini kullanarak projeyi seç
project = rf.workspace().project(model_id)
model = project.version(version_number).model

print(f"'{test_image_path}' üzerinde tespit yapılıyor...")

# Modeli çalıştır ve tahminde bulun
prediction = model.predict(test_image_path, confidence=40, overlap=30)
predictions_data = prediction.json()

# Tespit sonuçlarını ekrana yazdır
print("Tespit Sonuçları (JSON):")
print(predictions_data)

# Sadece bir tespit varsa resmi kaydet ve ekranda göster
if len(predictions_data['predictions']) > 0:
    print("\nSigara tespiti yapıldı!")

    # Tespit edilen kutucukların çizildiği resmi kaydet
    saved_image_name = "sonuc.jpg"
    prediction.save(saved_image_name)
    print(f"Sonuçlar '{saved_image_name}' dosyasına kaydedildi.")

    print("Sonuç resmi ekranda gösteriliyor... (Kapatmak için pencereye tıklayıp bir tuşa basın)")

    img = cv2.imread(saved_image_name)

    # Resmi, yukarıda belirtilen yüzdeye göre yeniden boyutlandır
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    new_dimensions = (width, height)
    resized_img = cv2.resize(img, new_dimensions, interpolation=cv2.INTER_AREA)

    # Yeniden boyutlandırılmış resmi yeni bir pencerede göster
    cv2.imshow("Tespit Sonucu", resized_img)

    # Bir tuşa basılana kadar pencerenin açık kalmasını sağlıyor
    cv2.waitKey(0)

    cv2.destroyAllWindows()

else:
    # Eğer hiçbir şey tespit edilmezse bilgilendir
    print("\nResimde sigara tespit edilmedi.")

print("\nİşlem tamam!")