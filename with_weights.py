from ultralytics import YOLO  # Değişti: ultralytics kütüphanesini ekledik
import cv2

# --- AYARLAR ---
# API Anahtarı ve Model ID'ye artık ihtiyacımız yok. Onların yerine lokal modelin yolu geldi.
LOCAL_MODEL_PATH = "Models/weights.pt"
IMAGE_PATH = "images/knife_and_cigarette.jpg"

# Bu kısım aynı kaldı
display_mapping = {
    'rokok': 'cigarette'
}


try:
    # +++ BU KISIM EKLENDİ (Lokal model ile ilgili) +++
    # Modeli lokal dosyadan yüklüyoruz
    print(f"'{LOCAL_MODEL_PATH}' modeli bilgisayarınızdan yükleniyor...")
    model = YOLO(LOCAL_MODEL_PATH)
    print("Model başarıyla yüklendi.")
    # +++ EKLENEN KISMIN SONU +++


    # --- BU KISIM SİLİNDİ (API ile ilgiliydi) ---
    # client = InferenceHTTPClient(...)
    # result = client.infer(...)
    # predictions = result.get('predictions', [])
    # --- SİLİNEN KISMIN SONU ---


    # Resmi OpenCV ile okuyor (Bu satır aynı kaldı)
    image = cv2.imread(IMAGE_PATH)

    # Resmin başarılı bir şekilde okunduğunu test ediyor (Bu kısım aynı kaldı)
    if image is None:
        raise FileNotFoundError(f"'{IMAGE_PATH}' adında bir resim bulunamadı.")

    # +++ BU KISIM EKLENDİ (Lokal model ile tespit) +++
    print(f"'{IMAGE_PATH}' resmi üzerinde nesne tespiti yapılıyor...")
    results = model(image)
    predictions = results[0].boxes
    print("Tespit işlemi tamamlandı.")
    # +++ EKLENEN KISMIN SONU +++


    if len(predictions) == 0:
        print("\nBu resimde herhangi bir nesne tespit edilemedi.")
    else:
        print("\n--- Tespit Sonuçları ---")
        # Döngü, yeni 'predictions' nesnesine göre güncellendi
        for box in predictions:
            # Sınıf adını ve güven skorunu alıyor
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            original_class_name = model.names[class_id]

            # Sınıf adını kendi istediğimiz şekilde değiştiriyoruz (Bu mantık aynı kaldı)
            display_name = display_mapping.get(original_class_name, original_class_name)

            # Kutu koordinatlarını alıyor (xyxy formatından direkt olarak)
            x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy().astype(int)

            # Nesne adı ve güven skorunu yazdırıyor (Bu satır aynı kaldı)
            print(f"-> Nesne: {display_name}, Güven Skoru: %{confidence * 100:.2f}")

            # GÖRSELLEŞTİRME KISMI (BU BÖLÜM HİÇ DEĞİŞMEDİ)
            # Tespit edilen nesnenin etrafına yeşil dikdörtgen çiziyor
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)

            # Nesne adı ve güven skoru kutucukların üstüne yazdırıyor
            label = f"{display_name}: {confidence:.2f}"
            label_position = (x_min, y_min - 10 if y_min > 20 else y_min + 20)
            cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Bu bölüm de hiç değişmedi
    # Gösterilecek pencerenin boyutunu %50 olarak ayarlıyoruz
    scale_percent = 50
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    # Yeniden boyutlandırılmış yeni bir resim oluşturuyoruz
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # Sonuçları içeren YENİDEN BOYUTLANDIRILMIŞ resmi yeni bir pencerede gösteriyor
    cv2.imshow("Tespit Sonucu", resized_image)
    cv2.waitKey(0)  # Kullanıcı bir tuşa basana kadar bekliyor
    cv2.destroyAllWindows()  # İşlemler bitince tüm pencereleri kapatıyor

except Exception as e:
    print(f"\n!!! BİR HATA OLUŞTU !!!")
    print(f"Hata Detayı: {e}")