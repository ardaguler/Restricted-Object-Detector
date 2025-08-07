from ultralytics import YOLO  # Değişti: ultralytics kütüphanesini ekledik
import cv2

LOCAL_MODEL_PATH = "Models/weights.pt"
IMAGE_PATH = "images/knife.jpg"

# Modelde cigarette ismi farklı olduğu için ismini değiştiriyor
display_mapping = {
    'rokok': 'cigarette'
}


try:
    # Modeli lokal dosyadan yüklüyoruz
    print(f"'{LOCAL_MODEL_PATH}' modeli bilgisayarınızdan yükleniyor...")
    model = YOLO(LOCAL_MODEL_PATH)
    print("Model başarıyla yüklendi.")

    # Resmi OpenCV ile okuyor
    image = cv2.imread(IMAGE_PATH)

    # Resmin başarılı bir şekilde okunduğunu test ediyor
    if image is None:
        raise FileNotFoundError(f"'{IMAGE_PATH}' adında bir resim bulunamadı.")

    print(f"'{IMAGE_PATH}' resmi üzerinde nesne tespiti yapılıyor...")
    results = model(image, conf=0.6) # confident score %60'tan fazla olanları gösteriyor
    predictions = results[0].boxes

    if len(predictions) == 0:
        print("\nTesimde herhangi bir tespit yapılamadı.")
    else:
        print("\n--- Tespit Sonuçları ---")
        for box in predictions:
            # Sınıf adını ve güven skorunu alıyor
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            original_class_name = model.names[class_id]

            # Sınıf adını kendi istediğimiz şekilde değiştiriyor
            display_name = display_mapping.get(original_class_name, original_class_name)

            # Kutu koordinatlarını alıyor (xyxy formatında)
            x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy().astype(int)

            # Nesne adı ve güven skorunu yazdırıyor
            print(f"-> Nesne: {display_name}, Güven Skoru: %{confidence * 100:.2f}")

            # GÖRSELLEŞTİRME KISMI
            # Tespit edilen nesnenin etrafına yeşil dikdörtgen çiziyor
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)

            # Nesne adı ve güven skoru kutucukların üstüne yazdırıyor
            label = f"{display_name}: {confidence:.2f}"
            label_position = (x_min, y_min - 10 if y_min > 20 else y_min + 20)
            cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Gösterilecek pencerenin boyutunu ayarlıyor
    scale_percent = 50
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    # Yeniden boyutlandırılmış yeni bir resim oluşturuyor
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # Sonuçlar pencerede gösteriliyor
    cv2.imshow("Tespit Sonucu", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print(f"\n!!! BİR HATA OLUŞTU !!!")
    print(f"Hata Detayı: {e}")