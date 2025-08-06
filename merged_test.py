from inference_sdk import InferenceHTTPClient
import cv2

# API Anahtarı
API_KEY = "ZoWplqEklSJHIhSCouku"

# Model ID
MODEL_ID = "merged-project-3-mvzae/1"

# Image Path
IMAGE_PATH = "images/sigara.jpg"

# Modelin orijinalinde class ismi farklı olduğu için değiştiriyoruz
display_mapping = {
    'rokok': 'cigarette'
}

# Roboflow'a bağlanmak için bir istemci (client)
try:
    client = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=API_KEY
    )

    print(f"'{IMAGE_PATH}' resmi analiz ediliyor")
    result = client.infer(IMAGE_PATH, model_id=MODEL_ID)

    # Gelen sonuçları işliyor
    predictions = result.get('predictions', [])

    # Resmi OpenCV ile okuyor
    image = cv2.imread(IMAGE_PATH)

    # Resmin başarılı bir şekilde okunduğunu test ediyor
    if image is None:
        raise FileNotFoundError(f"'{IMAGE_PATH}' adında bir resim bulunamadı.")

    # Resmin boyutlarını değişkenlerde tutuyor
    h, w, _ = image.shape

    if not predictions:
        print("\nBu resimde herhangi bir nesne tespit edilemedi.")
    else:
        print("\n--- Tespit Sonuçları ---")
        for pred in predictions:
            # Sınıf adını ve güven skorunu alıyor
            original_class_name = pred['class']
            confidence = pred['confidence']

            # Sınıf adını kendi istediğimiz şekilde değiştiriyoruz
            display_name = display_mapping.get(original_class_name, original_class_name)

            # Roboflow'un verdiği merkez (x,y) ve genişlik/yükseklik değerlerini
            # OpenCV'nin kullandığı sol-üst ve sağ-alt köşe koordinatlarına çeviriyoruz
            x_center = pred['x']
            y_center = pred['y']
            box_width = pred['width']
            box_height = pred['height']

            x_min = int(x_center - box_width / 2)
            y_min = int(y_center - box_height / 2)
            x_max = int(x_center + box_width / 2)
            y_max = int(y_center + box_height / 2)

            # Nesne adı ve güven skorunu yazdırıyor
            print(f"-> Nesne: {display_name}, Güven Skoru: %{confidence * 100:.2f}")



            # GÖRSELLEŞTİRME KISMI
            # Tespit edilen nesnenin etrafına yeşil dikdörtgen çiziyor
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)

            # Nesne adı ve güven skoru kutucukların üstüne yazdırıyor
            label = f"{display_name}: {confidence:.2f}"
            label_position = (x_min, y_min - 10 if y_min > 20 else y_min + 20)
            cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Gösterilecek pencerenin boyutunu %75 olarak ayarlıyoruz
    scale_percent = 50
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    # Yeniden boyutlandırılmış yeni bir resim oluşturuyoruz
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # Sonuçları içeren YENİDEN BOYUTLANDIRILMIŞ resmi yeni bir pencerede gösteriyor
    cv2.imshow("Tespit Sonucu", resized_image)
    cv2.waitKey(0)  # Kullanıcı bir tuşa basana kadar bekliyor
    cv2.destroyAllWindows()  # İşmeler bitince tüm pencereleri kapatıyor

except Exception as e:
    print(f"\n!!! BİR HATA OLUŞTU !!!")
    print(f"Hata Detayı: {e}")
