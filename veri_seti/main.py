import json
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

class BilgiDogrulamaModeli:
    def __init__(self):
        # modelin kalıcı ayarları
        self.gereksiz_kelimeler = ["bu", "ve", "ile", "için", "bir", "çok", "lütfen", "herkes"]
        self.vectorizer = CountVectorizer()
        self.ai_modeli = LogisticRegression()
        self.egitildi_mi = False

    def metin_temizle(self, metin):
        metin = metin.lower()
        metin = "".join([karakter for karakter in metin if not karakter.isdigit()])
        metin = "".join([karakter for karakter in metin if karakter not in string.punctuation])
        
        kelimeler = metin.split()
        anlamli_kelimeler = [kelime for kelime in kelimeler if kelime not in self.gereksiz_kelimeler]
        
        # listeyi birleştir
        return " ".join(anlamli_kelimeler) 

    def json_verisi_yukle(self, dosya_yolu):
        # JSON dosyasını okuyan fonksiyon
        with open(dosya_yolu, "r", encoding="utf-8") as dosya:
            return json.load(dosya)

    def modeli_egit(self, dosya_yolu):
        # veriyi yükle ve listelere ayır
        veri_seti = self.json_verisi_yukle(dosya_yolu)
        
        temiz_veriler = []
        etiketler = []

        for veri in veri_seti:
            temiz_cumle = self.metin_temizle(veri["metin"])
            temiz_veriler.append(temiz_cumle)
            etiketler.append(veri["etiket"])

        # Kelimeleri sayılara çevir ve modeli eğit
        vektorler = self.vectorizer.fit_transform(temiz_veriler)
        self.ai_modeli.fit(vektorler, etiketler)
        self.egitildi_mi = True
        print("Yapay zeka modeli JSON verisiyle başarıyla eğitildi!")

    def tahmin_et(self, mesaj):
        # yeni mesajı test et
        if not self.egitildi_mi:
            return "Hata: Model henüz eğitilmedi!"
        
        temiz_mesaj = self.metin_temizle(mesaj)
        yeni_vektor = self.vectorizer.transform([temiz_mesaj])
        tahmin = self.ai_modeli.predict(yeni_vektor)[0]
        
        if tahmin == 1:
            return "BİLGİ KİRLİLİĞİ (SAHTE) olabilir."
        else:
            return "GERÇEK (DOĞRULANMIŞ) görünüyor."

if __name__ == "__main__":
    # model nesnesi oluştur
    afet_modeli = BilgiDogrulamaModeli()
    
    afet_modeli.modeli_egit("veri_seti.json")
    
    test_mesaji = "AFAD duyurusu: ŞOK İDDİA baraj patladı acil yayalım!!!"
    print(f"\nTest edilen mesaj: {test_mesaji}")
    
    sonuc = afet_modeli.tahmin_et(test_mesaji)
    print(f"YAPAY ZEKA KARARI: {sonuc}")