# Heart Disease Detection
Verilen .csv uzantılı verisetinde, yapay sinir ağının kalp hastalığı varlığını belirleyebilmesi için gerekli değerler bulunuyor.
Sinir ağı oluşturmanın en önemli adımlarından biri , işlemleri yapacak fonksiyonların belirlenmesidir. Basit bir veri seti olduğundan aktivasyon fonksiyonlarını relu ve sigmoid
seçtim.

## Sinir ağı oluşturma süreci şöyle : 
1. Kütüphanelerin import edilmesi,
2. Verisetinin yüklenmesi,
3. Verisetinden tahmin yürütülecek olan sütunun ayrılması,
4. Giriş ve çıkış verilerinin eğitim ve test olarak bölünmesi,
5. Yapay sinir ağının oluşumu,
6. Katmanların belirlenmesi,
7. Modeli compile etme,
8. Sinir ağını teste etme,
9. Sonuçların listlenmesi ve görsellerin getirilmesi.

Test ederken Batch Size ve Epoch seçimi önemlidir. Gereksiz boyutlarda belirlenmesi eğitimin süresini arttırır. Epoch sayısının yüksek Batch Size'ın düşük olması bu veri seti için
avantaj sağlar.
