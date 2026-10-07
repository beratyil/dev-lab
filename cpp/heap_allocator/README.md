# heap_allocator — Doluluk artınca heap yavaşlıyor mu?

Sabit bir bellek havuzu üzerinde çalışan, işletim sisteminin `delete`'ine ihtiyaç
duymayan küçük bir heap yöneticisi ve bunun **doluluk oranına göre hız testi**.

Sorulan soru: *Havuz doldukça `allocate` / `deallocate` yavaşlıyor mu?*

Kısa cevap: **Arama stratejisine bağlı.**

- **First-fit** ile evet, ciddi şekilde yavaşlıyor: %50 dolulukta ortalama allocate
  ~400 kat, %75'te ~10.000 kat yavaşlıyor. Ayrıca parçalanma yüzünden havuz ~%92'nin
  üstüne hiç çıkamıyor.
- **TLSF** ile hayır: %10'dan %99.5'e kadar allocate süresi ~14 ns civarında sabit kalıyor,
  her allocate tam 1 arama adımı.

## Dosyalar

| Dosya | İçerik |
|---|---|
| `heap.hpp` | Heap yöneticisi (header-only, C++11). `FirstFitHeap` ve `TlsfHeap`, `mem::make` / `mem::destroy` yardımcıları. |
| `test_heap.cpp` | Doğruluk testleri: iki heap için de temel al/bırak, birleştirme, havuzu tüketme, geçersiz/çift free, nesne yaşam döngüsü ve 300.000 adımlık rastgele stres testi. |
| `bench_heap.cpp` | Doluluk seviyesine göre allocate/deallocate süresini ölçen benchmark. |

## Senaryo

İşletim sistemi `new` veriyor ama `delete` yok (gömülü / RTOS benzeri ortamlar).
Havuz açılışta **bir kez** alınır, sonraki bütün ayırma/iade işlemleri bu havuzun
içinde döner; OS'e hiçbir şey geri verilmez.

```cpp
#include "heap.hpp"

static unsigned char* pool = new unsigned char[4u << 20];  // bir kez
static mem::TlsfHeap   heap(pool, 4u << 20);

Foo* f = mem::make<Foo>(heap, 1, 2);   // allocate + placement new
mem::destroy(heap, f);                 // ~Foo() + havuza iade

void* p = heap.allocate(100);          // ham bellek
heap.deallocate(p);
```

## Tasarım

### Blok düzeni

Bloklar 16 byte hizalı, boyları 16'nın katı:

```
[ size|flags ][ prev_phys ][ payload .............................. ]
```

- `size_flags`: blok boyu (başlık dahil); en düşük bit "boş" bayrağı.
- `prev_phys`: bellekte hemen önceki blok → sol komşuyla O(1) birleştirme.
- Blok boşken payload'un başı `next_free` / `prev_free` olarak kullanılır;
  dolu blokta başlık yalnızca 16 byte (64-bit).
- Havuzun sonunda boyu 0, "dolu" işaretli bir **sentinel** başlık var; böylece sağ
  komşu birleştirmesi ek sınır kontrolü gerektirmiyor.
- `allocate` gerekirse bloğu böler (`split`), `deallocate` iki komşuyla da birleştirir
  (coalescing). Böylece yan yana iki boş blok hiç kalmaz.

### İki strateji (tek fark boş blok araması)

`Heap<FreeIndex>` şablonu bölme/birleştirme/istatistiği ortak yapar; boş blok
araması şablon parametresiyle gelir:

| | `FirstFitHeap` | `TlsfHeap` |
|---|---|---|
| Boş blok yapısı | Tek çift yönlü liste (LIFO ekleme) | Boyut sınıflarına ayrılmış listeler + iki bitmap |
| Arama | Listeyi baştan gez, ilk yeten blok | `fl` (2'nin kuvveti) + `sl` (32 alt aralık) bitmap'lerinde tek bit işlemi |
| Karmaşıklık | O(boş blok sayısı) | O(1) |
| Doluluğa bağımlılık | Var | Yok |

TLSF'de arama sırasında istek boyu bir üst alt-sınıfa yuvarlanır (`mapping_search`),
böylece bulunan listedeki **her** blok isteğe yeter, listeyi gezmeye gerek kalmaz.
512 byte altındaki bloklar 16 byte adımlı doğrusal sınıflarda tutulur.

### Güvenlik ve teşhis

- `deallocate`, havuz dışı / hizasız adresleri ve çift free'yi yakalayıp yok sayar
  (`invalid_frees()` sayacı). Tam garanti değildir, basit bir korumadır.
- `check()` fiziksel blok zincirini ve boş liste/bitmap yapısını baştan sona
  doğrular. O(blok sayısı), sadece test/teşhis için.
- Sayaçlar: `used_bytes()`, `free_bytes()`, `failed_allocs()`, `search_steps()`.
- Thread-safe **değildir**; birden fazla görev kullanacaksa çağıran taraf kilitler.
- `mem::make` yapıcı istisna fırlatırsa belleği iade etmez (`-fno-exceptions`
  ortamı varsayılır). `alignof(T) > 16` olan tipler derleme hatası verir.

## Derleme ve çalıştırma

```bash
# Doğruluk testleri
g++ -std=c++11 -O2 -Wall -Wextra test_heap.cpp -o test_heap
./test_heap

# Benchmark
g++ -std=c++11 -O2 bench_heap.cpp -o bench_heap
./bench_heap                       # varsayılan: 200.000 işlem, 9 doluluk seviyesi
./bench_heap 50000 0.5 0.9 0.99    # [işlem_sayısı] [seviye1 seviye2 ...]
```

> Not: First-fit yüksek doluluklarda çok yavaşladığı için varsayılan ayarlarla tam
> benchmark birkaç dakika (bu makinede ~10 dk) sürebilir. Hızlı bir deneme için
> işlem sayısını düşürün.

## Benchmark yöntemi

Her doluluk hedefi için, iki heap de **aynı tohumla aynı istek dizisini** görür:

1. Taze 16 MB heap, hedef doluluğa kadar doldurulur.
2. **Isınma** (200.000 işlem): doluluk hedefin altındaysa allocate, değilse rastgele
   bir blok bırakılır → gerçekçi parçalanma oluşur.
3. **Ölçüm** (200.000 işlem): aynı döngü, her allocate/deallocate `steady_clock` ile
   tek tek zamanlanır; zamanlayıcının kendi maliyeti (medyan) çıkarılır.

- İstek boyu karışımı: %80 küçük (16–256 B), %15 orta (257–2048 B), %5 büyük (2–16 KB).
- Allocate başarısız olursa "uygulama" rastgele bir bloğu bırakarak yer açar.
- Havuz önceden `memset` ile dokunulur; page fault'lar ölçüme girmez.

Tablo sütunları:

| Sütun | Anlamı |
|---|---|
| Hedef / Gerçek | İstenen doluluk / ölçüm boyunca ortalama gerçek doluluk |
| allocOrt, p50, p99, max | Allocate süresi (ns) |
| freeOrt | Ortalama deallocate süresi (ns) |
| Basarsz | Başarısız allocate oranı |
| adim/alloc | Allocate başına gezilen boş blok sayısı |

## Örnek sonuçlar

`g++ -std=c++11 -O2`, Linux x86-64 (bulut container). Süreler ns. Mutlak değerler
makineye göre değişir; önemli olan doluluğa göre **eğilim**.

```
Hedef   Heap      Gercek  | allocOrt     p50      p99       max | freeOrt | Basarsz | adim/alloc
-------------------------------------------------------------------------------------------
 10.0%  FirstFit    10.0% |       21      12      151     30457 |      44 |   0.00% |       1.5
 10.0%  TLSF        10.0% |       14       9       47     29864 |      16 |   0.00% |       1.0

 25.0%  FirstFit    25.0% |       16       3      179     38010 |      40 |   0.00% |       1.7
 25.0%  TLSF        25.0% |       14       8       54     52065 |      17 |   0.00% |       1.0

 50.0%  FirstFit    50.0% |     6524      10   182451   2811184 |     257 |   3.46% |     345.6
 50.0%  TLSF        50.0% |       15       9       83     38549 |      42 |   0.00% |       1.0

 75.0%  FirstFit    75.0% |   210938      24  1816154  19739274 |     292 |  14.73% |    3340.0
 75.0%  TLSF        75.0% |       14       8       80     34675 |      62 |   0.00% |       1.0

 90.0%  FirstFit    90.0% |   517794      93  1899250  13690512 |     291 |  38.74% |    8685.3
 90.0%  TLSF        90.0% |       14       8       58     28783 |      66 |   0.72% |       1.0

 95.0%  FirstFit    92.0% |   517371     209  1873510  41158700 |     297 |  44.47% |    9276.3
 95.0%  TLSF        95.0% |       14       7       67     27945 |      85 |   1.59% |       1.0

 98.0%  FirstFit    92.0% |   512608     224  1787031 167893721 |     300 |  44.87% |    9359.6
 98.0%  TLSF        98.0% |       13       8       53     24712 |      74 |   2.51% |       1.0

 99.0%  FirstFit    92.0% |   507583     222  1810200  11449310 |     293 |  44.85% |    9295.3
 99.0%  TLSF        99.0% |       15       8       88     45138 |     111 |   3.55% |       1.0

 99.5%  FirstFit    92.1% |   490278     201  1795951  11600039 |     291 |  44.59% |    9215.8
 99.5%  TLSF        99.5% |       17      12      118     50503 |     131 |  24.96% |       1.0
```

## Yorum

**First-fit doluluk arttıkça çöküyor.**

- %25'e kadar hızlı (~20 ns). %50'de ortalama allocate ~6.5 µs, %75'te ~210 µs,
  %90 ve üstünde ~0.5 ms. p99 değerleri ~2 ms'ye çıkıyor.
- Sebep `adim/alloc` sütununda görülüyor: %10'da 1.5 olan arama adımı %90'da
  ~8.700'e çıkıyor. Havuz parçalandıkça boş liste küçük, işe yaramaz bloklarla dolu
  oluyor ve büyük istekler için listenin neredeyse tamamı geziliyor.
- Medyan (p50) düşük kalırken ortalamanın patlaması, maliyetin büyük isteklerde
  yoğunlaştığını gösteriyor: küçük istek listenin başında yer buluyor, büyük istek
  bütün listeyi geziyor (çoğu zaman da sonunda bulamıyor).
- Parçalanma yüzünden havuz **~%92'nin üstüne çıkamıyor**; %95, %98, %99 ve %99.5 hedefleri
  pratikte aynı noktada takılıyor ve allocate'lerin ~%45'i başarısız oluyor.

**TLSF doluluktan bağımsız.**

- Allocate ortalaması %10'dan %99.5'e kadar ~13–17 ns, p50 ~8 ns, p99 < 100 ns.
- Her allocate tam **1** arama adımı: bitmap ile doğru liste doğrudan bulunuyor.
- Parçalanmaya da daha dayanıklı: %99 hedefe gerçekten ulaşabiliyor; başarısızlık
  oranı %99 dolulukta bile ~%3.5. %99.5'te ise kalan boşluk büyük istekler için
  yetmediğinden başarısızlık ~%25'e çıkıyor; yine de süre artmıyor (~17 ns).
- Deallocate süresi dolulukla hafifçe artıyor (~16 → ~130 ns). Algoritma yine O(1);
  artış, birleştirilen komşu bloklara erişimin yüksek dolulukta cache'e daha az
  sığmasından kaynaklanıyor olabilir.

**`max` sütunu hakkında:** iki heap'te de ara sıra on mikrosaniyeler mertebesinde
tekil değerler var. Bunlar algoritmadan değil ortamdan (OS kesmeleri, zamanlayıcı,
context switch) gelen gürültü; TLSF'de adım sayısı sabit 1 olduğu için bu kesin.
Gerçek zamanlı bir sistemde en kötü durum için `max` yerine algoritmik sınıra
(TLSF: sabit sayıda işlem) bakmak gerekir.

**Sonuç:** "Doluluk artınca yavaşlıyor mu?" sorusunun cevabı first-fit için *evet*,
TLSF için *hayır*. Belirlenimci (deterministik) süre gereken ortamlarda TLSF tercih
edilmeli.
