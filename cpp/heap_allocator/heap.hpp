// heap.hpp
// -----------------------------------------------------------------------------
// Sabit bir bellek havuzu üzerinde çalışan, OS'in delete'ine ihtiyaç duymayan
// heap yöneticisi.
//
// Senaryo: İşletim sistemi "new" veriyor ama "delete" yok. Havuz açılışta BİR KEZ
// new (veya statik dizi) ile alınır; sonraki bütün allocate/deallocate işleri bu
// havuzun içinde döner. OS'e hiçbir şey geri verilmez.
//
//   static unsigned char*  pool = new unsigned char[4u << 20];  // bir kez
//   static mem::TlsfHeap   heap(pool, 4u << 20);
//
//   Foo* f = mem::make<Foo>(heap, 1, 2);   // placement new
//   mem::destroy(heap, f);                 // ~Foo() + havuza iade
//
// İki strateji, aynı blok düzeni; tek fark boş blok aramasıdır:
//   FirstFitHeap : Tek boş liste, ilk uyan blok. Basit ama arama süresi boş
//                  blok sayısıyla büyür, O(n).
//   TlsfHeap     : Two-Level Segregated Fit. Boyut sınıflarına ayrılmış listeler
//                  + bitmap. allocate ve deallocate O(1); süre doluluktan bağımsız.
//
// Blok düzeni (bloklar kAlign = 16 byte hizalı, boyları 16'nın katı):
//   [ size|flags ][ prev_phys ][ payload .............................. ]
//   Blok boşken payload'un başı next_free / prev_free olarak kullanılır.
//   Havuzun sonunda boyu 0 olan, "dolu" işaretli bir sentinel başlık durur;
//   böylece komşu birleştirme (coalescing) ek sınır kontrolü gerektirmez.
//
// Thread-safe değildir; birden fazla görev kullanacaksa çağıran taraf kilitler.
// C++11 ile derlenir.
// -----------------------------------------------------------------------------
#pragma once

#include <cstddef>
#include <cstdint>
#include <new>
#include <utility>

namespace mem {

constexpr std::size_t kAlign = 16;  // payload hizalaması

constexpr std::size_t align_up(std::size_t x) { return (x + kAlign - 1) & ~(kAlign - 1); }

struct BlockHeader {
    std::size_t  size_flags;  // blok boyu (başlık dahil) | bit0 = boş
    BlockHeader* prev_phys;   // bellekte hemen önceki blok (ilk blokta nullptr)
    BlockHeader* next_free;   // yalnızca blok boşken anlamlı
    BlockHeader* prev_free;   // yalnızca blok boşken anlamlı

    static constexpr std::size_t kFreeBit = 1;

    std::size_t size() const    { return size_flags & ~(kAlign - 1); }
    bool        is_free() const { return (size_flags & kFreeBit) != 0; }
    void set_size(std::size_t s) { size_flags = s | (size_flags & kFreeBit); }
    void set_free(bool f) { size_flags = f ? (size_flags | kFreeBit) : (size_flags & ~kFreeBit); }

    BlockHeader* next_phys() const {
        return reinterpret_cast<BlockHeader*>(reinterpret_cast<std::uintptr_t>(this) + size());
    }
};

// Dolu blokta yalnızca size_flags + prev_phys yer kaplar (64-bit'te 16 byte).
constexpr std::size_t kHeaderSize = align_up(offsetof(BlockHeader, next_free));
// Boş blok, liste işaretçilerini taşıyabilmeli.
constexpr std::size_t kMinBlock =
    align_up(sizeof(BlockHeader) > kHeaderSize ? sizeof(BlockHeader) : kHeaderSize + 1);

inline void*        payload_of(BlockHeader* b) { return reinterpret_cast<char*>(b) + kHeaderSize; }
inline BlockHeader* header_of(void* p) { return reinterpret_cast<BlockHeader*>(static_cast<char*>(p) - kHeaderSize); }

namespace detail {
// En anlamlı 1 bitin indeksi (x != 0)
inline unsigned msb_index(std::size_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return 63u - static_cast<unsigned>(__builtin_clzll(static_cast<unsigned long long>(x)));
#else
    unsigned r = 0;
    while (x >>= 1) ++r;
    return r;
#endif
}
// En az anlamlı 1 bitin indeksi (x != 0)
inline unsigned lsb_index(std::uint32_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<unsigned>(__builtin_ctz(x));
#else
    unsigned r = 0;
    while (!(x & 1u)) { x >>= 1; ++r; }
    return r;
#endif
}
}  // namespace detail

// -----------------------------------------------------------------------------
// Strateji 1: First-fit — tek çift yönlü boş liste, LIFO ekleme.
// -----------------------------------------------------------------------------
class FirstFitIndex {
public:
    static constexpr std::size_t kMaxBlockSize = ~std::size_t(0) & ~(kAlign - 1);

    void insert(BlockHeader* b) {
        b->prev_free = nullptr;
        b->next_free = head_;
        if (head_) head_->prev_free = b;
        head_ = b;
    }
    void remove(BlockHeader* b) {
        if (b->prev_free) b->prev_free->next_free = b->next_free;
        else              head_ = b->next_free;
        if (b->next_free) b->next_free->prev_free = b->prev_free;
    }
    BlockHeader* find(std::size_t need) {
        for (BlockHeader* b = head_; b; b = b->next_free) {
            ++steps_;
            if (b->size() >= need) return b;
        }
        return nullptr;
    }
    bool verify(std::size_t* count) const {
        std::size_t n = 0;
        const BlockHeader* prev = nullptr;
        for (const BlockHeader* b = head_; b; b = b->next_free) {
            if (!b->is_free() || b->prev_free != prev) return false;
            prev = b;
            ++n;
        }
        *count = n;
        return true;
    }
    std::uint64_t steps() const { return steps_; }
    void reset_steps() { steps_ = 0; }

private:
    BlockHeader*  head_  = nullptr;
    std::uint64_t steps_ = 0;
};

// -----------------------------------------------------------------------------
// Strateji 2: TLSF — Two-Level Segregated Fit.
//   1. seviye (fl): boyutun 2'nin kuvveti sınıfı
//   2. seviye (sl): her sınıf 32 eşit aralığa bölünür
//   İki bitmap sayesinde uygun liste tek bit işlemiyle bulunur: O(1).
// -----------------------------------------------------------------------------
class TlsfIndex {
public:
    static constexpr unsigned    kSlLog2    = 5;
    static constexpr unsigned    kSlCount   = 1u << kSlLog2;              // 32
    static constexpr unsigned    kAlignLog2 = 4;                          // log2(16)
    static constexpr unsigned    kFlShift   = kSlLog2 + kAlignLog2;       // 9
    static constexpr unsigned    kFlMax     = sizeof(std::size_t) >= 8 ? 32 : 30;
    static constexpr unsigned    kFlCount   = kFlMax - kFlShift + 1;
    static constexpr std::size_t kSmallBlock   = std::size_t(1) << kFlShift;  // 512
    static constexpr std::size_t kMaxBlockSize = (std::size_t(1) << kFlMax) - kAlign;

    TlsfIndex() {
        fl_bitmap_ = 0;
        for (unsigned i = 0; i < kFlCount; ++i) {
            sl_bitmap_[i] = 0;
            for (unsigned j = 0; j < kSlCount; ++j) heads_[i][j] = nullptr;
        }
    }

    void insert(BlockHeader* b) {
        unsigned fl, sl;
        mapping_insert(b->size(), fl, sl);
        BlockHeader* head = heads_[fl][sl];
        b->prev_free = nullptr;
        b->next_free = head;
        if (head) head->prev_free = b;
        heads_[fl][sl] = b;
        fl_bitmap_     |= 1u << fl;
        sl_bitmap_[fl] |= 1u << sl;
    }
    void remove(BlockHeader* b) {
        unsigned fl, sl;
        mapping_insert(b->size(), fl, sl);
        if (b->prev_free) b->prev_free->next_free = b->next_free;
        else              heads_[fl][sl] = b->next_free;
        if (b->next_free) b->next_free->prev_free = b->prev_free;
        if (!heads_[fl][sl]) {
            sl_bitmap_[fl] &= ~(1u << sl);
            if (!sl_bitmap_[fl]) fl_bitmap_ &= ~(1u << fl);
        }
    }
    BlockHeader* find(std::size_t need) {
        ++steps_;
        if (need > kMaxBlockSize) return nullptr;
        unsigned fl, sl;
        mapping_search(need, fl, sl);
        if (fl >= kFlCount) return nullptr;
        std::uint32_t sl_map = sl_bitmap_[fl] & (~0u << sl);
        if (!sl_map) {  // bu sınıfta yok: daha büyük ilk sınıfa geç
            std::uint32_t fl_map = fl_bitmap_ & (~0u << (fl + 1));
            if (!fl_map) return nullptr;
            fl     = detail::lsb_index(fl_map);
            sl_map = sl_bitmap_[fl];
        }
        sl = detail::lsb_index(sl_map);
        return heads_[fl][sl];
    }
    bool verify(std::size_t* count) const {
        std::size_t n = 0;
        for (unsigned i = 0; i < kFlCount; ++i) {
            if (((fl_bitmap_ >> i) & 1u) != (sl_bitmap_[i] ? 1u : 0u)) return false;
            for (unsigned j = 0; j < kSlCount; ++j) {
                const BlockHeader* head = heads_[i][j];
                if (((sl_bitmap_[i] >> j) & 1u) != (head ? 1u : 0u)) return false;
                const BlockHeader* prev = nullptr;
                for (const BlockHeader* b = head; b; b = b->next_free) {
                    unsigned fl, sl;
                    mapping_insert(b->size(), fl, sl);
                    if (!b->is_free() || b->prev_free != prev || fl != i || sl != j) return false;
                    prev = b;
                    ++n;
                }
            }
        }
        *count = n;
        return true;
    }
    std::uint64_t steps() const { return steps_; }
    void reset_steps() { steps_ = 0; }

private:
    // Bloğun ait olduğu liste
    static void mapping_insert(std::size_t size, unsigned& fl, unsigned& sl) {
        if (size < kSmallBlock) {  // küçük bloklar: 16 byte adımlı doğrusal
            fl = 0;
            sl = static_cast<unsigned>(size >> kAlignLog2);
        } else {
            const unsigned f = detail::msb_index(size);
            sl = static_cast<unsigned>(size >> (f - kSlLog2)) ^ kSlCount;
            fl = f - (kFlShift - 1);
        }
    }
    // İstek için aranacak liste: yukarı yuvarlanır ki listedeki HER blok yetsin
    static void mapping_search(std::size_t size, unsigned& fl, unsigned& sl) {
        if (size >= kSmallBlock) size += (std::size_t(1) << (detail::msb_index(size) - kSlLog2)) - 1;
        mapping_insert(size, fl, sl);
    }

    std::uint32_t fl_bitmap_;
    std::uint32_t sl_bitmap_[kFlCount];
    BlockHeader*  heads_[kFlCount][kSlCount];
    std::uint64_t steps_ = 0;
};

static_assert((std::size_t(1) << TlsfIndex::kAlignLog2) == kAlign, "kAlignLog2 kAlign ile uyumsuz");
static_assert(TlsfIndex::kFlCount <= 32, "fl bitmap 32 bit");

// -----------------------------------------------------------------------------
// Ortak heap: blok bölme, birleştirme, istatistik. Arama stratejisi şablonla gelir.
// -----------------------------------------------------------------------------
template <class FreeIndex>
class Heap {
public:
    struct Stats {
        std::size_t capacity;      // bloklara ayrılabilen toplam byte
        std::size_t used;          // dolu blokların toplamı (başlık dahil)
        std::size_t used_blocks;
        std::size_t free_blocks;
        std::size_t largest_free;  // en büyük boş blok (başlık dahil)
    };

    Heap(void* pool, std::size_t bytes) { init(pool, bytes); }
    Heap(const Heap&) = delete;
    Heap& operator=(const Heap&) = delete;

    void* allocate(std::size_t n) {
        if (n == 0) n = 1;
        if (n > kMaxRequest) { ++failed_; return nullptr; }
        std::size_t need = align_up(n + kHeaderSize);
        if (need < kMinBlock) need = kMinBlock;

        BlockHeader* b = index_.find(need);
        if (!b) { ++failed_; return nullptr; }
        index_.remove(b);
        split(b, need);
        b->set_free(false);
        used_ += b->size();
        return payload_of(b);
    }

    void deallocate(void* p) {
        if (!p) return;
        const std::uintptr_t a = reinterpret_cast<std::uintptr_t>(p);
        // Basit koruma (tam garanti değildir): havuz dışı / hizasız adres veya çift free
        if (a < begin_ + kHeaderSize || a >= end_ || ((a - begin_) % kAlign) != 0) { ++invalid_frees_; return; }
        BlockHeader* b = header_of(p);
        if (b->is_free()) { ++invalid_frees_; return; }

        used_ -= b->size();
        b->set_free(true);

        BlockHeader* next = b->next_phys();
        if (next->is_free()) {  // sağ komşuyla birleş
            index_.remove(next);
            b->set_size(b->size() + next->size());
            b->next_phys()->prev_phys = b;
        }
        BlockHeader* prev = b->prev_phys;
        if (prev && prev->is_free()) {  // sol komşuyla birleş
            index_.remove(prev);
            prev->set_size(prev->size() + b->size());
            prev->next_phys()->prev_phys = prev;
            b = prev;
        }
        index_.insert(b);
    }

    std::size_t   capacity() const      { return capacity_; }
    std::size_t   used_bytes() const    { return used_; }
    std::size_t   free_bytes() const    { return capacity_ - used_; }
    std::uint64_t failed_allocs() const { return failed_; }
    std::uint64_t invalid_frees() const { return invalid_frees_; }
    std::uint64_t search_steps() const  { return index_.steps(); }
    void reset_counters() { failed_ = 0; invalid_frees_ = 0; index_.reset_steps(); }

    // Fiziksel blok zincirini ve boş liste yapısını baştan sona doğrular.
    // O(blok sayısı): test/teşhis içindir, sıcak yolda çağırma.
    bool check(Stats* out = nullptr) const {
        Stats s = {capacity_, 0, 0, 0, 0};
        if (!first_) { if (out) *out = s; return capacity_ == 0; }
        const BlockHeader* prev = nullptr;
        bool prev_free = false;
        std::size_t total = 0;
        for (const BlockHeader* b = first_; b != sentinel_; b = b->next_phys()) {
            const std::size_t sz = b->size();
            if (sz < kMinBlock || total + sz > capacity_) return false;  // bozuk boy
            if (b->prev_phys != prev) return false;                       // kopuk zincir
            if (b->is_free()) {
                if (prev_free) return false;                              // birleşmemiş komşular
                ++s.free_blocks;
                if (sz > s.largest_free) s.largest_free = sz;
            } else {
                ++s.used_blocks;
                s.used += sz;
            }
            prev_free = b->is_free();
            total += sz;
            prev = b;
        }
        std::size_t listed = 0;
        const bool ok = sentinel_->prev_phys == prev && total == capacity_ && s.used == used_ &&
                        index_.verify(&listed) && listed == s.free_blocks;
        if (out) *out = s;
        return ok;
    }

private:
    static constexpr std::size_t kMaxRequest = FreeIndex::kMaxBlockSize - kHeaderSize - kAlign;

    void init(void* pool, std::size_t bytes) {
        const std::uintptr_t start = reinterpret_cast<std::uintptr_t>(pool);
        const std::uintptr_t a = (start + kAlign - 1) & ~std::uintptr_t(kAlign - 1);
        const std::uintptr_t e = (start + bytes) & ~std::uintptr_t(kAlign - 1);
        if (!pool || e <= a || e - a < kMinBlock + kHeaderSize) return;  // kullanılamaz: capacity 0

        std::size_t size = static_cast<std::size_t>(e - a) - kHeaderSize;  // sentinel'e yer bırak
        if (size > FreeIndex::kMaxBlockSize) size = FreeIndex::kMaxBlockSize;

        first_ = reinterpret_cast<BlockHeader*>(a);
        first_->size_flags = size | BlockHeader::kFreeBit;
        first_->prev_phys  = nullptr;
        sentinel_ = first_->next_phys();
        sentinel_->size_flags = 0;  // boy 0, dolu: birleştirme burada durur
        sentinel_->prev_phys  = first_;
        begin_    = a;
        end_      = reinterpret_cast<std::uintptr_t>(sentinel_);
        capacity_ = size;
        index_.insert(first_);
    }

    // Bloğun fazlası yeni bir boş blok olabilecek kadar büyükse ikiye böl
    void split(BlockHeader* b, std::size_t need) {
        const std::size_t total = b->size();
        if (total - need < kMinBlock) return;
        BlockHeader* rest = reinterpret_cast<BlockHeader*>(reinterpret_cast<char*>(b) + need);
        rest->size_flags = (total - need) | BlockHeader::kFreeBit;
        rest->prev_phys  = b;
        rest->next_phys()->prev_phys = rest;
        b->set_size(need);
        index_.insert(rest);
    }

    FreeIndex      index_;
    BlockHeader*   first_         = nullptr;
    BlockHeader*   sentinel_      = nullptr;
    std::uintptr_t begin_         = 0;
    std::uintptr_t end_           = 0;
    std::size_t    capacity_      = 0;
    std::size_t    used_          = 0;
    std::uint64_t  failed_        = 0;
    std::uint64_t  invalid_frees_ = 0;
};

using FirstFitHeap = Heap<FirstFitIndex>;
using TlsfHeap     = Heap<TlsfIndex>;

// -----------------------------------------------------------------------------
// Nesne yardımcıları: new/delete yerine bunlar kullanılır.
// (Yapıcı istisna fırlatırsa bellek iade edilmez; -fno-exceptions ortamı varsayılır.)
// -----------------------------------------------------------------------------
template <class T, class H, class... Args>
T* make(H& heap, Args&&... args) {
    static_assert(alignof(T) <= kAlign, "T, heap hizalamasından (16) fazlasını istiyor");
    void* p = heap.allocate(sizeof(T));
    return p ? ::new (p) T(std::forward<Args>(args)...) : nullptr;
}

template <class T, class H>
void destroy(H& heap, T* obj) {
    if (!obj) return;
    obj->~T();
    heap.deallocate(obj);
}

}  // namespace mem
