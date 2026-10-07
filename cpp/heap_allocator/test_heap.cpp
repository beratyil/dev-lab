// test_heap.cpp — doğruluk testleri
// Derleme: g++ -std=c++11 -O2 -Wall -Wextra test_heap.cpp -o test_heap
#include "heap.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

int g_failures = 0;
#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("  HATA %s:%d  %s\n", __FILE__, __LINE__, #cond);    \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

constexpr std::size_t kPoolBytes = 1u << 20;  // 1 MB
unsigned char* g_pool = nullptr;              // bir kez new, hiç delete yok

bool aligned(void* p) { return (reinterpret_cast<std::uintptr_t>(p) % mem::kAlign) == 0; }

// Her şey iade edilmiş mi: tek boş blok, kullanım sıfır, yapı tutarlı
template <class H>
bool is_pristine(const H& h) {
    typename H::Stats s;
    return h.check(&s) && s.used == 0 && s.used_blocks == 0 && s.free_blocks == 1;
}

template <class H>
void test_basic() {
    H h(g_pool, kPoolBytes);
    CHECK(h.capacity() > 0);
    CHECK(is_pristine(h));
    void* p = h.allocate(100);
    CHECK(p && aligned(p));
    std::memset(p, 0xAB, 100);
    CHECK(h.used_bytes() >= 100);
    CHECK(h.check());
    h.deallocate(p);
    CHECK(is_pristine(h));

    void* z = h.allocate(0);
    CHECK(z != nullptr);
    h.deallocate(z);
    CHECK(h.allocate(kPoolBytes) == nullptr);  // sığmaz
    CHECK(h.allocate(~std::size_t(0)) == nullptr);  // taşma koruması
    CHECK(is_pristine(h));
}

template <class H>
void test_coalesce() {
    H h(g_pool, kPoolBytes);
    void* a = h.allocate(64);
    void* b = h.allocate(64);
    void* c = h.allocate(64);
    h.deallocate(a);
    h.deallocate(c);  // c, kalan büyük blokla birleşir; b ortada dolu
    typename H::Stats s;
    CHECK(h.check(&s));
    CHECK(s.free_blocks == 2);
    h.deallocate(b);  // hepsi tek bloğa birleşmeli
    CHECK(is_pristine(h));
}

template <class H>
void test_exhaust() {
    H h(g_pool, kPoolBytes);
    std::vector<void*> v;
    for (;;) {
        void* p = h.allocate(64);
        if (!p) break;
        CHECK(aligned(p));
        v.push_back(p);
    }
    CHECK(!v.empty());
    CHECK(h.free_bytes() < mem::align_up(64 + mem::kHeaderSize));  // gerçekten dolu
    CHECK(h.check());
    std::mt19937 rng(1);
    std::shuffle(v.begin(), v.end(), rng);
    for (void* p : v) h.deallocate(p);
    CHECK(is_pristine(h));
}

template <class H>
void test_invalid_free() {
    H h(g_pool, kPoolBytes);
    void* p = h.allocate(32);
    void* q = h.allocate(32);  // p'nin birleşmemesi için arada dolu blok
    h.deallocate(p);
    h.deallocate(p);  // çift free: yok sayılmalı
    int local = 0;
    h.deallocate(&local);  // havuza ait olmayan adres
    CHECK(h.invalid_frees() == 2);
    CHECK(h.check());
    h.deallocate(q);
    CHECK(is_pristine(h));
}

struct Tracked {
    static int alive;
    int    a;
    double b;
    Tracked(int x, double y) : a(x), b(y) { ++alive; }
    ~Tracked() { --alive; }
};
int Tracked::alive = 0;

template <class H>
void test_objects() {
    H h(g_pool, kPoolBytes);
    Tracked* t = mem::make<Tracked>(h, 7, 2.5);
    CHECK(t && aligned(t) && t->a == 7 && t->b == 2.5 && Tracked::alive == 1);
    mem::destroy(h, t);
    CHECK(Tracked::alive == 0);
    CHECK(is_pristine(h));
}

// Rastgele al/bırak; her bloğa etiket baytı yazılır, iade edilirken kontrol edilir.
// Bloklar çakışırsa etiket bozulur. Havuz sık sık dolar: dolu durum da sınanır.
template <class H>
void test_random_stress() {
    H h(g_pool, kPoolBytes);
    struct Item { unsigned char* p; std::size_t n; unsigned char tag; };
    std::vector<Item> live;
    std::mt19937 rng(42);
    unsigned long fails = 0;
    for (int i = 0; i < 300000; ++i) {
        const bool do_alloc = live.empty() || (rng() % 100) < 55;
        if (do_alloc) {
            const std::size_t n = 1 + rng() % 4096;
            unsigned char* p = static_cast<unsigned char*>(h.allocate(n));
            if (!p) { ++fails; continue; }
            CHECK(aligned(p));
            const unsigned char tag = static_cast<unsigned char>(rng());
            std::memset(p, tag, n);
            live.push_back(Item{p, n, tag});
        } else {
            const std::size_t k = rng() % live.size();
            const Item it = live[k];
            live[k] = live.back();
            live.pop_back();
            bool intact = true;
            for (std::size_t j = 0; j < it.n; ++j)
                if (it.p[j] != it.tag) { intact = false; break; }
            CHECK(intact);
            h.deallocate(it.p);
        }
        if (i % 5000 == 0) CHECK(h.check());
    }
    CHECK(fails > 0);  // havuz gerçekten doluya dayandı mı
    for (std::size_t k = 0; k < live.size(); ++k) h.deallocate(live[k].p);
    CHECK(is_pristine(h));
}

template <class H>
void run_all(const char* name) {
    const int before = g_failures;
    test_basic<H>();
    test_coalesce<H>();
    test_exhaust<H>();
    test_invalid_free<H>();
    test_objects<H>();
    test_random_stress<H>();
    std::printf("%-14s %s\n", name, g_failures == before ? "TAMAM" : "HATALI");
}

}  // namespace

int main() {
    g_pool = new unsigned char[kPoolBytes];
    run_all<mem::FirstFitHeap>("FirstFitHeap");
    run_all<mem::TlsfHeap>("TlsfHeap");
    std::printf("%s\n", g_failures == 0 ? "Tum testler gecti." : "Hata var!");
    return g_failures == 0 ? 0 : 1;
}
