// bench_heap.cpp — doluluk seviyesine göre allocate/deallocate süresi
// Derleme: g++ -std=c++11 -O2 bench_heap.cpp -o bench_heap
//
// Her doluluk hedefi için:
//   1) Taze heap, hedef doluluğa kadar doldurulur.
//   2) Isınma: rastgele al/bırak ile gerçekçi parçalanma oluşturulur.
//      (doluluk < hedef ise allocate, değilse rastgele bir bloğu bırak)
//   3) Ölçüm: aynı döngü, her allocate/deallocate tek tek zamanlanır.
// Allocate başarısız olursa uygulama rastgele bir bloğu bırakır (yer açar).
// İki heap aynı tohumla aynı istek dizisini görür.
#include "heap.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kPoolBytes  = 16u << 20;  // 16 MB
int                   kWarmupOps  = 200000;     // komut satırından değiştirilebilir
int                   kMeasureOps = 200000;

unsigned char* g_pool = nullptr;  // bir kez new, hiç delete yok

inline std::int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}

// İstek boyu karışımı: %80 küçük (16-256), %15 orta (257-2048), %5 büyük (2049-16384)
std::size_t random_size(std::mt19937& rng) {
    const std::uint32_t r = rng() % 100;
    if (r < 80) return 16 + rng() % 241;
    if (r < 95) return 257 + rng() % 1792;
    return 2049 + rng() % 14336;
}

// İki ardışık now() çağrısının medyan maliyeti; ölçümlerden çıkarılır
std::int64_t timer_overhead_ns() {
    std::vector<std::int64_t> v(200000);
    for (std::size_t i = 0; i < v.size(); ++i) {
        const std::int64_t a = now_ns();
        v[i] = now_ns() - a;
    }
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}

struct Result {
    double fill;  // ölçüm sırasındaki ortalama gerçek doluluk
    double alloc_mean, alloc_p50, alloc_p99, alloc_max;
    double free_mean;
    double fail_pct;
    double steps;  // allocate başına boş blok arama adımı
};

template <class H>
Result run(double target, std::int64_t overhead) {
    H heap(g_pool, kPoolBytes);
    std::mt19937 rng(2026);
    std::vector<void*> live;
    live.reserve(1 << 18);
    const std::size_t target_bytes = static_cast<std::size_t>(target * heap.capacity());

    std::vector<std::int64_t> alloc_ns;
    alloc_ns.reserve(kMeasureOps);
    std::int64_t free_ns = 0;
    long frees = 0, allocs = 0, fails = 0;
    double fill_sum = 0;
    bool timed = false;

    auto free_random = [&]() {
        const std::size_t i = rng() % live.size();
        void* p = live[i];
        live[i] = live.back();
        live.pop_back();
        if (timed) {
            const std::int64_t t0 = now_ns();
            heap.deallocate(p);
            free_ns += now_ns() - t0 - overhead;
            ++frees;
        } else {
            heap.deallocate(p);
        }
    };

    auto step = [&]() {
        if (live.empty() || heap.used_bytes() < target_bytes) {
            const std::size_t n = random_size(rng);
            void* p;
            if (timed) {
                const std::int64_t t0 = now_ns();
                p = heap.allocate(n);
                alloc_ns.push_back(now_ns() - t0 - overhead);
                ++allocs;
                if (!p) ++fails;
            } else {
                p = heap.allocate(n);
            }
            if (p) live.push_back(p);
            else if (!live.empty()) free_random();  // yer yok: uygulama bir şey bırakır
        } else {
            free_random();
        }
        if (timed) fill_sum += double(heap.used_bytes()) / double(heap.capacity());
    };

    for (int guard = 0; heap.used_bytes() < target_bytes && guard < 10000000; ++guard) step();
    for (int i = 0; i < kWarmupOps; ++i) step();

    heap.reset_counters();
    timed = true;
    for (int i = 0; i < kMeasureOps; ++i) step();

    for (std::size_t i = 0; i < alloc_ns.size(); ++i) alloc_ns[i] = std::max<std::int64_t>(0, alloc_ns[i]);
    std::sort(alloc_ns.begin(), alloc_ns.end());
    double sum = 0;
    for (std::size_t i = 0; i < alloc_ns.size(); ++i) sum += double(alloc_ns[i]);

    Result r;
    r.fill       = fill_sum / kMeasureOps;
    r.alloc_mean = sum / double(alloc_ns.size());
    r.alloc_p50  = double(alloc_ns[alloc_ns.size() / 2]);
    r.alloc_p99  = double(alloc_ns[std::size_t(0.99 * double(alloc_ns.size() - 1))]);
    r.alloc_max  = double(alloc_ns.back());
    r.free_mean  = frees ? std::max(0.0, double(free_ns) / double(frees)) : 0.0;
    r.fail_pct   = 100.0 * double(fails) / double(allocs);
    r.steps      = double(heap.search_steps()) / double(allocs);
    return r;
}

void print_row(double target, const char* name, const Result& r) {
    std::printf("%5.1f%%  %-9s %6.1f%% | %8.0f %7.0f %8.0f %9.0f | %7.0f | %6.2f%% | %9.1f\n",
                target * 100, name, r.fill * 100, r.alloc_mean, r.alloc_p50, r.alloc_p99,
                r.alloc_max, r.free_mean, r.fail_pct, r.steps);
}

}  // namespace

// Kullanım: ./bench_heap [islem_sayisi] [seviye1 seviye2 ...]   örn: ./bench_heap 50000 0.5 0.99
int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc > 1) kWarmupOps = kMeasureOps = std::atoi(argv[1]);
    std::vector<double> levels = {0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98, 0.99, 0.995};
    if (argc > 2) {
        levels.clear();
        for (int i = 2; i < argc; ++i) levels.push_back(std::atof(argv[i]));
    }
    g_pool = new unsigned char[kPoolBytes];
    std::memset(g_pool, 0, kPoolBytes);  // sayfalar önceden dokunulsun; page fault ölçüme girmesin
    const std::int64_t ovh = timer_overhead_ns();

    std::printf("Havuz %zu MB | seviye basina %d olcum islemi | zamanlayici ek yuku %lld ns (cikarildi)\n",
                kPoolBytes >> 20, kMeasureOps, static_cast<long long>(ovh));
    std::printf("Sureler ns. 'adim' = allocate basina gezilen bos blok sayisi.\n\n");
    std::printf("Hedef   Heap      Gercek  | allocOrt     p50      p99       max | freeOrt | Basarsz | adim/alloc\n");
    std::printf("-------------------------------------------------------------------------------------------\n");

    for (double t : levels) {
        print_row(t, "FirstFit", run<mem::FirstFitHeap>(t, ovh));
        print_row(t, "TLSF", run<mem::TlsfHeap>(t, ovh));
        std::printf("\n");
    }
    return 0;
}
