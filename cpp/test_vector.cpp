#include <iostream>
#include <vector>
#include <list>
#include <chrono>
#include <iomanip>

/*
    This function demonstrates the behavior of iterators in a vector when the vector is resized.
    It initializes a vector with some integers, creates an iterator pointing to the first element, and then resizes the vector by adding more elements.
    The output shows the value pointed to by the iterator before and after resizing.
    Note that resizing a vector can invalidate iterators, so care must be taken when using them after modifications to the vector.
*/
void test_vector()
{
    std::vector<int> vec = {1, 2, 3, 4, 5};

    std::vector<int>::iterator it = vec.begin();
    
    std::cout << "Before resizing the element, the iterator points to: " << *it << std::endl; // This line will cause a compilation error because 'it' is an iterator to int, not a pair.

    int arr[] = {10, 20, 30, 40, 50};

    for (int i = 0; i < 5; ++i)
    {
        vec.push_back(arr[i]);
    }

    std::cout << "After resizing the element, the iterator points to: " << *it << std::endl; // This line will cause a compilation error because 'it' is an iterator to int, not a pair.
}

/*
    This function measures the speed difference between std::vector and std::list for three access patterns:
    appending at the back, walking over every element, and inserting at the front.
    Each section is timed with std::chrono and the elapsed time is reported in milliseconds.
    Vector stores its elements contiguously, so it wins whenever the CPU cache can prefetch the next element,
    while list stores every element in a separate node and only wins where it can splice a node in without shifting the rest.
*/
void vector_vs_list()
{
    typedef std::chrono::high_resolution_clock clock;

    const int N = 1000000;      // element count for the back-insertion and traversal tests
    const int FRONT_N = 50000;  // smaller, because inserting at the front of a vector is O(n) per insertion

    std::vector<int> vec;
    std::list<int> lst;

    // 1) Appending at the back.
    clock::time_point start = clock::now();
    for (int i = 0; i < N; ++i)
    {
        vec.push_back(i);
    }
    double vec_push_ms = std::chrono::duration<double, std::milli>(clock::now() - start).count();

    start = clock::now();
    for (int i = 0; i < N; ++i)
    {
        lst.push_back(i);
    }
    double lst_push_ms = std::chrono::duration<double, std::milli>(clock::now() - start).count();

    // 2) Walking over every element. The sums are printed so the loops cannot be optimized away.
    start = clock::now();
    long long vec_sum = 0;
    for (std::vector<int>::const_iterator it = vec.begin(); it != vec.end(); ++it)
    {
        vec_sum += *it;
    }
    double vec_walk_ms = std::chrono::duration<double, std::milli>(clock::now() - start).count();

    start = clock::now();
    long long lst_sum = 0;
    for (std::list<int>::const_iterator it = lst.begin(); it != lst.end(); ++it)
    {
        lst_sum += *it;
    }
    double lst_walk_ms = std::chrono::duration<double, std::milli>(clock::now() - start).count();

    // 3) Inserting at the front. Vector has to shift every existing element, list only relinks a node.
    std::vector<int> vec_front;
    std::list<int> lst_front;

    start = clock::now();
    for (int i = 0; i < FRONT_N; ++i)
    {
        vec_front.insert(vec_front.begin(), i);
    }
    double vec_front_ms = std::chrono::duration<double, std::milli>(clock::now() - start).count();

    start = clock::now();
    for (int i = 0; i < FRONT_N; ++i)
    {
        lst_front.push_front(i);
    }
    double lst_front_ms = std::chrono::duration<double, std::milli>(clock::now() - start).count();

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n--- vector vs list ---\n";
    std::cout << std::left << std::setw(30) << "operation"
              << std::right << std::setw(12) << "vector (ms)"
              << std::setw(12) << "list (ms)"
              << std::setw(12) << "ratio" << "\n";

    std::cout << std::left << std::setw(30) << "push_back (1M)"
              << std::right << std::setw(12) << vec_push_ms
              << std::setw(12) << lst_push_ms
              << std::setw(12) << lst_push_ms / vec_push_ms << "x\n";

    std::cout << std::left << std::setw(30) << "traversal (1M)"
              << std::right << std::setw(12) << vec_walk_ms
              << std::setw(12) << lst_walk_ms
              << std::setw(12) << lst_walk_ms / vec_walk_ms << "x\n";

    std::cout << std::left << std::setw(30) << "insert at front (50K)"
              << std::right << std::setw(12) << vec_front_ms
              << std::setw(12) << lst_front_ms
              << std::setw(12) << vec_front_ms / lst_front_ms << "x\n";

    std::cout << "ratio = how many times slower the losing container is.\n";
    std::cout << "checksums: " << vec_sum << " / " << lst_sum << std::endl; // keeps the traversal loops alive
}


int main()
{
    test_vector();
    vector_vs_list();
    return 0;
}
