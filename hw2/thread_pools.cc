#include <iostream>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>

class ThreadPool {
public:
    ThreadPool(size_t numThreads)
        : stop(false) {
        for (size_t i = 0; i < numThreads; ++i) {
            threads.emplace_back([this] {
                while (true) {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(queueMutex);
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });

                        if (stop && tasks.empty()) {
                            return;
                        }

                        task = std::move(tasks.front());
                        tasks.pop();
                    }

                    task();
                }
            });
        }
    }

    template <typename Function>
    void Enqueue(Function&& f) {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            tasks.emplace(std::forward<Function>(f));
        }
        condition.notify_one();
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            stop = true;
        }

        condition.notify_all();
        for (std::thread& thread : threads) {
            thread.join();
        }
    }
    std::queue<std::function<void()>> tasks;

private:
    std::vector<std::thread> threads;
    
    std::mutex queueMutex;
    std::condition_variable condition;
    std::atomic<bool> stop;
};

void fnc(int i) {
    std::cout << i << std::endl;
}

int main() {
    ThreadPool pool(4); // Create a thread pool with 4 threads

    // Enqueue tasks for execution
    for (int i = 0; i < 10; ++i) {
        pool.Enqueue([i] {
            fnc(i);
        });
    }

    return 0; // The ThreadPool destructor will wait for all tasks to complete and join the threads.
}
