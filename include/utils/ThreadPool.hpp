#pragma once

#include <thread>
#include <vector>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>
#include <memory>

namespace options::utils {

class ThreadPool {
public:
    explicit ThreadPool(std::size_t num_threads = std::thread::hardware_concurrency())
        : stop_flag_(false) {
        
        workers_.reserve(num_threads);
        
        for (std::size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this] { worker_thread(); });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_flag_ = true;
        }
        
        condition_.notify_all();
        
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    template<typename F, typename... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
        using ReturnType = std::invoke_result_t<F, Args...>;
        
        auto task = std::make_shared<std::packaged_task<ReturnType()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        std::future<ReturnType> result = task->get_future();
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            
            if (stop_flag_) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }
            
            tasks_.emplace([task]() { (*task)(); });
        }
        
        condition_.notify_one();
        return result;
    }

    std::size_t size() const {
        return workers_.size();
    }

    std::size_t pending_tasks() const {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        return tasks_.size();
    }

    void wait_for_completion() {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        completion_condition_.wait(lock, [this] {
            return tasks_.empty() && active_tasks_.load() == 0;
        });
    }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    
    mutable std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::condition_variable completion_condition_;
    
    std::atomic<bool> stop_flag_;
    std::atomic<std::size_t> active_tasks_{0};

    void worker_thread() {
        while (true) {
            std::function<void()> task;
            
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                condition_.wait(lock, [this] { return stop_flag_ || !tasks_.empty(); });
                
                if (stop_flag_ && tasks_.empty()) {
                    return;
                }
                
                task = std::move(tasks_.front());
                tasks_.pop();
                active_tasks_.fetch_add(1, std::memory_order_relaxed);
            }
            
            task();
            
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                active_tasks_.fetch_sub(1, std::memory_order_relaxed);
                if (tasks_.empty() && active_tasks_.load() == 0) {
                    completion_condition_.notify_all();
                }
            }
        }
    }
};

template<typename T>
class LockFreeQueue {
public:
    LockFreeQueue() : head_(new Node), tail_(head_.load()) {}

    ~LockFreeQueue() {
        while (Node* const old_head = head_.load()) {
            head_.store(old_head->next);
            delete old_head;
        }
    }

    void enqueue(T item) {
        Node* const new_node = new Node;
        Node* const prev_tail = tail_.exchange(new_node);
        prev_tail->data = std::move(item);
        prev_tail->next = new_node;
    }

    bool dequeue(T& result) {
        Node* head = head_.load();
        if (head->next == nullptr) {
            return false;
        }
        
        result = std::move(head->data);
        head_.store(head->next);
        delete head;
        return true;
    }

    bool empty() const {
        Node* head = head_.load();
        return head->next == nullptr;
    }

private:
    struct Node {
        std::atomic<Node*> next{nullptr};
        T data;
    };

    std::atomic<Node*> head_;
    std::atomic<Node*> tail_;
};

class WorkStealingThreadPool {
public:
    explicit WorkStealingThreadPool(std::size_t num_threads = std::thread::hardware_concurrency())
        : done_(false), joiner_(workers_) {
        
        try {
            for (std::size_t i = 0; i < num_threads; ++i) {
                work_queues_.emplace_back(std::make_unique<LockFreeQueue<std::function<void()>>>());
                workers_.emplace_back(&WorkStealingThreadPool::worker_thread, this, i);
            }
        } catch (...) {
            done_ = true;
            throw;
        }
    }

    ~WorkStealingThreadPool() {
        done_ = true;
    }

    template<typename F>
    std::future<std::invoke_result_t<F>> submit(F f) {
        using ResultType = std::invoke_result_t<F>;
        
        std::packaged_task<ResultType()> task(std::move(f));
        std::future<ResultType> result(task.get_future());
        
        if (local_work_queue_) {
            local_work_queue_->enqueue(std::move(task));
        } else {
            pool_work_queue_.enqueue(std::move(task));
        }
        
        return result;
    }

private:
    using TaskType = std::function<void()>;
    
    std::atomic<bool> done_;
    LockFreeQueue<TaskType> pool_work_queue_;
    std::vector<std::unique_ptr<LockFreeQueue<TaskType>>> work_queues_;
    std::vector<std::thread> workers_;

    static thread_local LockFreeQueue<TaskType>* local_work_queue_;
    static thread_local std::size_t my_index_;

    class ThreadJoiner {
    public:
        explicit ThreadJoiner(std::vector<std::thread>& threads) : threads_(threads) {}
        
        ~ThreadJoiner() {
            for (auto& thread : threads_) {
                if (thread.joinable()) {
                    thread.join();
                }
            }
        }

    private:
        std::vector<std::thread>& threads_;
    };

    ThreadJoiner joiner_;

    void worker_thread(std::size_t my_index) {
        my_index_ = my_index;
        local_work_queue_ = work_queues_[my_index].get();
        
        while (!done_) {
            run_pending_task();
        }
    }

    bool pop_task_from_local_queue(TaskType& task) {
        return local_work_queue_ && local_work_queue_->dequeue(task);
    }

    bool pop_task_from_pool_queue(TaskType& task) {
        return pool_work_queue_.dequeue(task);
    }

    bool pop_task_from_other_thread_queue(TaskType& task) {
        for (std::size_t i = 0; i < work_queues_.size(); ++i) {
            const std::size_t index = (my_index_ + i + 1) % work_queues_.size();
            if (work_queues_[index]->dequeue(task)) {
                return true;
            }
        }
        return false;
    }

    void run_pending_task() {
        TaskType task;
        
        if (pop_task_from_local_queue(task) ||
            pop_task_from_pool_queue(task) ||
            pop_task_from_other_thread_queue(task)) {
            task();
        } else {
            std::this_thread::yield();
        }
    }
};

thread_local LockFreeQueue<std::function<void()>>* WorkStealingThreadPool::local_work_queue_ = nullptr;
thread_local std::size_t WorkStealingThreadPool::my_index_ = 0;

class ParallelExecutor {
public:
    template<typename Iterator, typename Function>
    static void parallel_for(Iterator begin, Iterator end, Function func, 
                           std::size_t num_threads = std::thread::hardware_concurrency()) {
        
        const auto distance = std::distance(begin, end);
        if (distance <= 0) return;
        
        const std::size_t chunk_size = std::max(1L, distance / static_cast<long>(num_threads));
        
        ThreadPool pool(num_threads);
        std::vector<std::future<void>> futures;
        
        auto current = begin;
        while (current != end) {
            auto chunk_end = current;
            std::advance(chunk_end, std::min(chunk_size, static_cast<std::size_t>(std::distance(current, end))));
            
            futures.push_back(pool.enqueue([current, chunk_end, func]() {
                for (auto it = current; it != chunk_end; ++it) {
                    func(*it);
                }
            }));
            
            current = chunk_end;
        }
        
        for (auto& future : futures) {
            future.wait();
        }
    }

    template<typename Container, typename Function>
    static auto parallel_transform(const Container& input, Function func, 
                                 std::size_t num_threads = std::thread::hardware_concurrency()) {
        
        using ResultType = std::invoke_result_t<Function, typename Container::value_type>;
        std::vector<ResultType> result(input.size());
        
        const std::size_t chunk_size = std::max(1UL, input.size() / num_threads);
        
        ThreadPool pool(num_threads);
        std::vector<std::future<void>> futures;
        
        for (std::size_t i = 0; i < input.size(); i += chunk_size) {
            const std::size_t end_idx = std::min(i + chunk_size, input.size());
            
            futures.push_back(pool.enqueue([&input, &result, func, i, end_idx]() {
                for (std::size_t j = i; j < end_idx; ++j) {
                    result[j] = func(input[j]);
                }
            }));
        }
        
        for (auto& future : futures) {
            future.wait();
        }
        
        return result;
    }

    template<typename Container, typename Function, typename T>
    static T parallel_reduce(const Container& input, Function func, T init_value,
                           std::size_t num_threads = std::thread::hardware_concurrency()) {
        
        if (input.empty()) return init_value;
        
        const std::size_t chunk_size = std::max(1UL, input.size() / num_threads);
        
        ThreadPool pool(num_threads);
        std::vector<std::future<T>> futures;
        
        for (std::size_t i = 0; i < input.size(); i += chunk_size) {
            const std::size_t end_idx = std::min(i + chunk_size, input.size());
            
            futures.push_back(pool.enqueue([&input, func, init_value, i, end_idx]() {
                T local_result = init_value;
                for (std::size_t j = i; j < end_idx; ++j) {
                    local_result = func(local_result, input[j]);
                }
                return local_result;
            }));
        }
        
        T final_result = init_value;
        for (auto& future : futures) {
            final_result = func(final_result, future.get());
        }
        
        return final_result;
    }
};

}