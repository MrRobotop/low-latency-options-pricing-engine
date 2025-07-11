#pragma once

#include <memory>
#include <vector>
#include <atomic>
#include <cstddef>
#include <new>

namespace options::utils {

template<typename T, std::size_t BlockSize = 1024>
class MemoryPool {
public:
    static constexpr std::size_t ALIGNMENT = alignof(T);
    static constexpr std::size_t OBJECT_SIZE = sizeof(T);
    
    MemoryPool() {
        allocate_new_block();
    }
    
    ~MemoryPool() {
        for (auto& block : blocks_) {
            std::aligned_alloc(ALIGNMENT, BlockSize * OBJECT_SIZE);
            std::free(block);
        }
    }
    
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    MemoryPool(MemoryPool&&) = delete;
    MemoryPool& operator=(MemoryPool&&) = delete;
    
    template<typename... Args>
    T* allocate(Args&&... args) {
        T* ptr = allocate_raw();
        if (ptr) {
            new (ptr) T(std::forward<Args>(args)...);
        }
        return ptr;
    }
    
    void deallocate(T* ptr) {
        if (ptr) {
            ptr->~T();
            deallocate_raw(ptr);
        }
    }
    
    T* allocate_raw() {
        std::size_t current_pos = current_position_.load(std::memory_order_relaxed);
        
        while (current_pos < BlockSize) {
            if (current_position_.compare_exchange_weak(
                current_pos, current_pos + 1, std::memory_order_acquire)) {
                
                return reinterpret_cast<T*>(
                    static_cast<char*>(current_block_) + current_pos * OBJECT_SIZE);
            }
        }
        
        allocate_new_block();
        return allocate_raw();
    }
    
    void deallocate_raw(T* ptr) {
        free_list_.push(ptr);
    }
    
    std::size_t get_allocated_count() const {
        return allocated_count_.load(std::memory_order_relaxed);
    }
    
    std::size_t get_deallocated_count() const {
        return deallocated_count_.load(std::memory_order_relaxed);
    }
    
    double get_utilization() const {
        std::size_t allocated = get_allocated_count();
        std::size_t total_capacity = blocks_.size() * BlockSize;
        return total_capacity > 0 ? static_cast<double>(allocated) / total_capacity : 0.0;
    }

private:
    struct FreeListNode {
        FreeListNode* next;
    };
    
    class LockFreeFreeList {
    public:
        void push(T* ptr) {
            FreeListNode* node = reinterpret_cast<FreeListNode*>(ptr);
            FreeListNode* old_head = head_.load(std::memory_order_relaxed);
            
            do {
                node->next = old_head;
            } while (!head_.compare_exchange_weak(
                old_head, node, std::memory_order_release, std::memory_order_relaxed));
        }
        
        T* pop() {
            FreeListNode* old_head = head_.load(std::memory_order_acquire);
            
            while (old_head != nullptr) {
                FreeListNode* new_head = old_head->next;
                if (head_.compare_exchange_weak(
                    old_head, new_head, std::memory_order_release, std::memory_order_relaxed)) {
                    
                    return reinterpret_cast<T*>(old_head);
                }
            }
            
            return nullptr;
        }
        
        bool empty() const {
            return head_.load(std::memory_order_acquire) == nullptr;
        }

    private:
        std::atomic<FreeListNode*> head_{nullptr};
    };
    
    void allocate_new_block() {
        void* new_block = std::aligned_alloc(ALIGNMENT, BlockSize * OBJECT_SIZE);
        if (!new_block) {
            throw std::bad_alloc();
        }
        
        {
            std::lock_guard<std::mutex> lock(blocks_mutex_);
            blocks_.push_back(new_block);
            current_block_ = new_block;
        }
        
        current_position_.store(0, std::memory_order_release);
    }
    
    std::vector<void*> blocks_;
    std::mutex blocks_mutex_;
    void* current_block_ = nullptr;
    std::atomic<std::size_t> current_position_{BlockSize};
    
    LockFreeFreeList free_list_;
    std::atomic<std::size_t> allocated_count_{0};
    std::atomic<std::size_t> deallocated_count_{0};
};

template<typename T>
class ObjectPool {
public:
    explicit ObjectPool(std::size_t initial_size = 100) {
        for (std::size_t i = 0; i < initial_size; ++i) {
            auto obj = std::make_unique<T>();
            available_objects_.push_back(std::move(obj));
        }
    }
    
    template<typename... Args>
    std::unique_ptr<T> acquire(Args&&... args) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (!available_objects_.empty()) {
            auto obj = std::move(available_objects_.back());
            available_objects_.pop_back();
            return obj;
        }
        
        return std::make_unique<T>(std::forward<Args>(args)...);
    }
    
    void release(std::unique_ptr<T> obj) {
        if (obj) {
            std::lock_guard<std::mutex> lock(mutex_);
            available_objects_.push_back(std::move(obj));
        }
    }
    
    std::size_t available_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return available_objects_.size();
    }

private:
    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<T>> available_objects_;
};

class StackAllocator {
public:
    explicit StackAllocator(std::size_t size) 
        : size_(size), memory_(std::aligned_alloc(64, size)), current_(0) {
        if (!memory_) {
            throw std::bad_alloc();
        }
    }
    
    ~StackAllocator() {
        std::free(memory_);
    }
    
    StackAllocator(const StackAllocator&) = delete;
    StackAllocator& operator=(const StackAllocator&) = delete;
    
    template<typename T>
    T* allocate(std::size_t count = 1) {
        const std::size_t alignment = alignof(T);
        const std::size_t size_needed = sizeof(T) * count;
        
        std::size_t aligned_current = (current_ + alignment - 1) & ~(alignment - 1);
        
        if (aligned_current + size_needed > size_) {
            return nullptr;
        }
        
        current_ = aligned_current + size_needed;
        return reinterpret_cast<T*>(static_cast<char*>(memory_) + aligned_current);
    }
    
    void reset() {
        current_ = 0;
    }
    
    std::size_t bytes_used() const {
        return current_;
    }
    
    std::size_t bytes_available() const {
        return size_ - current_;
    }
    
    double utilization() const {
        return static_cast<double>(current_) / size_;
    }

private:
    std::size_t size_;
    void* memory_;
    std::size_t current_;
};

template<typename T, std::size_t Capacity>
class FixedSizeAllocator {
public:
    FixedSizeAllocator() {
        for (std::size_t i = 0; i < Capacity; ++i) {
            free_indices_.push(i);
        }
    }
    
    template<typename... Args>
    T* allocate(Args&&... args) {
        if (free_indices_.empty()) {
            return nullptr;
        }
        
        std::size_t index = free_indices_.top();
        free_indices_.pop();
        
        T* ptr = reinterpret_cast<T*>(&storage_[index]);
        new (ptr) T(std::forward<Args>(args)...);
        
        return ptr;
    }
    
    void deallocate(T* ptr) {
        if (!ptr) return;
        
        ptr->~T();
        
        std::size_t index = ptr - reinterpret_cast<T*>(storage_.data());
        if (index < Capacity) {
            free_indices_.push(index);
        }
    }
    
    std::size_t capacity() const {
        return Capacity;
    }
    
    std::size_t available() const {
        return free_indices_.size();
    }
    
    std::size_t used() const {
        return Capacity - available();
    }

private:
    alignas(T) std::array<std::byte, sizeof(T) * Capacity> storage_;
    std::stack<std::size_t> free_indices_;
};

}