// Copyright (C) 2025 Qianyue He
//
// This program is free software: you can redistribute it and/or
// modify it under the terms of the GNU Affero General Public License
// as published by the Free Software Foundation, either
// version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
// the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General
// Public License along with this program. If not, see
//
//             <https://www.gnu.org/licenses/>.

/**
 * @author Qianyue He
 * @brief Header only. An improved version of thread pool (can be reused in the
 * future projects), improved from my another repo:
 * https://github.com/Enigmatisms/culina/tree/master/src/misc/thread_pool
 * @date 2025.06.06
 */

#pragma once
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace utils {

/**
 * @brief My own updated and generalized thread pool
 * @tparam ThreadLocalData: thread local data is used for reduction after the
 * multi-threading task
 */
template <typename ThreadLocalData> class ThreadPool {
  public:
    using TaskType = std::function<void(ThreadLocalData &)>;

    ThreadPool(size_t num_workers, std::function<ThreadLocalData()> init_local,
               std::function<ThreadLocalData(std::vector<ThreadLocalData> &&)>
                   reduce_func)
        : num_workers(num_workers), init_local(std::move(init_local)),
          reduce_func(std::move(reduce_func)), is_closed(false) {
        thread_local_data.reserve(num_workers);
        for (size_t i = 0; i < num_workers; ++i) {
            thread_local_data.emplace_back(this->init_local());
        }

        workers.reserve(num_workers);
        for (size_t i = 0; i < num_workers; ++i) {
            workers.emplace_back([this, i] { worker_thread(i); });
        }
    }

    ThreadPool(ThreadPool &&other) noexcept
        : num_workers(other.num_workers),
          init_local(std::move(other.init_local)),
          reduce_func(std::move(other.reduce_func)), is_closed(false),
          task_queue(std::move(other.task_queue)),
          thread_local_data(std::move(other.thread_local_data)) {
        other.is_closed = true;
        other.workers.clear();
    }

    ~ThreadPool() { run_until_close(); }

    // submit the work to the thread and return a future, this involves some
    // complex packing, unpacking and variadic template, this version provides a
    // future as return value, should out task requires return value
    template <typename F, typename... Args>
    auto enqueue(F &&f, Args &&...args) -> std::future<
        typename std::result_of_t<F(ThreadLocalData &, Args...)>> {
        using ReturnType =
            typename std::result_of_t<F(ThreadLocalData &, Args...)>;

        auto task =
            std::make_shared<std::packaged_task<ReturnType(ThreadLocalData &)>>(
                [func = std::forward<F>(f),
                 ... args = std::forward<Args>(args)](ThreadLocalData &local) {
                    return func(local, std::forward<decltype(args)>(args)...);
                });

        std::future<ReturnType> result = task->get_future();

        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (is_closed) {
                std::cerr
                    << "[ThreadPool] Can not enqueue on closed thread pool.\n";
                throw std::runtime_error("Enqueue on closed ThreadPool");
            }

            task_queue.emplace_back(
                [task](ThreadLocalData &local) { (*task)(local); });
        }

        condition.notify_one();
        return result;
    }

    // in my SBVH, I might just use this return-value-free version
    template <typename F, typename... Args>
    void enqueue_void(F &&f, Args &&...args) {
        auto task = [f = std::forward<F>(f),
                     ... args = std::forward<Args>(args)](
                        ThreadLocalData &local) mutable {
            f(local, std::forward<decltype(args)>(args)...);
        };

        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (is_closed) {
                std::cerr
                    << "[ThreadPool] Can not enqueue on closed thread pool.\n";
                throw std::runtime_error("Enqueue on closed ThreadPool");
            }
            task_queue.emplace_back(std::move(task));
        }

        condition.notify_one();
    }

    // shut down the thread pool and reduce over the local data
    void run_until_close() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            is_closed = true;
        }

        condition.notify_all();
        for (std::thread &worker : workers) {
            if (worker.joinable())
                worker.join();
        }
    }

    ThreadLocalData get_reduced() {
        return reduce_func(std::move(thread_local_data));
    }

  private:
    void worker_thread(size_t thread_id) {
        while (true) {
            TaskType task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                condition.wait(
                    lock, [this] { return !task_queue.empty() || is_closed; });

                if (is_closed && task_queue.empty()) {
                    return;
                }

                task = std::move(task_queue.front());
                task_queue.pop_front();
            }

            task(thread_local_data[thread_id]);
        }
    }

    const size_t num_workers;
    std::function<ThreadLocalData()> init_local;
    std::function<ThreadLocalData(std::vector<ThreadLocalData> &&)> reduce_func;

    std::atomic<bool> is_closed;
    std::mutex queue_mutex;
    std::condition_variable condition;

    std::deque<TaskType> task_queue;
    std::vector<ThreadLocalData> thread_local_data;
    std::vector<std::thread> workers;
};

} // end namespace utils
