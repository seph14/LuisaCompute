#include <DXRuntime/CommandQueue.h>
#include <DXRuntime/CommandBuffer.h>
#include <DXRuntime/CommandAllocator.h>
#include <Resource/GpuAllocator.h>
#include <DXApi/LCEvent.h>
// WaitOnAddress/WakeByAddressAll are exported by the OneCore umbrella
// import library only (documented library for synchapi.h).
#pragma comment(lib, "OneCore.lib")
namespace lc::dx {
CommandQueue::CommandQueue(
    Device *device,
    GpuAllocator *resourceAllocator,
    D3D12_COMMAND_LIST_TYPE type)
    : _device(device),
      _resource_allocator(resourceAllocator),
      _type(type),
      _thd([this] { _execute_thread(); }) {
    auto CreateQueue = [&] {
        D3D12_COMMAND_QUEUE_DESC queueDesc = {};
        queueDesc.Type = type;
        queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT;
        switch (type) {
            case D3D12_COMMAND_LIST_TYPE_DIRECT:
                queueDesc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_HIGH;
                break;
            default:
                queueDesc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
                break;
        }
        ThrowIfFailed(device->device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(_queue.GetAddressOf())));
    };
    if (device->device_settings) {
        _queue = {device->device_settings->CreateQueue(type), false};
        if (!_queue) [[unlikely]] {
            CreateQueue();
        }
    } else {
        CreateQueue();
    }
    ThrowIfFailed(device->device->CreateFence(
        0,
        D3D12_FENCE_FLAG_NONE,
        IID_PPV_ARGS(&_cmd_fence)));
}
CommandQueue::AllocatorPtr CommandQueue::create_allocator(size_t maxAllocCount) {
    if (maxAllocCount != std::numeric_limits<uint64_t>::max()) {
        if (_last_frame > maxAllocCount)
            complete(_last_frame - maxAllocCount);
    }
    auto newPtr = _allocator_pool.dequeue();
    if (newPtr) {
        (*newPtr)->get_buffer()->update_command_buffer(_device);
        return std::move(*newPtr);
    }
    return AllocatorPtr(new CommandAllocator(_device, _resource_allocator, _type));
}

void CommandQueue::add_event(LCEvent const *evt, uint64_t fenceIdx) {
    ++_last_frame;
    _mtx.lock();
    _executed_allocators.enqueue(evt, fenceIdx, true);
    ReleaseSemaphore(_wake_sem, 1, nullptr);
    _mtx.unlock();
}

void CommandQueue::_execute_thread() {
    while (_enabled) {
        uint64_t fence;
        bool wakeupThread;
        auto Weakup = [&] {
            if (wakeupThread) {
                uint64_t prev_value = _executed_frame;
                while (prev_value < fence && !_executed_frame.compare_exchange_weak(prev_value, fence)) {
                    std::this_thread::yield();
                }
                WakeByAddressAll((void *)&_executed_frame);
            }
        };
        auto ExecuteAllocator = [&](AllocatorPtr &b) {
            b->complete(this, _cmd_fence.Get(), fence);
            b->reset(this);
            _allocator_pool.enqueue(std::move(b));
            Weakup();
        };
        auto ExecuteCallbacks = [&](vstd::vector<vstd::function<void()>> &vec) {
            for (auto &&i : vec) {
                i();
            }
            Weakup();
        };

        auto ExecuteEvent = [&](LCEvent const *evt) {
            _device->wait_fence(evt->fence(), fence);
            {
                std::lock_guard lck(evt->event_mtx);
                evt->finished_event = std::max<uint64_t>(fence, evt->finished_event);
                WakeByAddressAll((void *)&evt->finished_event);
            }
            if (wakeupThread) {
                _executed_frame++;
                WakeByAddressAll((void *)&_executed_frame);
            }
        };
        auto ExecuteHandle = [&](WaitFence) {
            _device->wait_fence(_cmd_fence.Get(), fence);
            Weakup();
        };
        while (true) {
            vstd::optional<CallbackEvent> b;
            {
                std::lock_guard lck{_mtx};
                b = _executed_allocators.dequeue();
            }
            if (!b) break;
            fence = b->fence;
            wakeupThread = b->wakeupThread;
            b->evt.multi_visit(
                ExecuteAllocator,
                ExecuteCallbacks,
                ExecuteEvent,
                ExecuteHandle);
        }
        while (_enabled && _executed_allocators.length() == 0) {
            // Producers release one token per enqueue under _mtx (and the
            // destructor before join()); the timeout is a missed-wakeup
            // backstop only.
            WaitForSingleObject(_wake_sem, 100);
        }
    }
}
void CommandQueue::force_sync(
    AllocatorPtr &alloc,
    CommandBuffer &cb) {
    cb._close();
    complete();
    auto curFrame = ++_last_frame;
    alloc->execute(this, _cmd_fence.Get(), curFrame, {}, false);
    alloc->complete(this, _cmd_fence.Get(), curFrame);
    alloc->reset(this);
    _executed_frame = curFrame;
    WakeByAddressAll((void *)&_executed_frame);

    cb._reset();
}
CommandQueue::~CommandQueue() {
    {
        std::lock_guard lck(_mtx);
        _enabled = false;
    }
    ReleaseSemaphore(_wake_sem, 1, nullptr);
    _thd.join();
    CloseHandle(_wake_sem);
}
void CommandQueue::wait_frame(uint64_t lastFrame) {
    if (lastFrame > 0)
        _queue->Wait(_cmd_fence.Get(), lastFrame);
}
void CommandQueue::signal() {
    auto curFrame = ++_last_frame;
    ThrowIfFailed(_queue->Signal(_cmd_fence.Get(), curFrame));
    _mtx.lock();
    _executed_allocators.enqueue(WaitFence{}, curFrame, true);
    ReleaseSemaphore(_wake_sem, 1, nullptr);
    _mtx.unlock();
}
void CommandQueue::execute(AllocatorPtr &&alloc, vstd::vector<vstd::function<void()>> &&callbacks, luisa::span<std::pair<IDXGISwapChain *, bool>> swapChains, bool cmdlist_is_empty) {
    auto curFrame = ++_last_frame;
    alloc->execute(this, _cmd_fence.Get(), curFrame, swapChains, cmdlist_is_empty);
    if (cmdlist_is_empty) {
        _allocator_pool.enqueue(std::move(alloc));
        if (!callbacks.empty()) {
            std::lock_guard lck{_mtx};
            _executed_allocators.enqueue(std::move(callbacks), curFrame, true);
            ReleaseSemaphore(_wake_sem, 1, nullptr);
        }
    } else {
        std::lock_guard lck{_mtx};
        _executed_allocators.enqueue(std::move(alloc), curFrame, callbacks.empty());
        ReleaseSemaphore(_wake_sem, 1, nullptr);
        if (!callbacks.empty()) {
            _executed_allocators.enqueue(std::move(callbacks), curFrame, true);
            ReleaseSemaphore(_wake_sem, 1, nullptr);
        }
    }
}

void CommandQueue::complete(uint64_t fence) {
    // WaitOnAddress re-checks the value before blocking, so a wake between
    // the load and the wait cannot be missed; the reload loop handles
    // spurious wakeups and fences beyond the woken value.
    auto cur = _executed_frame.load(std::memory_order_acquire);
    while (cur < fence) {
        WaitOnAddress(&_executed_frame, &cur, sizeof(cur), INFINITE);
        cur = _executed_frame.load(std::memory_order_acquire);
    }
}
void CommandQueue::complete() {
    complete(_last_frame);
}

}// namespace lc::dx
