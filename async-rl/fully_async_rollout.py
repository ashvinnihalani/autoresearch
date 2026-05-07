"""Persistent fully asynchronous rollout worker for slime.

Adapted from THUDM/slime examples/fully_async/fully_async_rollout.py at
commit 82007faf4b398abd32bd8e07f9638f6cfeb70729.
"""

import asyncio
import atexit
import queue
import threading
import time

from slime.rollout.sglang_rollout import GenerateState, generate_and_rm_group
from slime.utils.async_utils import run
from slime.utils.types import Sample


_global_worker = None
_worker_lock = threading.Lock()


def get_global_worker(args, data_buffer):
    """Get or create the process-global async rollout worker."""
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            print("Creating new global async worker...")
            _global_worker = AsyncRolloutWorker(
                args,
                data_buffer,
                concurrency=args.sglang_server_concurrency,
            )
            _global_worker.start()
        return _global_worker


def stop_global_worker():
    """Stop the process-global async rollout worker."""
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


class AsyncRolloutWorker:
    """Thread-backed worker that keeps rollout generation running continuously."""

    def __init__(self, args, data_buffer, concurrency=10):
        self.args = args
        self.data_buffer = data_buffer
        self.concurrency = concurrency
        self.running = True
        self.output_queue = queue.Queue(maxsize=1000)
        self.worker_thread = None
        self.state = GenerateState(args)

    async def continuous_worker_loop(self):
        """Keep pulling prompt groups and launching generation tasks."""
        print("Continuous async rollout worker started")

        active_tasks = set()
        max_concurrent_tasks = self.args.rollout_batch_size
        group_id_counter = 0

        while self.running:
            try:
                if active_tasks:
                    done_tasks = {task for task in active_tasks if task.done()}
                    for task in done_tasks:
                        try:
                            task.result()
                        except Exception as exc:
                            print(f"Task failed with exception: {exc}")
                    active_tasks -= done_tasks

                while len(active_tasks) < max_concurrent_tasks and self.running:
                    samples = self.data_buffer.get_samples(1)

                    for group in samples:
                        group_id = group_id_counter
                        group_id_counter += 1

                        task = asyncio.create_task(
                            generate_and_rm_group(
                                self.args,
                                group,
                                sampling_params=self.state.sampling_params.copy(),
                                evaluation=False,
                            )
                        )

                        def make_callback(gid):
                            def task_done_callback(done_task):
                                result = done_task.result()
                                self.output_queue.put((gid, result))

                            return task_done_callback

                        task.add_done_callback(make_callback(group_id))
                        active_tasks.add(task)
                        break

                await asyncio.sleep(1)

            except Exception as exc:
                print(f"Error in continuous worker loop: {exc}")
                await asyncio.sleep(1)

        if active_tasks:
            print(f"Waiting for {len(active_tasks)} continuous tasks to complete...")
            await asyncio.wait(active_tasks)

        print("Continuous async rollout worker stopped")

    def worker_thread_func(self):
        asyncio.run(self.continuous_worker_loop())

    def start(self):
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self.worker_thread_func, daemon=True)
            self.worker_thread.start()
            print("Started continuous async worker thread")

    def stop(self):
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
        print("Stopped async worker thread")

    def get_completed_groups(self) -> list[tuple]:
        completed = []
        while True:
            try:
                completed.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return completed

    def get_queue_size(self) -> int:
        return self.output_queue.qsize()


async def generate_rollout_async(args, rollout_id: int, data_buffer) -> list[list[Sample]]:
    """Generate one training batch by draining the persistent worker queue."""
    assert args.rollout_global_dataset

    worker = get_global_worker(args, data_buffer)
    target_data_size = args.rollout_batch_size

    data = []
    completed_groups = {}
    do_print = True

    print(f"Starting async rollout generation for {target_data_size} groups")
    print(f"Global worker queue size: {worker.get_queue_size()}")

    start_time = time.time()
    last_progress_time = start_time
    no_progress_timeout = 30.0

    while len(data) < target_data_size:
        completed = worker.get_completed_groups()

        made_progress = False
        for group_id, group in completed:
            completed_groups[group_id] = group
            made_progress = True

        if made_progress:
            last_progress_time = time.time()

        processed_any = False
        available_ids = list(completed_groups.keys())
        for group_id in available_ids:
            if len(data) >= target_data_size:
                break

            group = completed_groups.pop(group_id)

            try:
                any_aborted = any(sample.status == Sample.Status.ABORTED for sample in group)
            except Exception:
                any_aborted = False

            if any_aborted:
                try:
                    data_buffer.add_samples([group])
                    print(f"Returned aborted group {group_id} to data buffer", flush=True)
                except Exception as exc:
                    print(f"Failed to return aborted group {group_id} to buffer: {exc}", flush=True)
                continue

            if do_print:
                print(
                    f"First rollout sample: {[group[0].prompt + group[0].response]}, "
                    f"label: {group[0].label}, reward: {group[0].reward}",
                    flush=True,
                )
                do_print = False

            data.append(group)
            processed_any = True

        current_time = time.time()
        if current_time - last_progress_time > no_progress_timeout:
            print(
                f"Warning: No progress for {no_progress_timeout}s. "
                f"Queue size: {worker.get_queue_size()}, "
                f"Collected: {len(data)}/{target_data_size}"
            )
            last_progress_time = current_time

        if not processed_any:
            await asyncio.sleep(0.01)

    duration = time.time() - start_time
    print(f"Rollout completed in {duration:.2f}s! Global worker queue size: {worker.get_queue_size()}")

    if data:
        print(
            f"Finish rollout: {[data[-1][0].prompt + data[-1][0].response]}, "
            f"label: {data[-1][0].label}, reward: {data[-1][0].reward}",
            flush=True,
        )

    data = sorted(data, key=lambda group: group[0].index)
    return data


def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation=False):
    if evaluation:
        raise ValueError("Evaluation mode not supported in simple async rollout")
    return run(generate_rollout_async(args, rollout_id, data_buffer))


atexit.register(stop_global_worker)
