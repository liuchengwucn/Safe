from loguru import logger
import uuid
from prover.lean.verifier import Lean4ServerScheduler
import multiprocessing as mp


def trace(func):
    @logger.catch()
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        logger.info(
            f"Called function {func.__name__} with args {args} and kwargs {kwargs}."
        )
        logger.info(f"Returned {result}.")
        return result

    return wrapper


def timeout_handler(func, args=(), kwargs={}, timeout_duration=1, default=None):
    def target(pipe):
        result = func(*args, **kwargs)
        pipe.send(result)

    parent_pipe, child_pipe = mp.Pipe()
    process = mp.Process(target=target, args=(child_pipe,))
    process.start()
    process.join(timeout_duration)
    if process.is_alive():
        print("Terminating due to timeout.")
        process.kill()
        # process.join()
        return default
    else:
        if parent_pipe.poll():
            return parent_pipe.recv()
        else:
            return default


def call_scheduler(scheduler_input):
    scheduler = Lean4ServerScheduler(
        max_concurrent_requests=len(scheduler_input),
        timeout=60,
        memory_limit=10,
        name=str(uuid.uuid4()),
    )
    request_id_list = scheduler.submit_all_request(scheduler_input)
    outputs_list = scheduler.get_all_request_outputs(request_id_list)

    scheduler.close()
    return outputs_list


def call_scheduler_with_timeout(scheduler_input):
    return timeout_handler(
        call_scheduler, args=(scheduler_input,), timeout_duration=600, default=[]
    )
