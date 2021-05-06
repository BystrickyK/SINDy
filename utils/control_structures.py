import signal
import time
from contextlib import contextmanager

def timeout_handler(signum, frame):
    raise TimeoutError()

@contextmanager
def timeout(seconds):
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


symbol_library = {0: '\\',
                  1: '|',
                  2: '/',
                  3: '|'}
def time_symbol():
    t = round(time.time()*10)
    tmod = t % 4
    return symbol_library[tmod]


