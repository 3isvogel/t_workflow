from concurrent.futures import as_completed, ThreadPoolExecutor
from typing import Callable
from collections.abc import Iterable

def load_dataset(file_list: Iterable, open_function: Callable) -> list[dict]:
    dataset = [i for i in range(len(file_list))]; i = 0
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(open_function, name) for name in file_list]
        for future in as_completed(futures):
            dataset[i] = future.result()
            i += 1
        return dataset
