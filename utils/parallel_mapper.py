import time
import traceback
from shutil import get_terminal_size
from multiprocess import Pool, Manager
from typing import Iterable, Callable, TypeVar, Optional, Generic
from rich.console import Console
console = Console()
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TaskProgressColumn
)

T = TypeVar('T')
R = TypeVar('R')


class ParallelMapper(Generic[T, R]):
    """
    A class for handling parallel mapping operations with a progress bar.

    Attributes:
        max_workers (Optional[int]): The maximum number of worker processes
        description (str): The description text of the progress bar
        progress_config (dict): The configuration items of the progress bar
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        description: str = "Processing",
        use_progress_bar: bool = True,
        progress_config: Optional[dict] = None,
    ):
        self.max_workers = max_workers
        self.description = "[bold red]" + description
        self.progress_config = progress_config or {}
        self.use_progress_bar = use_progress_bar

    def map(self, func: Callable[..., R], *items: Iterable[T]) -> list[R]:
        """
        Parallel mapping of a function to multiple iterable objects, with progress tracking.

        Parameters:
            func: The function to be applied to each element combination
            items: One or more iterable input objects

        Returns:
            A list containing all results

        Throws:
            Exception: If any subtask processing fails
        """
        counter = Manager().Value('Q', 0)
        total = len(list(items[0]))

        def wrapper(*args):
            try:
                result = func(*args)
                counter.value += 1
                return result
            except Exception as e:
                raise RuntimeError(f"\n[ERROR] SUBTASK FAILED:\n\n- Args:{args}\n\n- Error Message:{str(e)}\n")

        def get_results(async_results, counter, total):
            if self.use_progress_bar:
                progress = Progress(
                    SpinnerColumn(finished_text="[green]✓"),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(bar_width=int(get_terminal_size().columns / 2)),
                    TaskProgressColumn("[progress.percentage]{task.percentage:>3.2f}% [{task.completed}/{task.total}]"),
                    TimeRemainingColumn(),
                    TimeElapsedColumn(),
                    **self.progress_config
                )

                with progress:
                    task = progress.add_task(self.description, total=total)
                    current = 0
                    while not async_results.ready():
                        cnt = counter.value
                        if cnt != current:
                            current = cnt
                            progress.update(task, completed=current)
                        time.sleep(0.1)
                    try:
                        results = list(async_results.get())
                        progress.update(task, completed=total)
                        progress.update(task, description="[bold green]Done")
                        return results
                    except Exception as e:
                        cnt = counter.value
                        if cnt != current:
                            progress.update(task, completed=cnt)
                        progress.update(task, description="[bold red]Failed")
                        raise RuntimeError(f"{e}")
            else:
                def log_progress(current, total):
                    if total == 0:
                        percentage = 0
                    else:
                        percentage = (current / total) * 100
                    console.log(
                        f"Progress: {percentage:>6.2f}% ({current}/{total})",
                        end='' if current < total else '\n')

                current = 0
                while not async_results.ready():
                    cnt = counter.value
                    if cnt != current:
                        current = cnt
                        log_progress(current, total)
                    time.sleep(0.1)
                try:
                    results = list(async_results.get())
                    log_progress(total, total)
                    return results
                except Exception as e:
                    cnt = counter.value
                    if cnt != current:
                        log_progress(cnt, total)
                    raise RuntimeError(f"{e}")

        with Pool(self.max_workers) as p:
            list_items = list(zip(*items))
            async_results = p.starmap_async(wrapper, list_items)
            results = get_results(async_results, counter, total)

        return results


def parallel_map(
    func: Callable[..., R],
    *items: Iterable[T],
    max_workers: Optional[int] = None,
    description: str = "Processing",
    use_progress_bar: bool = False,
) -> list[R]:
    """A convenience function for parallel mapping with a progress bar."""
    mapper = ParallelMapper(max_workers=max_workers,
                            description=description,
                            use_progress_bar=use_progress_bar)
    return mapper.map(func, *items)
