import json
import shutil

from transformers import AutoTokenizer, PreTrainedTokenizer
from typing import List, Dict, Union, Optional, Tuple
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from itertools import islice

# Global tokenizer for subprocesses
_global_tokenizer: PreTrainedTokenizer = None # type: ignore

def _init_worker(tokenizer_path: str) -> None:
    """Load tokenizer when initializing a subprocess"""
    global _global_tokenizer
    _global_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path) # type: ignore

def _count_file(path: Union[str, Path]) -> Tuple[str, int]:
    """Count the number of tokens in a single SVG file"""
    p = Path(path)
    text = p.read_text(encoding='utf-8')
    length = len(_global_tokenizer.encode(text))
    return p.name, length

def _count_chars_file(path: Union[str, Path]) -> Tuple[str, int]:
    """Parallel helper: count the number of characters in a single SVG file"""
    p = Path(path)
    text = p.read_text(encoding='utf-8')
    return p.name, len(text)

def _copy_file(args: Tuple[Path, Path]) -> str:
    source_file, target_file = args
    shutil.copy2(source_file, target_file)
    return source_file.name

class TokenCounter:
    """
    A tool class for calculating and processing text tokens.

    This class uses AutoTokenizer from Hugging Face's transformers library to load a pre-trained tokenizer,
    and provides functions for calculating token numbers, getting token IDs, and getting readable token strings.
    
    Example usage:
        # Assume you have a tokenizer model locally or on Hugging Face Hub
        # tokenizer_path = "path/to/your/tokenizer"
        # tokenizer_path = "gpt2" # Use the model on Hugging Face Hub
        
        try:
            # Initialize
            counter = TokenCounter("gpt2")
            
            # Use the tool
            text = "Hello, world! This is a test."
            
            # 1. Calculate token length
            token_len = counter.count_tokens(text)
            print(f"Token count: {token_len}")
            
            # 2. Get token IDs after tokenization
            token_ids = counter.get_token_ids(text)
            print(f"Token IDs: {token_ids}")

            # 3. Get token strings after tokenization
            token_strings = counter.get_tokens_str(text)
            print(f"Token strings: {token_strings}")
            
            # 4. Get all information at once
            processed_info = counter.process(text)
            import json
            print(f"Processed Info:\\n{json.dumps(processed_info, indent=2)}")

        except Exception as e:
            print(f"An error occurred: {e}")

    """
    
    tokenizer_path: str
    tokenizer: PreTrainedTokenizer

    def __init__(self, tokenizer_path: str="PATH_TO_TOKENIZER") -> None:
        """
        Initialize TokenCounter.
        
        :param tokenizer_path: Path to the pre-trained tokenizer model or the model name on Hugging Face Hub
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path) # type: ignore
            self.tokenizer_path = tokenizer_path
            _global_tokenizer = self.tokenizer
        except Exception as e:
            print(f"Error loading tokenizer from '{tokenizer_path}': {e}")
            raise
    def readsvg(self, path: Union[str, Path]) -> str:
        """Read SVG file"""
        p = Path(path)
        return p.read_text(encoding='utf-8')

    def count_file(self, path: Union[str, Path]) -> Tuple[str, int]:
        """Count the number of tokens in a single SVG file"""
        p = Path(path)
        text = p.read_text(encoding='utf-8')
        length = len(self.tokenizer.encode(text))
        return p.name, length
    
    def count_tokens(self, text: str) -> int:
        """
        Calculate the number of tokens in the given text.
        
        :param text: Input string to calculate the number of tokens
        :return: The number of tokens
        """
        # .encode() method returns a list of token IDs, the length of which is the number of tokens
        try:
            # Use return_length to get the length, avoiding generating a complete list of token IDs
            return self.tokenizer.encode(text, return_length=True)[1]
        except AttributeError:
            # Fall back to the original method (old version tokenizer may not support)
            return len(self.tokenizer.encode(text))

    def get_token_ids(self, text: str) -> List[int]:
        """
        Get the list of token IDs for the given text.
        
        :param text: Input string to tokenize
        :return: A list containing token IDs
        """
        return self.tokenizer.encode(text)

    def get_tokens_str(self, text: str) -> List[str]:
        """
        Get the tokenization result (string form) for the given text.
        
        :param text: Input string to tokenize
        :return: A list containing readable token strings
        """
        return self.tokenizer.tokenize(text)

    def process(self, text: str) -> Dict[str, Union[int, List[int], List[str]]]:
        """
        Perform comprehensive tokenization on the given text, returning a dictionary containing all relevant information.
        
        :param text: Input string to process
        :return: A dictionary containing token number, token ID list, and token string list
        """
        token_ids = self.get_token_ids(text)
        token_strings = self.tokenizer.convert_ids_to_tokens(token_ids)
        
        return {
            'token_count': len(token_ids),
            'token_ids': token_ids,
            'tokens': token_strings
        }

    def count_chars(self, file_path: Union[str, Path]) -> int:
        """
        Return the number of characters in the given file, for quick filtering.
        :param file_path: SVG file path
        :return: The number of characters in the file
        """
        text = Path(file_path).read_text(encoding='utf-8')
        return len(text)
    
    def _count_line_tokens(self, line: str) -> int:
        return len(self.tokenizer.encode(line))

    def _count_line_chars(self, line: str) -> int:
        return len(line)
    
    def _count_svg_tokens(self, line: str, chat_template: str) -> int:
        data = json.loads(line)
        if chat_template == "InternVL":
            svg_code = data['conversations'][1]['value']
            return len(self.tokenizer.encode(svg_code))
        elif chat_template == 'alpaca':
            svg_code = data['output']
            return len(self.tokenizer.encode(svg_code))
        else:
            raise ValueError(f"Unsupported chat template: {chat_template}")
        
    def _count_svg_chars(self, line: str, chat_template: str) -> int:
        data = json.loads(line)
        if chat_template == "InternVL":
            svg_code = data['conversations'][1]['value']
            return len(svg_code)
        elif chat_template == 'alpaca':
            svg_code = data['output']
            return len(svg_code)
        else:
            raise ValueError(f"Unsupported chat template: {chat_template}")

    def find_shortest_svg(self, dir_path: str, n: int, save_path: Optional[str] = None, fastmode: bool = False) -> Dict[str, int]:
        """
        Given a directory containing several SVG files, count the token number of all SVGs, and return the names and token lengths of the shortest n files.
        If save_path is not None, copy the selected files to the directory and save the result dictionary as metadata.json.

        :param dir_path: Directory path containing SVG files
        :param n: Number of files to return
        :param save_path: Target save directory, default is None
        :param fastmode: (bool) Whether to enable fast character mode (parallel counting characters), default is False
        :return: Dict[str, int], key is the file name, value is the token number or character number
        """
        svg_paths = list(Path(dir_path).glob("*.svg"))
        if not svg_paths:
            return {}

        if fastmode:
            # Parallel fast mode: count characters
            with Pool(processes=cpu_count()) as pool:
                results = list(tqdm(pool.imap_unordered(_count_chars_file, svg_paths), total=len(svg_paths)))
        else:
            # Parallel counting tokens
            with Pool(processes=cpu_count(), initializer=_init_worker, initargs=(self.tokenizer_path,)) as pool:
                results = list(tqdm(pool.imap_unordered(_count_file, svg_paths), total=len(svg_paths)))

        # Sort and take the first n
        selected = sorted(results, key=lambda x: x[1])[:n]
        result_dict = {name: length for name, length in selected}

        # If save_path is specified, copy files and write metadata.json
        if save_path:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            for name, _ in selected:
                shutil.copy2(Path(dir_path) / name, save_dir / name)
            with (save_dir / "metadata.json").open("w", encoding="utf-8") as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=2)

        return result_dict
    
    def calculate_token_length_range(
        self, 
        input_path: List[str], 
        chat_template: str = "InternVL",
        processes: Optional[int] = None,
        batch_lines: int = 1024, 
        use_svg: bool = False, 
        fastmode: bool = False
    ) -> Dict[str, Union[int, float]]:
        """
        Calculate the token length distribution of SVG code in the file
        """
        processes = processes or cpu_count()
        files = []
        for file in input_path:
            file = Path(file)
            if file.is_dir():
                files.extend(list(file.glob("*.jsonl")))
            else:
                files.append(file)
                
        total_lines = 0
        for fp in files:
            with fp.open("r", encoding="utf-8") as f:
                total_lines += sum(1 for _ in f)
            
        from functools import partial
        
        if fastmode:
            if use_svg:
                worker_fn = partial(self._count_svg_chars, chat_template=chat_template)
            else:
                worker_fn = partial(self._count_line_chars)
        else:
            if use_svg:
                worker_fn = partial(self._count_svg_tokens, chat_template=chat_template)
            else:
                worker_fn = partial(self._count_line_tokens)
        
        pool_kwargs = {}
        if not fastmode:
            pool_kwargs = dict(initializer=_init_worker, initargs=(self.tokenizer_path,))
            
        from multiprocessing import Pool
        min_len, max_len = 100000, 0
        sum_len, cnt_lines = 0, 0
        with Pool(processes=processes, **pool_kwargs) as pool, \
            tqdm(total=total_lines, desc="Calculating token length") as pbar:
            for file in files:
                with open(file, 'r') as f:
                    while True:
                        batch = list(islice(f, batch_lines))
                        if not batch:
                            break
                        lengths = list(pool.imap(worker_fn, batch, chunksize=128 if fastmode else 32))
                        for length in lengths:
                            sum_len += length
                            cnt_lines += 1
                            if length < min_len:
                                min_len = length
                            if length > max_len:
                                max_len = length
                        pbar.update(len(batch))
        
        return {
            "min": min_len,
            "max": max_len,
            "avg": sum_len / cnt_lines,
            "cnt": cnt_lines,
        }
            
        
    def filter_svg_by_length(self, dir_path: str, output_path: str, min_length: int = 0, max_length: int = 25000, fastmode: bool = False, save_metadata: bool = False) -> Dict[str, int]:
        """
        Filter out SVG files with length in [min_length, max_length], and copy to the output path.
        
        :param dir_path: Input directory path containing SVG files
        :param output_path: Output directory path, will be created if it does not exist
        :param min_length: Minimum length threshold (token number or character number)
        :param max_length: Maximum length threshold (token number or character number)
        :param fastmode: Whether to enable fast character mode (count characters instead of tokens), default is False
        :param save_metadata: Whether to save metadata file, default is False
        :return: Dict[str, int], key is the name of the filtered file, value is its length
        """
        svg_paths = list(Path(dir_path).glob("*.svg"))
        if not svg_paths:
            print("No SVG files found")
            return {}

        print(f"Found {len(svg_paths)} SVG files, starting to count {'characters' if fastmode else 'tokens'}...")

        if fastmode:
            # Parallel fast mode: count characters
            with Pool(processes=cpu_count()) as pool:
                results = list(tqdm(pool.imap_unordered(_count_chars_file, svg_paths), 
                                  total=len(svg_paths), desc="Counting characters"))
        else:
            # Parallel mode: count tokens
            with Pool(processes=cpu_count(), initializer=_init_worker, initargs=(self.tokenizer_path,)) as pool:
                results = list(tqdm(pool.imap_unordered(_count_file, svg_paths), 
                                  total=len(svg_paths), desc="Counting tokens"))

        filtered_files = [(name, length) for name, length in results if length >= min_length and length <= max_length]
        
        if not filtered_files:
            print(f"No files found between {min_length} and {max_length}")
            return {}

        print(f"Found {len(filtered_files)} files (length between {min_length} and {max_length})")

        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory created/confirmed exists: {output_dir}")

        # Parallel copying filtered files
        print("Starting to copy files...")
        input_dir = Path(dir_path)
        copy_tasks = [(input_dir / name, output_dir / name) for name, _ in filtered_files]
        
        with Pool(processes=cpu_count()) as pool:
            list(tqdm(pool.imap_unordered(_copy_file, copy_tasks), 
                     total=len(copy_tasks), desc="Copying files"))

        result_dict = {name: length for name, length in filtered_files}

        # Optional: save metadata
        if save_metadata:
            metadata_file = output_dir / "metadata.json"
            with metadata_file.open("w", encoding="utf-8") as f:
                json.dump({
                    "total_files": len(filtered_files),
                    "max_length_threshold": max_length,
                    "mode": "characters" if fastmode else "tokens",
                    "files": result_dict
                }, f, ensure_ascii=False, indent=2)
            print(f"Metadata saved to: {metadata_file}")

        print(f"Processing completed! Copied {len(filtered_files)} files")

        return result_dict
    
    def filter_jsonl_by_range_dir(
            self,
            input_path: Union[str, Path],
            save_path: Union[str, Path],
            min_len: int,
            max_len: int,
            fastmode: bool = False,
            use_svg: bool = False,
            chat_template: str = "InternVL",
            processes: Optional[int] = None,
            batch_lines: int = 1024,
            recursive: bool = False,
        ) -> Dict[str, Union[int, str]]:
            """
            Filter all JSONL in the directory: keep samples with "line length" in [min_len, max_len], and write to save_path.
            - Length unit: token (default) or characters (fastmode=True)
            - Multi-process to count length; main process writes streamily, low memory usage
            - Boundary is closed interval: min_len <= length <= max_len

            :param input_path: Directory or jsonl file path containing several .jsonl
            :param save_path: Output JSONL file path (will overwrite same name file)
            :param min_len: Minimum length (inclusive)
            :param max_len: Maximum length (inclusive)
            :param fastmode: True=by characters; False=by tokenizer tokens
            :param use_svg: True=by SVG length; False=by JSONL line length
            :param processes: Number of processes (default CPU cores)
            :param batch_lines: Number of lines to send to the process pool per batch
            :param recursive: Whether to recursively filter subdirectories
            :return: Statistics information dictionary
            """
            input_path = Path(input_path)
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            if input_path.is_file():
                files = [input_path]
            else:
                files = sorted(input_path.rglob("*.jsonl") if recursive else input_path.glob("*.jsonl"))
            if not files:
                print(f"[filter] No .jsonl found in {input_path}")
                save_path.write_text("", encoding="utf-8")
                return {
                    "input": str(input_path),
                    "output": str(save_path),
                    "mode": "characters" if fastmode else "tokens",
                    "total_files": 0,
                    "total_lines": 0,
                    "kept": 0,
                    "dropped": 0,
                    "min": min_len,
                    "max": max_len,
                }

            total_lines = 0
            for fp in files:
                with fp.open("r", encoding="utf-8") as f:
                    total_lines += sum(1 for _ in f)

            processes = processes or cpu_count()
            
            from functools import partial
            if fastmode:
                if use_svg:
                    worker_fn = partial(self._count_svg_chars, chat_template=chat_template)
                else:
                    worker_fn = partial(self._count_line_chars)
            else:
                if use_svg:
                    worker_fn = partial(self._count_svg_tokens, chat_template=chat_template)
                else:
                    worker_fn = partial(self._count_line_tokens)
                
            pool_kwargs = {}
            if not fastmode:
                pool_kwargs = dict(initializer=_init_worker, initargs=(self.tokenizer_path,))
            from multiprocessing import Pool
            kept = 0
            dropped = 0

            with Pool(processes=processes, **pool_kwargs) as pool, \
                open(save_path, "w", encoding="utf-8") as fout, \
                tqdm(total=total_lines, desc="Filtering samples", unit="ln") as pbar:

                for fp in files:
                    with fp.open("r", encoding="utf-8") as fin:
                        while True:
                            batch = list(islice(fin, batch_lines))
                            if not batch:
                                break
                            lengths = list(pool.imap(worker_fn, batch, chunksize=128 if fastmode else 32))
                            for line, L in zip(batch, lengths):
                                if min_len <= L <= max_len:
                                    fout.write(line if line.endswith("\n") else (line + "\n"))
                                    kept += 1
                                else:
                                    dropped += 1
                            pbar.update(len(batch))

            return {
                "input": str(input_path),
                "output": str(save_path),
                "mode": "characters" if fastmode else "tokens",
                "total_files": len(files),
                "total_lines": total_lines,
                "kept": kept,
                "dropped": dropped,
                "min": min_len,
                "max": max_len,
            }
 
