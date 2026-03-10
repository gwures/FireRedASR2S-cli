import atexit
import logging
import os
import subprocess
import threading
import time
import uuid
import weakref
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Set, Tuple, Union

logger = logging.getLogger("fireredasr2s.file_service")

_file_service_instances: Set[weakref.ref] = set()


def _cleanup_all_instances():
    for ref in list(_file_service_instances):
        instance = ref()
        if instance is not None:
            try:
                instance._cleanup_tracked_files()
            except Exception as e:
                logger.warning(f"Error during atexit cleanup: {e}")
    _file_service_instances.clear()


atexit.register(_cleanup_all_instances)

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".aac",
    ".ogg",
    ".m4a",
    ".wma",
    ".opus",
    ".amr",
    ".ape",
    ".aiff",
    ".aif",
    ".au",
    ".bwf",
    ".ac3",
    ".dts",
    ".eac3",
    ".thd",
    ".truehd",
    ".mp2",
    ".vqf",
    ".sd2",
    ".mp4",
    ".avi",
    ".mkv",
    ".mov",
    ".wmv",
    ".webm",
    ".flv",
    ".ts",
    ".mts",
    ".m2ts",
    ".vob",
    ".3gp",
    ".mxf",
    ".asf",
    ".rmvb",
    ".rm",
    ".ogv",
    ".f4v",
    ".m4v",
    ".divx",
    ".xvid",
}


def validate_file(file_path: Path) -> Tuple[bool, str]:
    """Validate if a file is valid for audio processing.

    Args:
        file_path: Path to the file to validate.

    Returns:
        Tuple of (is_valid: bool, reason: str).
    """
    if not file_path.exists():
        return False, "文件不存在"

    if not file_path.is_file():
        return False, "不是文件"

    if file_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        return False, f"不支持的文件格式: {file_path.suffix}"

    return True, "有效"


def collect_audio_files(
    paths: List[str],
    recursive: bool = False,
    on_error: Optional[Callable[[str, str], None]] = None,
    on_skip: Optional[Callable[[str, str], None]] = None,
) -> List[Path]:
    """Collect audio files from given paths.

    Args:
        paths: List of file or directory paths to search.
        recursive: Whether to search directories recursively.
        on_error: Optional callback(path_str, error_msg) for path errors.
        on_skip: Optional callback(path_str, reason) for skipped files.

    Returns:
        List of unique audio file paths.
    """
    files: List[Path] = []
    seen_names: Set[str] = set()

    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            if on_error:
                on_error(path_str, "路径不存在")
            continue

        if path.is_file():
            if path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
                abs_path = path.resolve()
                filename = abs_path.name
                if filename not in seen_names:
                    files.append(abs_path)
                    seen_names.add(filename)
                else:
                    if on_skip:
                        on_skip(path_str, "重复文件")
            else:
                if on_skip:
                    on_skip(path_str, f"不支持的文件格式: {path.suffix}")
        elif path.is_dir():
            pattern = "**/*" if recursive else "*"
            for found_path in path.glob(pattern):
                if (
                    found_path.is_file()
                    and found_path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
                ):
                    abs_path = found_path.resolve()
                    filename = abs_path.name
                    if filename not in seen_names:
                        files.append(abs_path)
                        seen_names.add(filename)

    return files


class FileServiceError(Exception):
    pass


class DirectoryCreationError(FileServiceError):
    pass


class FFmpegNotFoundError(FileServiceError):
    pass


class FFmpegConversionError(FileServiceError):
    def __init__(
        self,
        message: str,
        returncode: Optional[int] = None,
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
    ):
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@dataclass
class ConversionResult:
    input_path: str
    output_path: Optional[str]
    success: bool
    error_message: Optional[str] = None
    exception: Optional[Exception] = None


class FileService:
    def __init__(self, temp_dir: str, output_dir: str, auto_cleanup: bool = True):
        self.temp_dir = Path(temp_dir)
        self.output_dir = Path(output_dir)
        self._tracked_temp_files: Set[str] = set()
        self._lock = threading.Lock()
        self._auto_cleanup = auto_cleanup
        self._shutdown_called = False
        try:
            self.temp_dir.mkdir(exist_ok=True)
            self.output_dir.mkdir(exist_ok=True)
        except OSError as e:
            raise DirectoryCreationError(f"Failed to create directory: {e}") from e
        cpu_count = os.cpu_count() or 4
        self._max_workers = max(2, cpu_count // 2)
        self._ffmpeg_threads = max(1, cpu_count // self._max_workers)
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_workers, thread_name_prefix="ffmpeg"
        )
        if auto_cleanup:
            _file_service_instances.add(weakref.ref(self))

    def __enter__(self) -> "FileService":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(wait=True)
        if self._auto_cleanup:
            self._cleanup_tracked_files()
        return False

    def _track_temp_file(self, file_path: str):
        with self._lock:
            self._tracked_temp_files.add(file_path)

    def _untrack_temp_file(self, file_path: str):
        with self._lock:
            self._tracked_temp_files.discard(file_path)

    def _cleanup_tracked_files(self):
        with self._lock:
            if not self._tracked_temp_files:
                return
            files_to_clean = list(self._tracked_temp_files)
            self._tracked_temp_files.clear()

        logger.info(f"Cleaning up {len(files_to_clean)} tracked temp files")
        for file_path in files_to_clean:
            try:
                path = Path(file_path)
                if path.exists():
                    path.unlink()
                    logger.debug(f"Deleted tracked temp file: {path}")
            except Exception as e:
                logger.warning(f"Failed to delete temp file {file_path}: {e}")

    def _get_output_path(self, input_path: Path) -> Path:
        """Generate unique output path for WAV conversion.

        Uses original filename prefix + UUID for uniqueness.
        Format: {original_stem}_{uuid8}.wav

        Args:
            input_path: Input file path.

        Returns:
            Unique output path for the converted WAV file.
        """
        safe_stem = self._sanitize_filename(input_path.stem)
        unique_suffix = uuid.uuid4().hex[:8]
        filename = f"{safe_stem}_{unique_suffix}.wav"
        return self.temp_dir / filename

    def _sanitize_filename(self, name: str) -> str:
        """Sanitize filename by removing dangerous characters.

        Args:
            name: Original filename (without extension).

        Returns:
            Sanitized filename safe for all filesystems.
        """
        dangerous_chars = ["/", "\\", ":", "*", "?", '"', "<", ">", "|", "\0"]
        sanitized = "".join(c if c not in dangerous_chars else "_" for c in name)
        if not sanitized or sanitized.strip() == "":
            sanitized = f"file_{int(time.time())}"
        return sanitized[:64]

    def _execute_conversion(self, input_path: Path, output_path: Path) -> str:
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if not input_path.is_file():
            raise FileServiceError(f"Input path is not a file: {input_path}")

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(input_path),
                    "-threads",
                    str(self._ffmpeg_threads),
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-acodec",
                    "pcm_s16le",
                    "-f",
                    "wav",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                encoding="utf-8",
                errors="ignore",
                shell=False,
                timeout=600,
            )
            logger.info(f"Converted {input_path} to {output_path}")
            self._track_temp_file(str(output_path))
            return str(output_path)
        except FileNotFoundError:
            raise FFmpegNotFoundError(
                "FFmpeg executable not found. Please install FFmpeg."
            )
        except subprocess.TimeoutExpired as e:
            error_msg = f"FFmpeg conversion timed out for {input_path} (timeout=600s)"
            logger.error(error_msg)
            raise FileServiceError(error_msg) from e
        except subprocess.CalledProcessError as e:
            error_msg = (
                f"FFmpeg conversion failed for {input_path}\n"
                f"Return code: {e.returncode}\n"
                f"Stderr: {e.stderr}\n"
                f"Stdout: {e.stdout}"
            )
            logger.error(error_msg)
            raise FFmpegConversionError(
                f"Conversion failed: {e.stderr}",
                returncode=e.returncode,
                stdout=e.stdout,
                stderr=e.stderr,
            ) from e
        except PermissionError as e:
            raise PermissionError(f"Permission denied writing to {output_path}") from e
        except OSError as e:
            raise FileServiceError(f"OS error during conversion: {e}") from e

    def _convert_single_file(self, input_path: str) -> ConversionResult:
        """Internal method that performs conversion and returns detailed result.

        This is the single entry point for all conversion logic, ensuring
        consistent error handling and logging.

        Args:
            input_path: Path to the input audio file.

        Returns:
            ConversionResult with detailed success/failure information.
        """
        input_path_obj = Path(input_path)
        output_path = self._get_output_path(input_path_obj)

        try:
            output = self._execute_conversion(input_path_obj, output_path)
            return ConversionResult(
                input_path=input_path, output_path=output, success=True
            )
        except FFmpegConversionError as e:
            error_msg = f"Return code {e.returncode}: {e.stderr}"
            logger.error(f"Conversion failed for {input_path}: {error_msg}")
            return ConversionResult(
                input_path=input_path,
                output_path=None,
                success=False,
                error_message=error_msg,
                exception=e,
            )
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Conversion failed for {input_path}: {error_msg}")
            return ConversionResult(
                input_path=input_path,
                output_path=None,
                success=False,
                error_message=error_msg,
                exception=e,
            )

    def convert(
        self,
        input_paths: List[str],
        progress_callback: Optional[
            Callable[[int, int, ConversionResult], None]
        ] = None,
    ) -> List[ConversionResult]:
        """Convert audio files to WAV format concurrently.

        This is the unified conversion method supporting both single and multiple files.
        It is fault-tolerant: a single file's failure will not interrupt the batch.

        Args:
            input_paths: List of paths to input audio files. For a single file,
                        pass a single-element list.
            progress_callback: Optional callback(completed, total, result) for progress.

        Returns:
            List of ConversionResult objects in the same order as input_paths.
            Each result contains:
            - success: bool indicating if conversion succeeded
            - output_path: path to WAV file (None if failed)
            - error_message: error description (None if succeeded)
            - exception: original exception (None if succeeded)

        Examples:
            Single file:
                results = service.convert(["audio.mp3"])
                if results[0].success:
                    print(f"Output: {results[0].output_path}")

            Multiple files with progress:
                def on_progress(done, total, result):
                    print(f"[{done}/{total}] {result.input_path}")

                results = service.convert(["a.mp3", "b.wav"], on_progress)
                for r in results:
                    if r.success:
                        print(f"OK: {r.output_path}")
                    else:
                        print(f"FAIL: {r.error_message}")
        """
        future_to_index = {
            self._executor.submit(self._convert_single_file, path): idx
            for idx, path in enumerate(input_paths)
        }
        results: List[ConversionResult] = [None] * len(input_paths)  # type: ignore
        completed = 0

        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                result = future.result()
                results[idx] = result
            except Exception as e:
                results[idx] = ConversionResult(
                    input_path=input_paths[idx],
                    output_path=None,
                    success=False,
                    error_message=str(e),
                    exception=e,
                )
            completed += 1
            if progress_callback:
                progress_callback(completed, len(input_paths), results[idx])

        return results

    def cleanup_temp_files(
        self, file_paths: List[str], on_progress: Optional[Callable[[str], None]] = None
    ) -> Tuple[int, int]:
        """Clean up temporary files.

        Args:
            file_paths: List of file paths to delete.
            on_progress: Optional callback(file_path) for progress updates.

        Returns:
            Tuple of (success_count, fail_count).
        """
        success_count = 0
        fail_count = 0

        for file_path in file_paths:
            path = Path(file_path)
            try:
                if path.exists():
                    path.unlink()
                    success_count += 1
                    if on_progress:
                        on_progress(str(path))
                    logger.debug(f"Deleted temp file: {path}")
                self._untrack_temp_file(file_path)
            except Exception as e:
                fail_count += 1
                logger.warning(f"Failed to delete temp file {path}: {e}")

        return success_count, fail_count

    def shutdown(self, wait: bool = True, cleanup: bool = False):
        """Shutdown the thread pool executor gracefully.

        Args:
            wait: If True, wait for pending tasks to complete.
            cleanup: If True, also cleanup all tracked temp files.
        """
        if self._shutdown_called:
            return
        self._shutdown_called = True
        self._executor.shutdown(wait=wait)
        if cleanup:
            self._cleanup_tracked_files()
        logger.info("FileService executor shutdown")
