#!/usr/bin/env python3

import argparse
import asyncio
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import List, Set

warnings.filterwarnings(
    "ignore", category=DeprecationWarning, message="pkg_resources is deprecated"
)

fire_red_asr_path = Path(__file__).parent / "FireRedASR2S"
if fire_red_asr_path.exists():
    sys.path.insert(0, str(fire_red_asr_path))

sys.path.insert(0, str(Path(__file__).parent))

os.environ["PYTHONPATH"] = (
    f"{str(fire_red_asr_path)};{os.environ.get('PYTHONPATH', '')}"
)

import config  # noqa: E402
from services import FileService  # noqa: E402
from services.punctuation import create_punctuation_service  # noqa: E402

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

formatter = logging.Formatter(
    "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
root_logger.addHandler(stream_handler)

logger = logging.getLogger("fireredasr2s.cli")

QUIET_MODE = False


def format_srt_time(ms: int) -> str:
    hours = ms // 3600000
    ms %= 3600000
    minutes = ms // 60000
    ms %= 60000
    seconds = ms // 1000
    milliseconds = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def print_progress(msg: str):
    """打印进度信息，即使在 --quiet 模式下也显示"""
    print(msg, flush=True)


def print_error(msg: str):
    """打印错误信息"""
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="FireRedASR2S 命令行工具 - 语音转文字",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  处理单个文件:
    python cli.py audio.mp3
  处理多个文件:
    python cli.py file1.mp3 file2.wav
  处理目录(递归):
    python cli.py /path/to/audio -r
  处理多个目录:
    python cli.py dir1 dir2 file.mp3 -r
  混合处理文件和目录:
    python cli.py audio.mp3 /path/to/videos -r
  带时间戳输出:
    python cli.py audio.mp3 --ts

  禁用 FP16:
    python cli.py audio.mp3 --nfp

  自定义输出目录:
    python cli.py audio.mp3 -o my_output

  使用 BFD 算法分批:
    python cli.py audio.mp3 --BFD
        """,
    )

    input_group = parser.add_argument_group("输入选项")
    input_group.add_argument(
        "paths", nargs="+", help="一个或多个文件/目录路径(自动识别类型)"
    )
    input_group.add_argument(
        "-r", "--recursive", action="store_true", help="递归搜索子目录中的文件"
    )

    output_group = parser.add_argument_group("输出选项")
    output_group.add_argument(
        "-o", "--output-dir", help="输出目录(默认: 各源文件同目录)"
    )

    config_group = parser.add_argument_group("配置选项")
    config_group.add_argument(
        "--fp", "--fp16", action="store_true", help="启用 FP16 精度加速(默认)"
    )
    config_group.add_argument(
        "--nfp", "--no-fp16", action="store_true", help="禁用 FP16"
    )
    config_group.add_argument(
        "--nts", "--no-timestamp", action="store_true", help="不返回词级时间戳(默认)"
    )
    config_group.add_argument(
        "--ts", "--timestamp", action="store_true", help="返回词级时间戳"
    )

    config_group.add_argument(
        "--dur",
        "--max-batch-dur-s",
        type=int,
        default=64,
        metavar="SECONDS",
        help="单batch最大总时长(秒，默认: 64)",
    )
    config_group.add_argument(
        "--BFD",
        action="store_true",
        help="使用 BFD (Best-Fit Decreasing) 算法进行分批（默认使用 WFD 算法）",
    )

    other_group = parser.add_argument_group("其他选项")
    other_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="DEBUG 及以上所有日志（含调试信息）",
    )
    other_group.add_argument(
        "-q", "--quiet", action="store_true", help="静默模式：仅 ERROR 级别日志"
    )

    return parser.parse_args()


def collect_files(args) -> List[Path]:
    files: List[Path] = []
    seen: Set[Path] = set()

    extensions = SUPPORTED_AUDIO_EXTENSIONS

    for path_str in args.paths:
        path = Path(path_str)
        if not path.exists():
            logger.error(f"路径不存在: {path_str}")
            continue

        if path.is_file():
            if path.suffix.lower() in extensions:
                abs_path = path.resolve()
                if abs_path not in seen:
                    files.append(abs_path)
                    seen.add(abs_path)
            else:
                logger.warning(f"跳过不支持的文件格式: {path_str}")
        elif path.is_dir():
            pattern = "**/*" if args.recursive else "*"
            for found_path in path.glob(pattern):
                if found_path.is_file() and found_path.suffix.lower() in extensions:
                    abs_path = found_path.resolve()
                    if abs_path not in seen:
                        files.append(abs_path)
                        seen.add(abs_path)

    return sorted(files)


def validate_file(file_path: Path) -> tuple[bool, str]:
    if not file_path.exists():
        return False, "文件不存在"

    if not file_path.is_file():
        return False, "不是文件"

    if file_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        return False, f"不支持的文件格式: {file_path.suffix}"

    return True, "有效"


async def process_files_async(files: List[Path], args):
    if args.nfp:
        use_fp16 = False
    else:
        use_fp16 = True

    if args.ts:
        plain_text_mode = False
    else:
        plain_text_mode = True

    max_batch_dur_s = args.dur
    use_bfd = args.BFD

    return_timestamp = not plain_text_mode

    unified_output_dir = Path(args.output_dir) if args.output_dir else None
    if unified_output_dir:
        unified_output_dir.mkdir(parents=True, exist_ok=True)

    valid_files = []
    for file_path in files:
        is_valid, reason = validate_file(file_path)
        if is_valid:
            valid_files.append(file_path)
        else:
            if not QUIET_MODE:
                logger.warning(f"跳过文件 {file_path.name}: {reason}")

    if not valid_files:
        print_error("没有有效的文件需要处理")
        return

    print_progress(f"准备处理 {len(valid_files)} 个文件")

    fire_red_asr_path = Path(__file__).parent / "FireRedASR2S"
    if fire_red_asr_path.exists():
        sys.path.insert(0, str(fire_red_asr_path))

    from core.worker import MixedASRSystem

    asr_config = config.ASR_CONFIG.copy()
    asr_config["use_half"] = use_fp16
    asr_config["return_timestamp"] = return_timestamp

    worker_config = {
        "model_paths": config.MODEL_PATHS,
        "asr": asr_config,
        "vad": config.VAD_CONFIG,
        "results_dir": str(config.TEMP_DIR),
        "max_batch_dur_s": max_batch_dur_s,
        "use_bfd": use_bfd,
    }

    wav_files_to_cleanup = []

    try:
        print_progress("正在转换音频格式...")

        file_service = FileService(str(config.TEMP_DIR), str(config.TEMP_DIR))
        wav_paths = file_service.convert_multiple_to_wav([str(f) for f in valid_files])

        for wav_path in wav_paths:
            if wav_path:
                wav_files_to_cleanup.append(Path(wav_path))

        audio_files = []
        for original_file, wav_path in zip(valid_files, wav_paths):
            if wav_path:
                audio_files.append(
                    {"audio_path": wav_path, "original_file": original_file}
                )

        if not audio_files:
            print_error("没有任务被创建")
            return

        print_progress("初始化 ASR 系统...")
        asr_system = MixedASRSystem(worker_config)

        print_progress(f"开始批量处理 {len(audio_files)} 个文件...")
        final_results, perf_stats = asr_system.batch_process(
            audio_files, max_batch_dur_s=max_batch_dur_s
        )

        completed_count = 0
        failed_count = 0
        output_files_info = []

        for i, result in enumerate(final_results):
            original_file = result.get("original_file", None)
            if original_file is None:
                original_file = Path(result["wav_path"])
            else:
                original_file = Path(original_file)

            task_id = original_file.stem
            audio_dur = result.get("dur_s", 0)

            if unified_output_dir:
                file_output_dir = unified_output_dir
            else:
                file_output_dir = original_file.parent

            output_srt = file_output_dir / f"{task_id}.srt"
            output_txt = file_output_dir / f"{task_id}.txt"

            with open(output_srt, "w", encoding="utf-8") as f:
                for j, sentence in enumerate(result.get("sentences", []), 1):
                    start_ms = sentence.get("start_ms", 0)
                    end_ms = sentence.get("end_ms", 0)
                    text = sentence.get("text", "")

                    start_time_srt = format_srt_time(start_ms)
                    end_time_srt = format_srt_time(end_ms)

                    f.write(f"{j}\n")
                    f.write(f"{start_time_srt} --> {end_time_srt}\n")
                    f.write(f"{text}\n\n")

            with open(output_txt, "w", encoding="utf-8") as f:
                f.write(result.get("text", ""))

            output_files_info.append(
                {
                    "original_file": original_file,
                    "output_txt": output_txt,
                    "text": result.get("text", ""),
                    "file_output_dir": file_output_dir,
                    "audio_dur": audio_dur,
                }
            )

            completed_count += 1
            output_info = (
                str(file_output_dir) if not unified_output_dir else str(file_output_dir)
            )
            print_progress(
                f"[{completed_count}/{len(final_results)}] 完成: {original_file.name} "
                f"(音频: {audio_dur:.2f}s) -> {output_info}"
            )

        print_progress("=" * 60)
        print_progress(f"处理完成! 成功: {completed_count}, 失败: {failed_count}")
        if unified_output_dir:
            print_progress(f"输出目录: {unified_output_dir.resolve()}")
        else:
            print_progress("输出位置: 各源文件同目录")
        print_progress("")
        print_progress("=" * 60)
        print_progress("【性能统计】")
        print_progress(f"总音频时长: {perf_stats['total_audio_dur_s']:.2f}s")
        print_progress(f"总处理用时: {perf_stats['total_processing_s']:.2f}s")
        print_progress(f"平均 RTF: {perf_stats['avg_rtf']:.3f}x")
        print_progress(f"整体 RTF: {perf_stats['overall_rtf']:.3f}x")
        print_progress("=" * 60)

        print_progress("")
        print_progress("初始化标点服务...")
        punctuation_service = create_punctuation_service(config.PUNCTUATION_CONFIG)

        if punctuation_service is not None and output_files_info:
            print_progress(
                f"开始并发对 {len(output_files_info)} 个文本文件进行标点处理..."
            )

            texts = [file_info["text"] for file_info in output_files_info]
            punctuated_texts = await punctuation_service.add_punctuation_batch_async(
                texts
            )

            punct_count = 0
            failed_count = 0
            for file_info, punctuated_text in zip(output_files_info, punctuated_texts):
                try:
                    if punctuated_text is None:
                        failed_count += 1
                        logger.warning(
                            f"标点处理失败: {file_info['original_file'].name}"
                        )
                        continue

                    original_file = file_info["original_file"]
                    file_output_dir = file_info["file_output_dir"]

                    task_id = original_file.stem
                    output_txt_pc = file_output_dir / f"{task_id}_pc.txt"

                    with open(output_txt_pc, "w", encoding="utf-8") as f:
                        f.write(punctuated_text)

                    punct_count += 1
                    print_progress(
                        f"[{punct_count}/{len(output_files_info)}] 标点完成: {original_file.name} -> {output_txt_pc.name}"
                    )
                except Exception as e:
                    failed_count += 1
                    logger.warning(
                        f"标点结果写入失败 {file_info['original_file'].name}: {e}"
                    )

            print_progress("=" * 60)
            print_progress(f"标点处理完成: 成功 {punct_count}, 失败 {failed_count}")
        elif punctuation_service is None:
            print_progress("标点服务未启用或配置不完整，跳过标点处理")

    finally:
        cleanup_errors = []
        try:
            file_service.shutdown()
        except Exception as e:
            cleanup_errors.append(f"file_service.shutdown: {e}")
        try:
            if wav_files_to_cleanup:
                if not QUIET_MODE:
                    print_progress("清理临时 WAV 文件...")
                for wav_file in wav_files_to_cleanup:
                    try:
                        if wav_file.exists():
                            wav_file.unlink()
                            if not QUIET_MODE:
                                logger.debug(f"已删除: {wav_file}")
                    except Exception as e:
                        logger.warning(f"删除临时文件失败 {wav_file}: {e}")
        except Exception as e:
            cleanup_errors.append(f"temp_files_cleanup: {e}")
        if cleanup_errors and not QUIET_MODE:
            logger.warning(f"清理过程中出现错误: {'; '.join(cleanup_errors)}")


def main():
    global QUIET_MODE
    args = parse_args()

    QUIET_MODE = args.quiet

    if args.verbose:
        root_logger.setLevel(logging.DEBUG)
    elif args.quiet:
        root_logger.setLevel(logging.ERROR)

    files = collect_files(args)

    if not files:
        print_error("没有找到符合条件的文件")
        sys.exit(1)

    print_progress(f"找到 {len(files)} 个文件")

    try:
        asyncio.run(process_files_async(files, args))
    except KeyboardInterrupt:
        print_progress("\n用户中断")
        sys.exit(130)
    except Exception as e:
        print_error(f"处理出错: {e}")
        if not QUIET_MODE:
            logger.error(f"处理出错: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
