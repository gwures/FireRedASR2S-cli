#!/usr/bin/env python3

import argparse
import logging
import os
import re
import sys
import warnings
from pathlib import Path
from typing import List

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

import config
from core.worker import MixedASRSystem
from services import FileService, SUPPORTED_AUDIO_EXTENSIONS, collect_audio_files, validate_file

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

_SRT_PUNC_PATTERN = re.compile(r'[][，。！？、；：""''（）【】《》,.!?;:\'"()<>]')


def format_srt_time(ms: int) -> str:
    hours = ms // 3600000
    ms %= 3600000
    minutes = ms // 60000
    ms %= 60000
    seconds = ms // 1000
    milliseconds = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


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
        "--nts", "--no-timestamp", action="store_true", help="不返回词级时间戳"
    )
    config_group.add_argument(
        "--ts", "--timestamp", action="store_true", help="返回词级时间戳(默认)"
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
    def on_error(path_str: str, error_msg: str):
        logger.error(f"{error_msg}: {path_str}")

    def on_skip(path_str: str, reason: str):
        if "不支持的文件格式" in reason:
            logger.warning(f"跳过{reason}: {path_str}")
        else:
            logger.info(f"跳过{reason}: {path_str}")

    return collect_audio_files(
        paths=args.paths,
        recursive=args.recursive,
        on_error=on_error,
        on_skip=on_skip,
    )


def process_files(files: List[Path], args):
    if args.nfp:
        use_fp16 = False
    else:
        use_fp16 = True

    if args.nts:
        return_timestamp = False
    else:
        return_timestamp = True

    max_batch_dur_s = args.dur
    use_bfd = args.BFD

    unified_output_dir = Path(args.output_dir) if args.output_dir else None
    if unified_output_dir:
        unified_output_dir.mkdir(parents=True, exist_ok=True)

    valid_files = []
    for file_path in files:
        is_valid, reason = validate_file(file_path)
        if is_valid:
            valid_files.append(file_path)
        else:
            logger.warning(f"跳过文件 {file_path.name}: {reason}")

    if not valid_files:
        logger.error("没有有效的文件需要处理")
        return

    logger.info(f"准备处理 {len(valid_files)} 个文件")

    asr_config = config.ASR_CONFIG.copy()
    asr_config["use_half"] = use_fp16
    asr_config["return_timestamp"] = return_timestamp

    worker_config = {
        "model_paths": config.MODEL_PATHS,
        "asr": asr_config,
        "vad": config.VAD_CONFIG,
        "punc": config.PUNC_CONFIG,
        "results_dir": str(config.TEMP_DIR),
        "max_batch_dur_s": max_batch_dur_s,
        "use_bfd": use_bfd,
    }

    wav_files_to_cleanup = []

    with FileService(str(config.TEMP_DIR), str(config.TEMP_DIR)) as file_service:
        try:
            logger.info("正在转换音频格式...")

            results = file_service.convert([str(f) for f in valid_files])

            audio_files = []
            conversion_failed_count = 0
            for result in results:
                if result.success:
                    wav_files_to_cleanup.append(Path(result.output_path))
                    audio_files.append({
                        "audio_path": result.output_path,
                        "original_file": Path(result.input_path)
                    })
                else:
                    conversion_failed_count += 1
                    logger.warning(f"转换失败 {result.input_path}: {result.error_message}")

            if not audio_files:
                logger.error("没有任务被创建")
                return

            logger.info("初始化 ASR 系统...")
            asr_system = MixedASRSystem(worker_config)

            logger.info(f"开始批量处理 {len(audio_files)} 个文件...")
            final_results, perf_stats = asr_system.batch_process(
                audio_files, max_batch_dur_s=max_batch_dur_s
            )

            completed_count = perf_stats.get("success_count", len(final_results))
            failed_count = perf_stats.get("error_count", 0) + conversion_failed_count
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

                srt_index = 1
                with open(output_srt, "w", encoding="utf-8") as f:
                    for sentence in result.get("sentences", []):
                        start_ms = sentence.get("start_ms", 0)
                        end_ms = sentence.get("end_ms", 0)
                        text = sentence.get("text", "")

                        srt_text = _SRT_PUNC_PATTERN.sub('', text)
                        if not srt_text.strip():
                            continue

                        start_time_srt = format_srt_time(start_ms)
                        end_time_srt = format_srt_time(end_ms)

                        f.write(f"{srt_index}\n")
                        f.write(f"{start_time_srt} --> {end_time_srt}\n")
                        f.write(f"{srt_text}\n\n")
                        srt_index += 1

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

                output_info = (
                    str(file_output_dir) if not unified_output_dir else str(file_output_dir)
                )
                logger.info(
                    f"[{i+1}/{len(final_results)}] 完成: {original_file.name} "
                    f"(音频: {audio_dur:.2f}s) -> {output_info}"
                )

            logger.info("=" * 60)
            logger.info(f"处理完成! 成功: {completed_count}, 失败: {failed_count}")
            if unified_output_dir:
                logger.info(f"输出目录: {unified_output_dir.resolve()}")
            else:
                logger.info("输出位置: 各源文件同目录")
            logger.info("")
            logger.info("=" * 60)
            logger.info("【性能统计】")
            logger.info(f"总音频时长: {perf_stats['total_audio_dur_s']:.2f}s")
            logger.info(f"总处理用时: {perf_stats['total_processing_s']:.2f}s")
            logger.info(f"平均 RTF: {perf_stats['avg_rtf']:.3f}x")
            logger.info(f"整体 RTF: {perf_stats['overall_rtf']:.3f}x")
            logger.info("=" * 60)

        finally:
            if wav_files_to_cleanup:
                logger.info("清理临时 WAV 文件...")
                success, fail = file_service.cleanup_temp_files(
                    [str(f) for f in wav_files_to_cleanup]
                )
                if fail > 0:
                    logger.warning(f"清理临时文件失败: {fail} 个")


def main():
    args = parse_args()

    if args.verbose:
        root_logger.setLevel(logging.DEBUG)
    elif args.quiet:
        root_logger.setLevel(logging.ERROR)

    files = collect_files(args)

    if not files:
        logger.error("没有找到符合条件的文件")
        sys.exit(1)

    logger.info(f"找到 {len(files)} 个文件")

    try:
        process_files(files, args)
    except KeyboardInterrupt:
        logger.info("\n用户中断")
        sys.exit(130)
    except Exception as e:
        logger.error(f"处理出错: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
