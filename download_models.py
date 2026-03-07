import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
)
logger = logging.getLogger("fireredasr2s.download_models")


def download_model(model_id, local_dir):
    """使用 modelscope 下载模型"""
    try:
        from modelscope import snapshot_download

        logger.info(f"正在下载模型: {model_id}")
        logger.info(f"保存路径: {local_dir}")

        snapshot_download(model_id=model_id, local_dir=local_dir)
        logger.info(f"模型 {model_id} 下载成功！")
        return True
    except ImportError:
        logger.error("请先安装 modelscope: pip install -U modelscope")
        return False
    except (OSError, IOError, ConnectionError, RuntimeError) as e:
        logger.error(f"下载失败: {e}")
        return False


def main():
    logger.info("FireRedASR2S 模型下载脚本")
    logger.info("=" * 50)

    base_dir = Path("./pretrained_models")
    base_dir.mkdir(exist_ok=True)

    models = [
        ("xukaituo/FireRedASR2-AED", base_dir / "FireRedASR2-AED"),
        ("xukaituo/FireRedVAD", base_dir / "FireRedVAD"),
        ("xukaituo/FireRedPunc", base_dir / "FireRedPunc"),
    ]

    all_success = True
    for model_id, local_dir in models:
        success = download_model(model_id, str(local_dir))
        if not success:
            all_success = False

    if all_success:
        logger.info("所有模型下载完成！")
        logger.info("请运行: python cli.py <音频文件> 开始转写")
        logger.info("查看帮助: python cli.py --help")
    else:
        logger.warning("部分模型下载失败，请检查错误信息并重新运行脚本")


if __name__ == "__main__":
    main()
