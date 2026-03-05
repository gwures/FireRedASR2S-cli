from pathlib import Path


def download_model(model_id, local_dir):
    """使用 modelscope 下载模型"""
    try:
        from modelscope import snapshot_download

        print(f"正在下载模型: {model_id}")
        print(f"保存路径: {local_dir}")

        snapshot_download(model_id=model_id, local_dir=local_dir)
        print(f"✅ 模型 {model_id} 下载成功！\n")
        return True
    except ImportError:
        print("❌ 请先安装 modelscope: pip install -U modelscope")
        return False
    except (OSError, IOError, ConnectionError, RuntimeError) as e:
        print(f"❌ 下载失败: {e}")
        return False


def main():
    print("FireRedASR2S 模型下载脚本")
    print("=" * 50)

    base_dir = Path("./pretrained_models")
    base_dir.mkdir(exist_ok=True)

    models = [
        ("xukaituo/FireRedASR2-AED", base_dir / "FireRedASR2-AED"),
        ("xukaituo/FireRedVAD", base_dir / "FireRedVAD"),
    ]

    all_success = True
    for model_id, local_dir in models:
        success = download_model(model_id, str(local_dir))
        if not success:
            all_success = False

    if all_success:
        print("🎉 所有模型下载完成！")
        print("请运行: python cli.py -f <音频文件> 开始转写")
        print("查看帮助: python cli.py --help")
    else:
        print("⚠️  部分模型下载失败，请检查错误信息并重新运行脚本")


if __name__ == "__main__":
    main()
