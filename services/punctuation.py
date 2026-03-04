import asyncio
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("fireredasr2s.punctuation")


class PunctuationService:
    def __init__(self, config: Dict):
        self.config = config
        self.endpoint = config.get("endpoint", "https://api.openai.com/v1/chat/completions")
        self.api_key = config.get("api_key", "")
        self.model = config.get("model", "gpt-3.5-turbo")
        self.prompt_template = config.get("prompt", "请为以下无标点文本添加适当的标点符号，只返回加标点后的文本，不要其他解释：\n{text}")
        self.max_timeout = config.get("max_timeout", 300)
        self.max_concurrent = config.get("max_concurrent", 5)
        self._async_client = None
        self._semaphore = None

    def _get_async_client(self):
        if self._async_client is None:
            import openai
            base_url = self.endpoint.replace("/chat/completions", "")
            self._async_client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=base_url,
                timeout=self.max_timeout
            )
        return self._async_client

    async def add_punctuation_async(self, text: str) -> str:
        if not text or not text.strip():
            return text

        prompt = self.prompt_template.replace("{text}", text)

        try:
            client = self._get_async_client()
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=4096
            )

            if response.choices and len(response.choices) > 0:
                punctuated_text = response.choices[0].message.content
                logger.info(f"标点添加成功: {text[:20]}... -> {punctuated_text[:20]}...")
                return punctuated_text.strip()
            else:
                logger.warning("LLM返回格式异常，回落使用原文")
                return text

        except ImportError as e:
            logger.warning(f"openai库未安装，无法添加标点: {e}，回落使用原文")
            return text
        except Exception as e:
            logger.warning(f"标点添加失败: {e}，回落使用原文")
            return text

    async def add_punctuation_batch_async(self, texts: List[str]) -> List[str]:
        if not texts:
            return []

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_with_semaphore(text: str) -> str:
            async with semaphore:
                return await self.add_punctuation_async(text)

        tasks = [process_with_semaphore(text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"标点处理失败 (索引 {i}): {result}")
                final_results.append(texts[i])
            else:
                final_results.append(result)
        
        return final_results

    def __del__(self):
        self._async_client = None


def create_punctuation_service(config: Dict) -> Optional[PunctuationService]:
    if not config.get("enabled", False):
        logger.info("标点功能未启用")
        return None

    if not config.get("api_key"):
        logger.warning("未配置api_key，标点功能不可用")
        return None

    try:
        service = PunctuationService(config)
        logger.info(f"标点服务已创建，使用endpoint: {service.endpoint}, model: {service.model}")
        return service
    except Exception as e:
        logger.warning(f"创建标点服务失败: {e}，将不使用标点功能")
        return None