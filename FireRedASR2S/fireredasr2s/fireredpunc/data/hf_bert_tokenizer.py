# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu, Junjie Chen)

import logging
import re
import traceback

from transformers import BertTokenizer

logger = logging.getLogger(__name__)

_RE_PURE_CN = re.compile(r"^[^a-zA-Z0-9']+$")
_RE_BERT_PREFIX = re.compile(r"^##")


class HfBertTokenizer:
    def __init__(self, huggingface_tokenizer_dir):
        self.tokenizer = BertTokenizer.from_pretrained(huggingface_tokenizer_dir)

    def tokenize(self, text, recover_unk=False):
        tokens = self.tokenizer.tokenize(text)
        tokens_id = self.tokenizer.convert_tokens_to_ids(tokens)
        if recover_unk:
            try:
                tokens = self._recover_unk(text.lower(), tokens)
            except Exception:
                traceback.print_exc()
        return tokens, tokens_id

    def _recover_unk(self, text, tokens):
        if "[UNK]" not in tokens:
            return tokens

        new_tokens = []
        text_no_space = text.replace(" ", "")

        if _RE_PURE_CN.match(text):
            tmp_text = text_no_space
            if len(tmp_text) == len(tokens):
                success = True
                for t, tok in zip(tmp_text, tokens):
                    if tok != "[UNK]" and t != tok:
                        success = False
                        break
                    new_tokens.append(t)
                if success:
                    return new_tokens
        new_tokens = []

        text_pos = 0
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token == "[UNK]":
                unk_count = 0
                j = i
                while j < len(tokens) and tokens[j] == "[UNK]":
                    unk_count += 1
                    j += 1

                post_token = ""
                if j < len(tokens):
                    post_token = _RE_BERT_PREFIX.sub("", tokens[j])

                if post_token:
                    remaining = text_no_space[text_pos:]
                    anchor_pos = remaining.find(post_token)
                    if anchor_pos != -1:
                        unk_chars = remaining[:anchor_pos]
                    else:
                        unk_chars = remaining[:unk_count]
                else:
                    unk_chars = text_no_space[text_pos : text_pos + unk_count]

                for k in range(unk_count):
                    if k < len(unk_chars):
                        new_tokens.append(unk_chars[k])
                    else:
                        new_tokens.append("")
                text_pos += len(unk_chars)
                i = j
            else:
                new_tokens.append(token)
                token_clean = _RE_BERT_PREFIX.sub("", token)
                text_pos += len(token_clean)
                i += 1

        new_tokens = [t for t in new_tokens if t and t != "[UNK]"]
        return new_tokens

    def detokenize(self, inputs, join_symbol="", replace_spm_space=True):
        raise NotImplementedError
