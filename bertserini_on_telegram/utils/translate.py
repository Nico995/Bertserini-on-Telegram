from typing import Tuple
import fasttext
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import os
import shutil
import requests
import json
import iso639

lang_map = {"ar": "ar_AR", "cs": "cs_CZ", "de": "de_DE", "en": "en_XX", "es": "es_XX", "et": "et_EE", "fi": "fi_FI", "fr": "fr_XX", "gu": "gu_IN", "hi": "hi_IN", "it": "it_IT", "ja": "ja_XX", "kk": "kk_KZ", "ko": "ko_KR", "lt": "lt_LT", "lv": "lv_LV", "my": "my_MM", "ne": "ne_NP", "nl": "nl_XX", "ro": "ro_RO", "ru": "ru_RU", "si": "si_LK", "tr": "tr_TR", "vi": "vi_VN", "zh": "zh_CN", "af": "af_ZA",
            "az": "az_AZ", "bn": "bn_IN", "fa": "fa_IR", "he": "he_IL", "hr": "hr_HR", "id": "id_ID", "ka": "ka_GE", "km": "km_KH", "mk": "mk_MK", "ml": "ml_IN", "mn": "mn_MN", "mr": "mr_IN", "pl": "pl_PL", "ps": "ps_AF", "pt": "pt_XX", "sv": "sv_SE", "sw": "sw_KE", "ta": "ta_IN", "te": "te_IN", "th": "th_TH", "tl": "tl_XX", "uk": "uk_UA", "ur": "ur_PK", "xh": "xh_ZA", "gl": "gl_ES", "sl": "sl_SI"}


class AutoTranslator:
    def __init__(self, pretrained_path: str = '/tmp', lang_map_path: str = './mbart50-m2m.json'):
        self.pretrained_path = pretrained_path
        self.model_url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
        model_path = self._download_pretrained()
        print(model_path)

        self.fasttext = fasttext.load_model(model_path)

        self.mbart = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        self.tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

        self.lang_map = lang_map

    def _download_pretrained(self):

        filename = 'fasttext_pretrained.bin'
        path = os.path.join(self.pretrained_path, filename)

        if not filename in os.listdir(self.pretrained_path):
            with requests.get(self.model_url, stream=True) as r:
                with open(path, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)

        return path

    def translate(self, text: str, src_lang: str = None, trg_lang: str = None) -> Tuple[str, Tuple[str, str]]:

        lang_iso639 = self.fasttext.predict(text)[0][0][-2:]

        if not src_lang:
            src_lang = self.lang_map.get(lang_iso639, None)

        if not trg_lang:
            trg_lang = self.lang_map.get(lang_iso639, None)

        if not src_lang:
            NotImplementedError(f'The language with iso369 code {lang_iso639} is not suported from Mbart50-m2m')
            return None, (iso639.to_name(lang_iso639), trg_lang)

        if 'en' in src_lang and 'en' in trg_lang:
            print('no need to translate')
            return text, (src_lang, trg_lang)

        self.tokenizer.src_lang = src_lang
        encoded_text = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.mbart.generate(
            **encoded_text,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[trg_lang]
        )

        translated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        return translated_text, (src_lang, trg_lang)
