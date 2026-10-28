# Speech foundation model sources

These files preserve the pretrained weights used by this project's validated tsumugi v2Pro Plus runtime. File sizes and SHA-256 hashes are recorded in `config/speech_model_manifest.json` at the repository root.

- `chinese-roberta-wwm-ext-large`: BERT configuration, tokenizer and masked-language-model weights distributed with [GPT-SoVITS pretrained models](https://huggingface.co/lj1995/GPT-SoVITS/tree/main/chinese-roberta-wwm-ext-large).
- `chinese-hubert-base`: HuBERT configuration, waveform preprocessing configuration and weights from the [same upstream distribution](https://huggingface.co/lj1995/GPT-SoVITS/tree/main/chinese-hubert-base).
- `sv/pretrained_eres2netv2w24s4ep4.ckpt`: the speaker embedding model used by [GPT-SoVITS v2Pro / v2ProPlus](https://github.com/RVC-Boss/GPT-SoVITS/blob/e7cd61ec3dcc9a34089f50ed169e3103a16ad675/GPT_SoVITS/sv.py).
- `fast_langdetect/lid.176.bin`: the [fastText language-identification model](https://fasttext.cc/docs/en/language-identification.html), distributed by its authors under CC-BY-SA 3.0. Attribution: Facebook AI Research / fastText; see the linked model page and [license](https://creativecommons.org/licenses/by-sa/3.0/).

The project-specific trained voice weights and reference recordings are stored separately under `VOICE_model/GPT_weights_v2ProPlus/tsumugi`. The original redundant reference-audio ZIP is not needed for inference.
