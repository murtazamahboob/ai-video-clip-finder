"""Captions — karaoke word timings, SRT/ASS export, multi-language.

Blueprint §1.4: 3–5 words per line, max 2 lines, 10% safe margin top and
bottom. The Part 3 multi-language upgrade is a deterministic word-map
translation for the demo languages (Spanish, Chinese) — a full machine-
translation pass is an adapter, and the README says so; the map never
invents a translation for a word it does not know, it keeps the original.
"""

from __future__ import annotations

from dataclasses import dataclass

from .transcript import Word, _fmt_time

MAX_WORDS_PER_LINE = 5
MAX_LINES = 2

# Small deterministic maps for the bundled demo vocabulary only.
ES = {"welcome": "bienvenidos", "today": "hoy", "mistake": "error",
      "money": "dinero", "customers": "clientes", "product": "producto",
      "users": "usuarios", "thanks": "gracias", "ship": "lanza",
      "price": "precio", "revenue": "ingresos", "launch": "lanzamiento"}
ZH = {"welcome": "欢迎", "today": "今天", "mistake": "错误",
      "money": "钱", "customers": "客户", "product": "产品",
      "users": "用户", "thanks": "谢谢", "price": "价格",
      "revenue": "收入", "launch": "发布"}


@dataclass
class CaptionBlock:
    start: float
    end: float
    lines: list[str]
    words: list[Word]

    @property
    def text(self) -> str:
        return " ".join(self.lines)


def build_captions(words: list[Word], max_words_per_line: int = MAX_WORDS_PER_LINE) -> list[CaptionBlock]:
    """Chunk words into blocks of at most 2 lines × 5 words, on word timing."""
    blocks: list[CaptionBlock] = []
    i = 0
    while i < len(words):
        chunk = words[i:i + max_words_per_line * MAX_LINES]
        lines = [" ".join(w.text for w in chunk[j:j + max_words_per_line])
                 for j in range(0, len(chunk), max_words_per_line)]
        blocks.append(CaptionBlock(chunk[0].start, chunk[-1].end, lines, list(chunk)))
        i += len(chunk)
    return blocks


def translate_word(word: str, lang: str) -> str:
    table = {"es": ES, "zh": ZH}.get(lang, {})
    clean = word.strip(".,!?").lower()
    mapped = table.get(clean)
    if mapped is None:
        return word  # never invent a translation
    return mapped.capitalize() if word[:1].isupper() else mapped


def translate_block(block: CaptionBlock, lang: str) -> CaptionBlock:
    lines = [" ".join(translate_word(w, lang) for w in line.split())
             for line in block.lines]
    return CaptionBlock(block.start, block.end, lines, block.words)


def captions_to_srt(blocks: list[CaptionBlock]) -> str:
    cues = []
    for i, b in enumerate(blocks, 1):
        cues.append(f"{i}\n{_fmt_time(b.start)} --> {_fmt_time(b.end)}\n"
                    + "\n".join(b.lines) + "\n")
    return "\n".join(cues)


def captions_to_ass(blocks: list[CaptionBlock]) -> str:
    """Minimal ASS with a bold centred style and 10% safe margins."""
    def _t(t: float) -> str:
        ms = int(round(t * 1000))
        return f"{ms // 3600000}:{(ms // 60000) % 60:02d}:{(ms // 1000) % 60:02d}.{ms % 1000 // 10:02d}"
    header = ("[Script Info]\nTitle: clip captions\nScriptType: v4.00+\n"
              "PlayResX: 1080\nPlayResY: 1920\n\n[V4+ Styles]\n"
              "Format: Name, Fontname, Fontsize, PrimaryColour, Bold, Alignment, "
              "MarginL, MarginR, MarginV\n"
              "Style: Default,Arial,72,&H00FFFFFF,1,2,108,108,192\n\n[Events]\n"
              "Format: Layer, Start, End, Style, Text\n")
    events = "".join(
        f"Dialogue: 0,{_t(b.start)},{_t(b.end)},Default,"
        + "\\N".join(b.lines) + "\n" for b in blocks)
    return header + events
