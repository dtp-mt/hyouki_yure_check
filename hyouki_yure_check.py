import sys
import os
import re
import time
import unicodedata
import difflib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set
import fitz  # PyMuPDF
import pandas as pd
from PySide6.QtCore import (
    Qt, QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Signal, QObject, QThread, QRect,
    QTimer, QCollator, QPoint
)
from PySide6.QtGui import (
    QFont, QPalette, QColor, QPainter, QPen, QAction
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTableView, QTabWidget, QLineEdit, QDoubleSpinBox, QSpinBox,
    QGroupBox, QFormLayout, QMessageBox, QTextEdit, QProgressBar, QAbstractItemView,
    QListWidget, QCheckBox, QStyledItemDelegate, QStyle, QProxyStyle,
    QMenu, QDialog, QDialogButtonBox, QFrame
)
from PySide6.QtWidgets import QHeaderView
# 先頭の import 群のどこかに追加
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices

# === 帯付き Levenshtein & しきい値付き類似度判定（新規追加） ===
from typing import Optional

def levenshtein_band(a: str, b: str, k: int) -> Optional[int]:
    """
    帯幅 k の制限付き Levenshtein 距離。
    距離が k を超えることが確定したら None を返して早期終了。
    """
    la, lb = len(a), len(b)
    if a == b:
        return 0
    if abs(la - lb) > k:
        return None  # 距離 > k が確定
    if la < lb:
        a, b = b, a
        la, lb = lb, la
    INF = k + 1
    prev = [INF] * (lb + 1)
    for j in range(min(lb, k) + 1):
        prev[j] = j
    for i in range(1, la + 1):
        start = max(1, i - k)
        end   = min(lb, i + k)
        cur = [INF] * (lb + 1)
        if start == 1:
            cur[0] = i
        for j in range(start, end + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            cur[j] = min(prev[j] + 1,         # 削除
                         cur[j-1] + 1,         # 挿入
                         prev[j-1] + cost)     # 置換/一致
        # 早期打切り：帯内すべてが k 超なら終了
        if min(cur[start:end+1]) > k:
            return None
        prev = cur
    return prev[lb] if prev[lb] <= k else None

def sim_with_threshold(a: str, b: str, th: float) -> Optional[float]:
    """
    正規化類似度 >= th のときその値を返す。未満または判定不能は None。
    """
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    L = max(la, lb)
    k = int((1.0 - th) * L)
    d = levenshtein_band(a, b, k)
    if d is None:
        return None
    return 1.0 - d / L
def _ngram_set(s: str, n: int = 2):
    return {s[i:i+n] for i in range(len(s)-n+1)} if len(s) >= n else {s}

# 事前コンパイル（モジュール先頭に一度だけ）
BULLET_RE = re.compile(
    "^[\u2022\u2023\u25E6\u2043\u2219\u25AA\u25CF\u25CB\u25A0\u25A1\u30FB\ufeff\\s\u3000]+"
)

def strip_leading_control_chars(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = BULLET_RE.sub("", s)
    while s and (unicodedata.category(s[0]) in ("Cc", "Cf")):
        s = s[1:]
    return s

# ===== MeCab (fugashi) =====
try:
    from fugashi import Tagger
    HAS_MECAB = True
    MECAB_TAGGER: Optional["Tagger"] = None
except Exception:
    HAS_MECAB = False
    MECAB_TAGGER = None

# ------------------------------------------------------------
# 共通ユーティリティ
# ------------------------------------------------------------
PAIR_COLUMNS = ["a", "b", "a_count", "b_count", "reason", "score", "scope"]
def empty_pairs_df() -> pd.DataFrame:
    return pd.DataFrame(columns=PAIR_COLUMNS)

JP_BLOCK = r"[ぁ-んァ-ン一-龥々〆ヵヶA-Za-zＡ-Ｚａ-ｚ0-9０-９]+"
CONNECT = r"[ー\-－–—・/／\_＿]"  # 連結記号
TOKEN_RE = re.compile(rf"{JP_BLOCK}(?:{CONNECT}{JP_BLOCK})*")
CONNECT_CHARS = set("ー-－–—・/／_＿")

def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)

def hira_to_kata(s: str) -> str:
    out = []
    for ch in s:
        o = ord(ch)
        out.append(chr(o + 0x60) if 0x3041 <= o <= 0x3096 else ch)
    return "".join(out)

def normalize_kana(s: str, drop_choon: bool = True) -> str:
    s = nfkc(s)
    s = hira_to_kata(s)
    s = "".join(ch for ch in s if 0x30A0 <= ord(ch) <= 0x30FF)
    if drop_choon:
        s = s.replace("ー", "")
    return s

def is_single_kana_char(s: str) -> bool:
    if not isinstance(s, str):
        return False
    s_norm = nfkc(s)
    if len(s_norm) != 1:
        return False
    o = ord(s_norm)
    return (0x3040 <= o <= 0x309F) or (0x30A0 <= o <= 0x30FF)

_ASCII = r"[A-Za-z0-9]"
def soft_join_lines_lang_aware(s: str) -> str:
    s = s.replace("\u00A0", " ").replace("\u3000", " ").replace("\r", "")
    s = re.sub(rf"({_ASCII})-\n((({_ASCII})))", r"\1\2", s)
    s = re.sub(rf"({_ASCII})\n((({_ASCII})))", r"\1 \2", s)
    s = s.replace("\n", "")
    s = re.sub(r"[ ]{2,}", " ", s)
    return s

@dataclass
class PageData:
    pdf: str
    page: int
    text_join: str

def extract_pages(pdf_path: str) -> List[PageData]:
    pages = []
    doc = fitz.open(pdf_path)
    try:
        for i in range(len(doc)):
            text = doc[i].get_text("text")
            pages.append(PageData(os.path.basename(pdf_path), i + 1, soft_join_lines_lang_aware(text)))
    finally:
        doc.close()
    return pages

# ------------------------------------------------------------
# MeCab helpers
# ------------------------------------------------------------
def ensure_mecab() -> Tuple[bool, Optional[str]]:
    global MECAB_TAGGER
    if not HAS_MECAB:
        return False, 'fugashi/unidic-lite が未インストールです。 pip install "fugashi[unidic-lite]"'
    if MECAB_TAGGER is None:
        try:
            MECAB_TAGGER = Tagger()
        except Exception as e:
            return False, f"MeCab 初期化失敗: {e}"
    return True, None

def feat_get(w, keys: List[str]) -> Optional[str]:
    f = getattr(w, "feature", None)
    if f is None:
        return None
    for k in keys:
        try:
            v = getattr(f, k)
            if v:
                return str(v)
        except Exception:
            pass
    return None

def tokenize_mecab(text: str, tagger: "Tagger"):
    out = []
    for w in tagger(text):
        surf = w.surface or ""
        pos1 = feat_get(w, ["pos1"]) or ""
        lemma = feat_get(w, ["lemma", "baseform"]) or surf
        reading = feat_get(w, ["reading", "pron", "pronBase", "kana", "kanaBase", "yomi"]) or ""
        ct = feat_get(w, ["cType", "conjType"]) or ""
        cf = feat_get(w, ["cForm", "conjForm"]) or ""
        if surf:
            out.append((surf, pos1, lemma, reading, ct, cf))
    return out

# ==== フレーズの読み正規化（複合語/文節用） ====
def phrase_reading_norm(s: str, tagger: "Tagger") -> str:
    """
    文字列 s を MeCab で再トークン化し、各語の読みを連結。
    その後 normalize_kana(drop_choon=True) で
    ・ひらがな→カタカナ
    ・カタカナ以外を除去
    ・長音「ー」を除去
    して返します（= 読みの“骨格”同士で比較できる）。
    """
    if not isinstance(s, str) or not s:
        return ""

    # 既存の tokenize_mecab を活用
    toks = tokenize_mecab(s, tagger)
    parts = []
    for surf, pos1, lemma, reading, ct, cf in toks:
        r = reading or ""
        if not r:
            # 読みが取れない場合は表層を NFKC→カタカナ化して代用
            r = hira_to_kata(nfkc(surf))
        parts.append(r)

    joined = "".join(parts)
    # ★ カタカナ以外は落ち、長音「ー」も削除される
    return normalize_kana(joined, drop_choon=True)

# ===== 読み正規化のキャッシュ（追加ブロック：貼り付け） =====
# phrase_reading_norm(s, tagger) を毎回呼ぶのは高コストなので、
# "語 → 読み(骨格)" をプロセス内でキャッシュします。
READ_NORM_CACHE: Dict[str, str] = {}

def phrase_reading_norm_cached(s: str) -> str:
    """語 s の読み(骨格)をキャッシュして返す。MeCab未使用時はフォールバック。"""
    if not isinstance(s, str) or not s:
        return ""
    v = READ_NORM_CACHE.get(s)
    if v is not None:
        return v
    ok, _ = ensure_mecab()
    if ok and MECAB_TAGGER is not None:
        v = phrase_reading_norm(s, MECAB_TAGGER)
    else:
        # フォールバック：NFKC→カタカナ→カタカナ以外除去→長音除去
        k = hira_to_kata(nfkc(s))
        v = normalize_kana(k, drop_choon=True)
    READ_NORM_CACHE[s] = v
    return v
# ===== 読みキャッシュ ここまで =====

# ------------------------------------------------------------
# Token抽出（正規表現ベース, MeCabなしフォールバック用）
# ------------------------------------------------------------
def extract_candidates_regex(pages: List[PageData], min_len=1, max_len=120, min_count=1):
    cnt = Counter()
    for p in pages:
        for m in TOKEN_RE.finditer(p.text_join):
            w = m.group(0)
            if min_len <= len(w) <= max_len:
                cnt[w] += 1
    arr = [(w, c) for w, c in cnt.items() if c >= min_count]
    arr.sort(key=lambda x: (-x[1], x[0]))
    return arr

# ------------------------------------------------------------
# Compound（連結記号でのみ結合）
# ------------------------------------------------------------
def mecab_compound_tokens_alljoin(text: str) -> List[str]:
    ok, msg = ensure_mecab()
    if not ok:
        raise RuntimeError(msg)
    toks = tokenize_mecab(text, MECAB_TAGGER)
    out: List[str] = []
    parts: List[str] = []
    has_conn = False
    prev_type: Optional[str] = None

    def flush():
        nonlocal parts, has_conn, prev_type
        if parts and has_conn:
            token = "".join(parts)
            if 1 <= len(token) <= 120:
                out.append(token)
        parts = []
        has_conn = False
        prev_type = None

    for surf, pos1, lemma, reading, ct, cf in toks:
        is_word = pos1.startswith(("名詞", "動詞", "形容詞", "助動詞"))
        is_conn = surf and all(ch in CONNECT_CHARS for ch in surf)
        if is_conn:
            if parts and prev_type == 'word':
                parts.append(surf)
                has_conn = True
                prev_type = 'conn'
                continue
        elif is_word:
            if not parts:
                parts = [surf]
                prev_type = 'word'
            else:
                if prev_type == 'conn':
                    parts.append(surf)
                    prev_type = 'word'
                else:
                    flush()
                    parts = [surf]
                    prev_type = 'word'
        else:
            flush()
    flush()
    return out

def extract_candidates_compound_alljoin(
    pages: List[PageData], min_len=1, max_len=120, min_count=1, top_k=0, use_mecab=True
):
    if use_mecab and HAS_MECAB:
        cnt = Counter()
        for p in pages:
            for w in mecab_compound_tokens_alljoin(p.text_join):
                w = strip_leading_control_chars(w)  # ★追加
                if not w:
                    continue
                if min_len <= len(w) <= max_len:
                    cnt[w] += 1
        arr = [(w, c) for w, c in cnt.items() if c >= min_count]
        arr.sort(key=lambda x: (-x[1], x[0]))
        return arr if (not top_k or top_k <= 0) else arr[:top_k]
    else:
        return extract_candidates_regex(pages, min_len=min_len, max_len=max_len, min_count=min_count)

# ------------------------------------------------------------
# 文節抽出（簡易ルール）
# ------------------------------------------------------------
PUNCT_SET = set('。、．，!.！？？」』〕）】]〉》"\\\'\')')
def mecab_bunsetsu_chunks(text: str) -> List[str]:
    ok, msg = ensure_mecab()
    if not ok:
        raise RuntimeError(msg)
    toks = tokenize_mecab(text, MECAB_TAGGER)
    chunks: List[str] = []
    cur: List[str] = []

    def flush():
        nonlocal cur
        if cur:
            s = "".join(cur).strip()
            if 1 <= len(s) <= 120:
                chunks.append(s)
        cur = []

    for surf, pos1, lemma, reading, ct, cf in toks:
        if surf in PUNCT_SET or pos1.startswith("記号"):
            flush()
            continue
        cur.append(surf)
        if pos1 in ("助詞", "助動詞"):
            flush()
    flush()
    return chunks

def extract_candidates_bunsetsu(
    pages: List[PageData], min_len=1, max_len=120, min_count=1, top_k=0
) -> List[Tuple[str, int]]:
    cnt = Counter()
    for p in pages:
        for ch in mecab_bunsetsu_chunks(p.text_join):
            ch = strip_leading_control_chars(ch)  # ★追加
            if not ch:
                continue
            if min_len <= len(ch) <= max_len:
                cnt[ch] += 1
    arr = [(w, c) for w, c in cnt.items() if c >= min_count]
    arr.sort(key=lambda x: (-x[1], x[0]))
    return arr if (not top_k or top_k <= 0) else arr[:top_k]

# ------------------------------------------------------------
# Similarity
# ------------------------------------------------------------
def levenshtein_sim(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    if a == b:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    if la < lb:
        a, b = b, a
        la, lb = lb, la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cb = b[j - 1]
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return 1.0 - prev[lb] / max(la, lb)

# ===== Edge-only difference (first/last 1-char) =====
def classify_edge_diff(a: str, b: str) -> str:
    """
    端（前後）1文字だけの差で説明できるかを判定し、
    種別を返す（該当なしは "" を返す）。
      "prefix_add" / "prefix_change" / "suffix_add" / "suffix_change" / ""
    """
    if a is None or b is None:
        return ""
    a = str(a); b = str(b)
    if a == b:
        return ""
    la, lb = len(a), len(b)

    # 前: 有無（片方が1文字長くて、2文字目以降が完全一致）
    if la == lb + 1 and a[1:] == b:
        return "prefix_add"
    if lb == la + 1 and b[1:] == a:
        return "prefix_add"

    # 前: 違い（同長で、2文字目以降が一致）
    if la == lb and la >= 1 and a[1:] == b[1:]:
        return "prefix_change"

    # 後: 有無（片方が1文字長くて、末尾1文字を除けば一致）
    if la == lb + 1 and a[:-1] == b:
        return "suffix_add"
    if lb == la + 1 and b[:-1] == a:
        return "suffix_add"

    # 後: 違い（同長で、末尾以外が一致）
    if la == lb and la >= 1 and a[:-1] == b[:-1]:
        return "suffix_change"

    return ""

def classify_edge_diff_jp(a: str, b: str) -> str:
    """端差の日本語ラベル（表示・フィルタ用）。"""
    m = classify_edge_diff(a, b)
    return {
        "prefix_add":   "前1字有無",
        "prefix_change":"前1字違い",
        "suffix_add":   "後1字有無",
        "suffix_change":"後1字違い",
    }.get(m, "")

# ==== ベクトル化ユーティリティ（新規） =====================================
_DIGIT_RE = re.compile(r"\d", flags=re.UNICODE)

def _edge_labels_vectorized(df: pd.DataFrame) -> pd.Series:
    a = df["a"].astype("string")
    b = df["b"].astype("string")
    la = a.str.len()
    lb = b.str.len()
    res = pd.Series("", index=df.index, dtype="string")

    # 前1字有無
    m = (la == lb + 1) & (a.str.slice(1) == b)
    res = res.mask(m, "前1字有無")
    m = (lb == la + 1) & (b.str.slice(1) == a) & (res == "")
    res = res.mask(m, "前1字有無")

    # 前1字違い
    m = (la == lb) & (la >= 1) & (a.str.slice(1) == b.str.slice(1)) & (res == "")
    res = res.mask(m, "前1字違い")

    # 後1字有無
    m = (la == lb + 1) & (a.str.slice(0, -1) == b) & (res == "")
    res = res.mask(m, "後1字有無")
    m = (lb == la + 1) & (b.str.slice(0, -1) == a) & (res == "")
    res = res.mask(m, "後1字有無")

    # 後1字違い
    m = (la == lb) & (la >= 1) & (a.str.slice(0, -1) == b.str.slice(0, -1)) & (res == "")
    res = res.mask(m, "後1字違い")
    return res

def _numeric_only_label_vectorized(df: pd.DataFrame) -> pd.Series:
    a = df["a"].astype("string")
    b = df["b"].astype("string")
    diff_orig = (a != b)
    has_digit = a.str.contains(_DIGIT_RE) | b.str.contains(_DIGIT_RE)
    a_wo = a.str.replace(_DIGIT_RE, "", regex=True)
    b_wo = b.str.replace(_DIGIT_RE, "", regex=True)
    mask = diff_orig & has_digit & (a_wo == b_wo)
    out = pd.Series("", index=df.index, dtype="string")
    out[mask] = "数字以外一致"
    return out
# ============================================================================

# ==== 数字以外一致（数字を全削除して比較） ====
def strip_digits_norm(s: str) -> str:
    """
    文字列 s から「十進数字（Unicode Nd）」だけを除去した文字列を返す。
    ※ ここでは NFKC 正規化は行わない（全角/半角など非数字差を温存する）
    """
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    return _DIGIT_RE.sub("", s)

def is_numeric_only_diff(a: str, b: str) -> bool:
    """
    a, b が“数字だけ異なる”なら True。
    条件:
      - a != b（元が同一なら False）
      - 片方以上に十進数字が含まれる
      - 十進数字をすべて除去した結果が一致する
    例:
      True:  "3kg" vs "３kg", "年1回" vs "年2回", "ver1.0" vs "ver１.0"
      False: "kg" vs "ｋg"（数字なし・幅違い）、"(" vs "（"（数字なし・記号幅違い）
    """
    if a is None or b is None:
        return False
    a_s, b_s = str(a), str(b)
    if a_s == b_s:
        return False
    # 少なくともどちらかに「数字」が含まれていること
    if not (_DIGIT_RE.search(a_s) or _DIGIT_RE.search(b_s)):
        return False
    # 数字を除去した残りが完全一致なら「数字だけ違う」
    return strip_digits_norm(a_s) == strip_digits_norm(b_s)

def numeric_only_label(a: str, b: str) -> str:
    """表示用ラベル（該当時のみ '数字以外一致' を返す）"""
    return "数字以外一致" if is_numeric_only_diff(a, b) else ""

# ------------------------------------------------------------
# Lexical（細粒度）語彙とペア
# ------------------------------------------------------------
def collect_lexicon_general(pages: List[PageData], top_k=4000, min_count=1) -> pd.DataFrame:
    ok, msg = ensure_mecab()
    if not ok:
        raise RuntimeError(msg)
    counts = Counter()
    readings_map: Dict[str, Counter] = defaultdict(Counter)
    lemma_map: Dict[str, Counter] = defaultdict(Counter)
    lemma_read_map: Dict[str, Counter] = defaultdict(Counter)
    pos_map: Dict[str, Counter] = defaultdict(Counter)

    for p in pages:
        for surf, pos1, lemma, reading, ct, cf in tokenize_mecab(p.text_join, MECAB_TAGGER):
            if not pos1.startswith(("名詞", "動詞", "形容詞", "助動詞")):
                continue
            counts[surf] += 1
            r = normalize_kana(reading or "", drop_choon=True)
            if r:
                readings_map[surf][r] += 1
            if lemma:
                lemma_read_map[lemma][r] += 1
            if lemma:
                lemma_map[surf][lemma] += 1
            pos_map[surf][pos1] += 1

    def top1(cnt: Counter) -> str:
        return cnt.most_common(1)[0][0] if cnt else ""

    items = [(s, c) for s, c in counts.items() if c >= min_count]
    items.sort(key=lambda x: (-x[1], x[0]))
    if top_k and top_k > 0:
        items = items[:top_k]

    lemma2readnorm = {lem: top1(lemma_read_map[lem]) for lem in lemma_read_map}
    rows = []
    for s, c in items:
        lem = top1(lemma_map[s])
        rows.append({
            "surface": s,
            "count": int(c),
            "reading": top1(readings_map[s]),
            "lemma": lem,
            "lemma_read": lemma2readnorm.get(lem, ""),
            "pos1": top1(pos_map[s]),
        })
    return pd.DataFrame(rows)

def build_synonym_pairs_general(
    df_lex: pd.DataFrame,
    read_sim_th=0.90,
    char_sim_th=0.85,
    scope="語彙",
    progress_cb=None
) -> pd.DataFrame:
    if df_lex.empty:
        return empty_pairs_df()

    words   = df_lex["surface"].tolist()
    counts  = df_lex["count"].tolist()
    lemmas  = df_lex["lemma"].fillna("").tolist()
    lemma_reads = df_lex["lemma_read"].fillna("").tolist()
    readings = df_lex["reading"].fillna("").tolist()

    n = len(words)
    norm_surface = [nfkc(w) for w in words]
    norm_read    = [normalize_kana(r or "", True) for r in readings]

    # 追加：n-gram セット前計算（文字類似の粗フィルタ）
    ng_char = [_ngram_set(s) for s in norm_surface]

    rows = []
    step = max(1, n // 100)
    for i in range(n):
        for j in range(i + 1, n):
            a, b   = words[i], words[j]
            ca, cb = int(counts[i]), int(counts[j])
            la, lb = lemmas[i], lemmas[j]
            ra, rb = norm_read[i], norm_read[j]
            sa, sb = norm_surface[i], norm_surface[j]

            reason = None
            score  = 0.0

            # 1) 基本形（lemma）一致を最優先
            if la and lb and la == lb and a != b:
                # lemmaそのものが片方の表層に一致なら "lemma"、それ以外は活用差とみなす
                if (a == la) or (b == la):
                    reason, score = "lemma", 1.0
                else:
                    reason, score = "inflect", 0.95
            else:
                # 2) 読み類似：長さ差ゲート + 帯付きLevenshtein
                if ra and rb:
                    sim_r = sim_with_threshold(ra, rb, read_sim_th)
                    if sim_r is not None:
                        reason, score = "reading", round(sim_r, 3)
                # 3) 文字類似：長さ差ゲート + n-gram 粗フィルタ + 帯付きLevenshtein
                if reason is None:
                    # 長さ差ゲート
                    L  = max(len(sa), len(sb))
                    kC = int((1.0 - char_sim_th) * L)
                    if abs(len(sa) - len(sb)) <= kC:
                        # n-gram 粗フィルタ
                        inter = len(ng_char[i] & ng_char[j])
                        union = len(ng_char[i] | ng_char[j])
                        if union > 0 and (inter / union) >= 0.30:
                            sim_c = sim_with_threshold(sa, sb, char_sim_th)
                            if sim_c is not None:
                                reason, score = "char", round(sim_c, 3)

            if reason:
                rows.append({
                    "a": a, "b": b, "a_count": ca, "b_count": cb,
                    "reason": reason, "score": score, "scope": scope
                })

        if progress_cb and (i % step == 0 or i == n - 1):
            progress_cb(i + 1, n)

    if not rows:
        return empty_pairs_df()

    return pd.DataFrame(rows).sort_values(
        ["reason", "score", "a_count", "b_count"],
        ascending=[True, False, False, False]
    )

# ===== [6-A] 2-gram 逆引きインデックス（追加） ==========================
from collections import defaultdict

def _build_bigram_inverted_index(ngram_list):
    """ngram_list: List[frozenset[str]] を受け取り、2-gram -> [idx,...] の dict を返す"""
    inv = defaultdict(list)
    for idx, grams in enumerate(ngram_list):
        for g in grams:
            inv[g].append(idx)
    # 必要なら重複排除（速度とメモリのバランスで適宜）
    for g, lst in inv.items():
        inv[g] = sorted(set(lst))
    return inv

def _candidate_ids_via_index(i: int, inv: dict, ngram_list, n: int):
    """
    文脈 i の候補 j を逆引きインデックスで取得（i < j のみ）。
    """
    Ai = ngram_list[i]
    pool = set()
    for g in Ai:
        pool.update(inv.get(g, ()))
    return [j for j in pool if j > i and j < n]
# =======================================================================

# ===== 並列版 build_synonym_pairs_char_only（関数ごと差し替え） =====
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import math

# ===== [6-B2] 並列ワーカー差し替え：逆引きインデックスで候補前絞り =====
def _pairs_chunk_worker(args):
    """
    子プロセス側：i 範囲を担当し、逆引きインデックスで候補を前絞りしてから判定。
    """
    (i_start, i_end, words, counts, norm_char, read_norm,
     ngram_list, char_sim_th, read_sim_th, scope) = args

    rows = []
    n = len(words)
    JACC_TH = 0.30

    # ここで一度だけ逆引きインデックスを構築（各ワーカー内で）
    from collections import defaultdict
    inv = defaultdict(list)
    for idx, grams in enumerate(ngram_list):
        for g in grams:
            inv[g].append(idx)
    for g, lst in inv.items():
        inv[g] = sorted(set(lst))

    for i in range(i_start, i_end):
        a = words[i]
        sa = norm_char[i]
        Ai = ngram_list[i]

        # --- 候補を前絞り ---
        pool = set()
        for g in Ai:
            pool.update(inv.get(g, ()))
        cand_js = [j for j in pool if j > i and j < n]

        for j in cand_js:
            b = words[j]

            # 読み類似（優先）
            if read_sim_th is not None and read_norm[i] and read_norm[j]:
                sim_r = sim_with_threshold(read_norm[i], read_norm[j], read_sim_th)
                if sim_r is not None:
                    rows.append({
                        "a": a, "b": b,
                        "a_count": counts[a], "b_count": counts[b],
                        "reason": "reading",
                        "score": round(sim_r, 3),
                        "scope": scope
                    })
                    continue

            # 文字類似：粗→本審査
            Aj = ngram_list[j]
            inter = len(Ai & Aj)
            union = len(Ai | Aj)
            if union == 0 or (inter / union) < JACC_TH:
                continue

            sb = norm_char[j]
            sim_c = sim_with_threshold(sa, sb, char_sim_th)
            if sim_c is not None:
                rows.append({
                    "a": a, "b": b,
                    "a_count": counts[a], "b_count": counts[b],
                    "reason": "char",
                    "score": round(sim_c, 3),
                    "scope": scope
                })
    return rows
# =======================================================================

def build_synonym_pairs_char_only(
    tokens: List[Tuple[str,int]],
    char_sim_th=0.90,
    read_sim_th=None,
    top_k=4000,
    scope="複合語",
    progress_cb=None,
    use_processes: bool = True,
    max_workers: Optional[int] = None
) -> pd.DataFrame:
    """
    並列対応・高速版：
      - 読み類似は（指定時）帯付きLevenshtein
      - 文字類似は n-gram Jaccard の粗フィルタ → 帯付きLevenshtein
      - i の範囲をチャンク分割し、ProcessPoolExecutor で分散
    """
    if not tokens:
        return empty_pairs_df()

    tokens = sorted(tokens, key=lambda x: (-x[1], x[0]))
    if top_k and top_k > 0:
        tokens = tokens[:top_k]

    words = [w for w, _ in tokens]
    counts = {w: int(c) for w, c in tokens}

    # 正規化文字列
    norm_char = [nfkc(w) for w in words]

    # 読み正規化（キャッシュ利用）
    use_reading = (read_sim_th is not None) and HAS_MECAB
    read_norm = [""] * len(words)
    if use_reading:
        ok, _ = ensure_mecab()
        if ok:
            read_norm = [phrase_reading_norm_cached(w) for w in words]

    # 2-gram の集合（frozenset にして子プロセスへシリアライズ負荷を軽減）
    ngram_list = [frozenset(_ngram_set(s)) for s in norm_char]

    n = len(words)
    if n < 2:
        return empty_pairs_df()

    # 逐次版（use_processes=False またはワーカ=1 の場合）
    if not use_processes:
        rows = _pairs_chunk_worker((0, n-1, words, counts, norm_char, read_norm,
                                    ngram_list, char_sim_th, read_sim_th, scope))
        if not rows:
            return empty_pairs_df()
        return pd.DataFrame(rows).sort_values(
            ["score", "a_count", "b_count"], ascending=[False, False, False]
        )

    # 並列版
    workers = max(1, (max_workers or os.cpu_count() or 1))
    if workers == 1 or n < 1024:
        # 要素数が少ない時はオーバーヘッド回避のため逐次
        rows = _pairs_chunk_worker((0, n-1, words, counts, norm_char, read_norm,
                                    ngram_list, char_sim_th, read_sim_th, scope))
        if not rows:
            return empty_pairs_df()
        return pd.DataFrame(rows).sort_values(
            ["score", "a_count", "b_count"], ascending=[False, False, False]
        )

    # i の分割（i は 0..n-2 を走査）
    # チャンク数は workers * 4 目安（小さすぎると負荷分散が偏る）
    total_i = n - 1
    chunks = max(workers * 4, 1)
    step = math.ceil(total_i / chunks)
    tasks = []
    futures = []
    args_common = (words, counts, norm_char, read_norm, ngram_list, char_sim_th, read_sim_th, scope)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for s in range(0, total_i, step):
            e = min(total_i, s + step)
            # 子プロセスにはタプルで引数一括渡し
            args = (s, e, *args_common)
            futures.append(ex.submit(_pairs_chunk_worker, args))
            tasks.append((s, e))

        rows_all = []
        # 進捗：チャンクの「終端 i」を元の n に写像して報告（おおよそでOK）
        for (s, e), fut in zip(tasks, as_completed(futures)):
            try:
                rows_all.extend(fut.result())
            except Exception:
                # サブタスク失敗は無視（ログ等が必要ならここで print など）
                pass
            if progress_cb:
                # i は 0..(n-1) を想定していたため、e をそのまま使う
                progress_cb(min(e, n-1), n)

    if not rows_all:
        return empty_pairs_df()

    return pd.DataFrame(rows_all).sort_values(
        ["score", "a_count", "b_count"], ascending=[False, False, False]
    )
# ===== 並列版 ここまで =====

# ------------------------------------------------------------
# 統合（★「読み類似」を最優先）
# ------------------------------------------------------------
REASON_PRIORITY = {"reading": 6.0, "lemma": 5.0, "inflect": 4.5, "lemma_read": 4.0, "char": 2.0}

def unify_pairs(*dfs: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for df in dfs:
        if df is None or df.empty:
            continue
        tmp = df.copy()
        for col in PAIR_COLUMNS:
            if col not in tmp.columns:
                tmp[col] = pd.NA
        frames.append(tmp[PAIR_COLUMNS].copy())

    if not frames:
        return empty_pairs_df()

    df = pd.concat(frames, ignore_index=True)

    def norm_row(r):
        a, b = r["a"], r["b"]
        if pd.isna(a) or pd.isna(b):
            return a, b
        return (a, b) if str(a) <= str(b) else (b, a)

    df[["a_n", "b_n"]] = df.apply(norm_row, axis=1, result_type="expand")

    def best_row(g):
        g = g.copy()
        g["prio"] = g["reason"].map(REASON_PRIORITY).fillna(1.0)
        g["sum_count"] = pd.to_numeric(g["a_count"], errors="coerce").fillna(0).astype(int) + \
                         pd.to_numeric(g["b_count"], errors="coerce").fillna(0).astype(int)
        g["score"] = pd.to_numeric(g["score"], errors="coerce").fillna(0.0)
        g = g.sort_values(["prio", "score", "sum_count"], ascending=[False, False, False])
        top = g.iloc[0].drop(labels=["prio", "sum_count"], errors="ignore")
        return top

    # --- applyを使わない版（高速＆FutureWarning回避） ---
    tmp = df.copy()

    # best_row() と同じ並べ替えキーを前計算
    tmp["prio"] = tmp["reason"].map(REASON_PRIORITY).fillna(1.0)
    tmp["sum_count"] = (
        pd.to_numeric(tmp["a_count"], errors="coerce").fillna(0).astype(int) +
        pd.to_numeric(tmp["b_count"], errors="coerce").fillna(0).astype(int)
    )
    tmp["score"] = pd.to_numeric(tmp["score"], errors="coerce").fillna(0.0)

    # グループ (a_n, b_n) ごとに「優先度・score・sum_count」の降順で並べ、先頭だけ残す
    tmp = tmp.sort_values(
        ["a_n", "b_n", "prio", "score", "sum_count"],
        ascending=[True, True, False, False, False]
    )

    best = (
        tmp.drop_duplicates(["a_n", "b_n"], keep="first")
        .drop(columns=["a_n", "b_n", "prio", "sum_count"], errors="ignore")
        .reset_index(drop=True)
    )

    # （元実装どおりの最終並べ替え）
    best = best.sort_values(["reason", "score", "a", "b"], ascending=[True, False, True, True])

    best = best.drop(columns=["a_n", "b_n", "prio", "sum_count"], errors="ignore")
    return best.sort_values(["reason", "score", "a", "b"], ascending=[True, False, True, True])

# ------------------------------------------------------------
# 変種のグループ化（lemma＋連結成分）→ gid 付与に利用
# ------------------------------------------------------------
def build_variation_groups(df_unified: pd.DataFrame, df_lex: Optional[pd.DataFrame] = None):
    from collections import defaultdict, Counter, deque
    g = defaultdict(set)

    if df_unified is not None and not df_unified.empty:
        for a, b in df_unified[['a','b']].itertuples(index=False):
            if isinstance(a, str) and isinstance(b, str) and a and b:
                g[a].add(b); g[b].add(a)

    surf2lemma = {}
    if df_lex is not None and not df_lex.empty:
        for s, l in df_lex[['surface','lemma']].itertuples(index=False):
            if isinstance(s, str) and s:
                surf2lemma[s] = l if isinstance(l, str) and l else s

    lemma_groups = defaultdict(list)
    for s, l in surf2lemma.items():
        lemma_groups[l].append(s)
    for mems in lemma_groups.values():
        for i in range(len(mems)):
            for j in range(i+1, len(mems)):
                a, b = mems[i], mems[j]
                g[a].add(b); g[b].add(a)

    visited = set()
    groups = []
    for node in list(g.keys()):
        if node in visited:
            continue
        q = deque([node]); visited.add(node)
        comp = []
        while q:
            u = q.popleft(); comp.append(u)
            for v in g[u]:
                if v not in visited:
                    visited.add(v); q.append(v)
        groups.append(sorted(set(comp)))

    df_groups_rows = []
    surf2gid: Dict[str, int] = {}
    gid2members: Dict[int, List[str]] = {}

    def best_label(members):
        if surf2lemma:
            cc = Counter(surf2lemma.get(s, s) for s in members)
            lab, cnt = cc.most_common(1)[0]
            if lab:
                return lab
        return sorted(members, key=lambda x: (len(x), x))[0]

    for gid, mems in enumerate(sorted(groups, key=lambda ms: (-len(ms), ms[0] if ms else "")), start=1):
        label = best_label(mems) if mems else ""
        for s in mems:
            surf2gid[s] = gid
        gid2members[gid] = mems
        df_groups_rows.append({'gid': gid, 'label': label, 'size': len(mems), 'members': ", ".join(mems)})

    df_groups = pd.DataFrame(df_groups_rows) if df_groups_rows else pd.DataFrame(
        columns=['gid','label','size','members']
    )
    return df_groups, surf2gid, gid2members

# ------------------------------------------------------------
# 表示用：英→日本語（列名）
# ------------------------------------------------------------
REASON_JA_MAP = {
    "lemma": "基本形一致",
    "lemma_read": "基本形の読み一致",
    "reading": "読み類似",
    "char": "文字類似",
    "inflect": "活用違い",
}
def apply_reason_ja(df: pd.DataFrame) -> pd.DataFrame:
    """reason列（英）→ 一致要因（日本語）列に置換"""
    if df is None or df.empty or "reason" not in df.columns:
        return df
    df = df.copy()
    df["一致要因"] = df["reason"].map(REASON_JA_MAP).fillna(df["reason"])
    df = df.drop(columns=["reason"])
    return df

# ------------------------------------------------------------
# Diff（インライン差分のみ）
# ------------------------------------------------------------
import html

def _esc_html(s: str) -> str:
    return "" if s is None else html.escape(str(s), quote=False)

def html_inline_diff(a: str, b: str) -> str:
    sm = difflib.SequenceMatcher(a=a or "", b=b or "")
    html = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'equal':
            html.append(_esc_html((a or "")[i1:i2]))
        elif op == 'delete':
            seg = _esc_html((a or "")[i1:i2])
            if seg:
                html.append(f"<s style='color:#d9480f'>{seg}</s>")
        elif op == 'insert':
            seg = _esc_html((b or "")[j1:j2])
            if seg:
                html.append(f"<span style='color:#1c7ed6;text-decoration:underline'>{seg}</span>")
        elif op == 'replace':
            del_seg = _esc_html((a or "")[i1:i2])
            ins_seg = _esc_html((b or "")[j1:j2])
            if del_seg:
                html.append(f"<s style='color:#d9480f'>{del_seg}</s>")
            if ins_seg:
                html.append(f"<span style='color:#1c7ed6;text-decoration:underline'>{ins_seg}</span>")
    return "".join(html)

def html_quick_ab_diff(a: str, b: str) -> str:
    import difflib
    def token_diff_for_A(a_line: str, b_line: str) -> str:
        sm2 = difflib.SequenceMatcher(a=a_line or "", b=b_line or "")
        parts = []
        for op, i1, i2, j1, j2 in sm2.get_opcodes():
            a_seg = (a_line or "")[i1:i2]
            b_seg = (b_line or "")[j1:j2]
            if op == "equal":
                parts.append(_esc_html(a_seg))
            elif op in ("delete", "replace"):
                if a_seg:
                    parts.append(f"<span class='tok-del'>{_esc_html(a_seg)}</span>")
        return "".join(parts)

    def token_diff_for_B(a_line: str, b_line: str) -> str:
        sm2 = difflib.SequenceMatcher(a=a_line or "", b=b_line or "")
        parts = []
        for op, i1, i2, j1, j2 in sm2.get_opcodes():
            a_seg = (a_line or "")[i1:i2]
            b_seg = (b_line or "")[j1:j2]
            if op == "equal":
                parts.append(_esc_html(b_seg))
            elif op in ("insert", "replace"):
                if b_seg:
                    parts.append(f"<span class='tok-ins'>{_esc_html(b_seg)}</span>")
        return "".join(parts)

    a = "" if a is None else str(a)
    b = "" if b is None else str(b)
    sm = difflib.SequenceMatcher(a=a.splitlines(), b=b.splitlines())
    ops = sm.get_opcodes()

    css = """
    <style>
    html, body, div, p { margin:0; padding:0; }
    body {
      font-family:'Yu Gothic UI','Noto Sans JP',sans-serif;
      color:#111; font-size:16px; line-height:1.80; letter-spacing:0.2px;
    }
    .page { padding-top:6px; padding-bottom:0; }
    .sec { margin:0 0 6px 0; }
    .sec:last-child { margin-bottom:0; }
    .sec-b { margin-top:12px; }
    .sec-head { font-weight:600; margin:0; line-height:0; }
    .sec-a .sec-head { color:#d9480f; }
    .sec-b .sec-head { color:#1c7ed6; }
    .diff { line-height:1.00; }
    table, .wrap { border-collapse:collapse; border-spacing:0; margin:0; }
    .wrap { width:100%; }
    .rail { width:18px; }
    .rail-a { background:#fa5252; }
    .rail-b { background:#4dabf7; }
    .body { padding:4px 10px 6px 10px; }
    .line { white-space:pre-wrap; padding:2px 4px; }
    .eq {}
    .add { background:#e7f5ff; }
    .del { background:#fff0f0; }
    .rep { background:#fff7e6; }
    .tok-ins { color:#1c7ed6; background:#d0ebff; border-radius:2px; }
    .tok-del { color:#c92a2a; background:#ffe3e3; border-radius:2px; }
    </style>
    """

    def esc(s: str) -> str:
        return _esc_html(s or "")

    a_lines = a.splitlines()
    b_lines = b.splitlines()

    a_rows = [
        "<div class='page sec sec-a'>",
        "<div class='diff'>",
        "<table class='wrap' cellspacing='0' cellpadding='0'><tr>",
        "<td class='rail rail-a'></td>",
        "<td class='body'>"
    ]
    b_rows = [
        "<div class='page sec sec-b'>",
        "<div class='diff'>",
        "<table class='wrap' cellspacing='0' cellpadding='0'><tr>",
        "<td class='rail rail-b'></td>",
        "<td class='body'>"
    ]

    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            for k in range(i2 - i1):
                text = esc(a_lines[i1 + k])
                a_rows.append(f"<div class='line eq'>{text}</div>")
                b_rows.append(f"<div class='line eq'>{text}</div>")
        elif tag == "delete":
            for k in range(i2 - i1):
                a_line = a_lines[i1 + k]
                a_rows.append(f"<div class='line del'>{token_diff_for_A(a_line, '')}</div>")
            for _ in range(i2 - i1):
                b_rows.append("<div class='line del'></div>")
        elif tag == "insert":
            for k in range(j2 - j1):
                b_line = b_lines[j1 + k]
                b_rows.append(f"<div class='line add'>{token_diff_for_B('', b_line)}</div>")
            for _ in range(j2 - j1):
                a_rows.append("<div class='line add'></div>")
        elif tag == "replace":
            h = max(i2 - i1, j2 - j1)
            for k in range(h):
                a_line = a_lines[i1 + k] if (i1 + k) < i2 else ""
                b_line = b_lines[j1 + k] if (j1 + k) < j2 else ""
                a_rows.append(f"<div class='line rep'>{token_diff_for_A(a_line, b_line) if a_line != '' else ''}</div>")
                b_rows.append(f"<div class='line rep'>{token_diff_for_B(a_line, b_line) if b_line != '' else ''}</div>")

    a_rows += ["</td></tr></table>", "</div>", "</div>"]
    b_rows += ["</td></tr></table>", "</div>", "</div>"]

    html = f"<html><head>{css}</head><body>{''.join(a_rows)}{''.join(b_rows)}</body></html>"
    return html

# ===== 置換ブロック: 縦並びサイド（A 上 / B 下）の差分ダイアログ =====
class DiffDialog(QDialog):
    def __init__(self, a_text: str, b_text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("差分表示")
        self.resize(600, 300)

        # ルートレイアウト
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # タイトル + 区切り罫
        title = QLabel("差分（a → b）")
        title.setStyleSheet("font-weight:700; font-size:14px; color:#111;")
        root.addWidget(title)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet("color:#dee2e6;")
        root.addWidget(divider)

        # 単一ビュー（上下スクロール一体）
        self.view = QTextEdit()
        self.view.setReadOnly(True)
        self.view.setFrameShape(QFrame.NoFrame)
        self.view.setContentsMargins(0, 0, 0, 0)
        self.view.setViewportMargins(0, 0, 0, 0)
        self.view.document().setDocumentMargin(0)
        self.view.setStyleSheet("QTextEdit { font-size:16px; }")
        root.addWidget(self.view, 1)

        # OK のみ
        btns = QDialogButtonBox(QDialogButtonBox.Ok, parent=self)
        btns.accepted.connect(self.accept)
        root.addWidget(btns)

        # HTML 生成して描画
        html = self._build_vertical_html(a_text or "", b_text or "")
        self.view.setHtml(html)

    # ===== 差し替えブロック：行番号なし・縦並び・太い帯レール付き =====
    def _build_vertical_html(self, a: str, b: str) -> str:
        import difflib

        a_lines = a.splitlines()
        b_lines = b.splitlines()
        sm = difflib.SequenceMatcher(a=a_lines, b=b_lines)
        ops = sm.get_opcodes()

        # --- CSSだけ差し替え：タイトル下のアキを徹底的に詰める（レール版） ---
        css = """
        <style>
        html, body, div, p { margin:0; padding:0; }
        body {
            font-family:'Yu Gothic UI','Noto Sans JP',sans-serif;
            color:#111; font-size:16px; line-height:1.80; letter-spacing:0.2px;
        }

        /* ページ上下の余白は最小限に */
        .page { padding-top:6px; padding-bottom:0; }

        /* セクションの下余白は少なめ、最後はゼロ */
        .sec  { margin:0 0 6px 0; }
        .sec:last-child { margin-bottom:0; }

        /* ★Bブロックの前だけスペース（必要量をここで調整） */
        .sec-b { margin-top:12px; }  /* 8～16pxあたりがおすすめ */

        /* ◀ 見出し行：行高を低め＋下マージン0で“見出し直下のアキ”を詰める */
        .sec-head {
            font-weight:600;
            margin:0;                /* ← 下マージン0に */
            line-height:0;        /* ← 行高を小さめに */
        }
        .sec-a .sec-head { color:#d9480f; }
        .sec-b .sec-head { color:#1c7ed6; }

        /* ▼ グレー背景や外枠は使わず、レール＋本文だけで構成 */
        .diff {
            line-height:1.00;
        }

        /* レイアウトはテーブル（QTextEditで安定） */
        table, .wrap { border-collapse:collapse; border-spacing:0; margin:0; }
        .wrap { width:100%; }

        /* ★レール（帯）の太さ：ここで調整（例：18px） */
        .rail   { width:18px; }
        .rail-a { background:#fa5252; }
        .rail-b { background:#4dabf7; }

        /* 本文側の余白：上だけさらに詰めるなら padding-top を小さく */
        .body { padding:4px 10px 6px 10px; }  /* ← 上4pxにして見出し直下をタイトに */

        /* 各行の内側余白（まだ広く感じるなら 1px に） */
        .line { white-space:pre-wrap; padding:2px 4px; }

        /* 行の種別背景 */
        .eq  {}
        .add { background:#e7f5ff; }
        .del { background:#fff0f0; }
        .rep { background:#fff7e6; }

        /* 文字単位ハイライト */
        .tok-ins { color:#1c7ed6; background:#d0ebff; border-radius:2px; }
        .tok-del { color:#c92a2a; background:#ffe3e3; border-radius:2px; }
        </style>
        """

        def esc(s: str) -> str:
            return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

        def token_diff_for_A(a_line: str, b_line: str) -> str:
            sm2 = difflib.SequenceMatcher(a=a_line or "", b=b_line or "")
            parts = []
            for op, i1, i2, j1, j2 in sm2.get_opcodes():
                a_seg = (a_line or "")[i1:i2]
                b_seg = (b_line or "")[j1:j2]
                if op == "equal":
                    parts.append(esc(a_seg))
                elif op == "delete":
                    if a_seg:
                        parts.append(f"<span class='tok-del'>{esc(a_seg)}</span>")
                elif op == "insert":
                    pass
                elif op == "replace":
                    if a_seg:
                        parts.append(f"<span class='tok-del'>{esc(a_seg)}</span>")
            return "".join(parts)

        def token_diff_for_B(a_line: str, b_line: str) -> str:
            sm2 = difflib.SequenceMatcher(a=a_line or "", b=b_line or "")
            parts = []
            for op, i1, i2, j1, j2 in sm2.get_opcodes():
                a_seg = (a_line or "")[i1:i2]
                b_seg = (b_line or "")[j1:j2]
                if op == "equal":
                    parts.append(esc(b_seg))
                elif op == "delete":
                    pass
                elif op == "insert":
                    if b_seg:
                        parts.append(f"<span class='tok-ins'>{esc(b_seg)}</span>")
                elif op == "replace":
                    if b_seg:
                        parts.append(f"<span class='tok-ins'>{esc(b_seg)}</span>")
            return "".join(parts)

        # Aブロック（上）
        a_rows = [
            "<div class='page sec sec-a'>",
            "<div class='diff'>",
            "<table class='wrap' cellspacing='0' cellpadding='0'><tr>",
            "<td class='rail rail-a'></td>",            # ← 太い帯のセル
            "<td class='body'>"
        ]
        # Bブロック（下）
        b_rows = [
            "<div class='page sec sec-b'>",
            "<div class='diff'>",
            "<table class='wrap' cellspacing='0' cellpadding='0'><tr>",
            "<td class='rail rail-b'></td>",            # ← 太い帯のセル
            "<td class='body'>"
        ]

        for tag, i1, i2, j1, j2 in ops:
            if tag == "equal":
                for k in range(i2 - i1):
                    text = esc(a_lines[i1 + k])
                    a_rows.append(f"<div class='line eq'>{text}</div>")
                    b_rows.append(f"<div class='line eq'>{text}</div>")
            elif tag == "delete":
                for k in range(i2 - i1):
                    a_line = a_lines[i1 + k]
                    a_rows.append(f"<div class='line del'>{token_diff_for_A(a_line, '')}</div>")
                for _ in range(i2 - i1):
                    b_rows.append("<div class='line del'></div>")
            elif tag == "insert":
                for k in range(j2 - j1):
                    b_line = b_lines[j1 + k]
                    b_rows.append(f"<div class='line add'>{token_diff_for_B('', b_line)}</div>")
                for _ in range(j2 - j1):
                    a_rows.append("<div class='line add'></div>")
            elif tag == "replace":
                h = max(i2 - i1, j2 - j1)
                for k in range(h):
                    a_line = a_lines[i1 + k] if (i1 + k) < i2 else ""
                    b_line = b_lines[j1 + k] if (j1 + k) < j2 else ""
                    a_rows.append(f"<div class='line rep'>{token_diff_for_A(a_line, b_line) if a_line != '' else ''}</div>")
                    b_rows.append(f"<div class='line rep'>{token_diff_for_B(a_line, b_line) if b_line != '' else ''}</div>")

        # クローズ
        a_rows += ["</td></tr></table>", "</div>", "</div>"]
        b_rows += ["</td></tr></table>", "</div>", "</div>"]

        html = f"<html><head>{css}</head><body>{''.join(a_rows)}{''.join(b_rows)}</body></html>"
        return html
    # ===== 差し替えブロックおわり =====

# ------------------------------------------------------------
# Qt Models / Delegate / Style / Proxy
# ------------------------------------------------------------
# ===== ツールチップ遅延表示フィルタ（完全版：差し替え用） =====
from typing import Optional
from PySide6.QtCore import QObject, QTimer, QPoint, QEvent, Qt, QModelIndex
from PySide6.QtWidgets import QToolTip, QTableView
from PySide6.QtGui import QHelpEvent

class ToolTipDebouncer(QObject):
    """
    QTableView の ToolTip を少し遅らせて表示し、ホバー移動中の無駄な
    ToolTipRole 計算（重いHTML生成）を抑制します。
    """
    def __init__(self, view: QTableView, delay_ms: int = 200, parent=None):
        super().__init__(parent)
        self.view = view
        self.delay_ms = int(max(0, delay_ms))
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self._pending_index: Optional[QModelIndex] = None
        self._pending_global_pos: Optional[QPoint] = None
        self.timer.timeout.connect(self._show_tooltip)

    def _show_tooltip(self):
        idx = self._pending_index
        self._pending_index = None  # 先にクリア
        if not idx or not idx.isValid():
            return
        model = self.view.model()
        if not model:
            return
        try:
            text = model.data(idx, Qt.ToolTipRole)
        except Exception:
            text = None
        if text:
            pos = self._pending_global_pos or self.view.mapToGlobal(QPoint(0, 0))
            QToolTip.showText(pos, text, self.view)
        self._pending_global_pos = None

    def eventFilter(self, obj, ev):
        # ToolTip イベントだけ捕捉して遅延表示
        if ev.type() == QEvent.ToolTip and isinstance(ev, QHelpEvent):
            # viewport 座標で index を取得し、グローバル座標も保存
            idx = self.view.indexAt(ev.pos())
            self._pending_index = idx if idx.isValid() else None
            self._pending_global_pos = ev.globalPos()
            # 高速移動中は何度も再起動され、実行は発火時（＝静止時）のみ
            self.timer.start(self.delay_ms)
            return True  # 既定の即時ツールチップは抑止
        # ビューから離れたらキャンセル
        if ev.type() in (QEvent.Leave, QEvent.Hide):
            self.timer.stop()
            QToolTip.hideText()
            self._pending_index = None
            self._pending_global_pos = None
        return False
# ===== ここまで =====

class UnifiedModel(QAbstractTableModel):
    def __init__(self, df=pd.DataFrame(), parent=None):
        super().__init__(parent)
        self._df = df.copy()
        self._checks: List[bool] = [False] * len(self._df)
        self._col_idx = {}
        self._row_palette = [QColor("#D6E9FF"), QColor("#DFF5D8"), QColor("#FFF1C2"), QColor("#F1D4FF")]
        self._row_alpha = 36

    def setDataFrame(self, df: pd.DataFrame, copy: bool = True):
        self.beginResetModel()
        self._df = df.copy() if copy else df
        # 既存ロジック
        self._checks = [False] * len(self._df)
        self._col_idx = {name: i for i, name in enumerate(self._df.columns)}
        self.endResetModel()

    def _row_color_for_gid(self, gid: Optional[int]) -> Optional[QColor]:
        if gid is None or pd.isna(gid):
            return None
        gid = int(gid)
        c = self._row_palette[(gid - 1) % len(self._row_palette)]
        qc = QColor(c)
        qc.setAlpha(self._row_alpha)
        return qc

    def rowCount(self, parent=QModelIndex()):
        return len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return 1 + len(self._df.columns)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            if section == 0:
                return "✓"
            name = str(self._df.columns[section - 1])
            if name == "a_count":
                return "a数"
            if name == "b_count":
                return "b数"
            if name == "score":
                return "類似度"
            if name.lower() == "scope":
                return "対象"
            return "一致要因" if name == "一致要因" else name
        return str(section + 1)

    def flags(self, index: QModelIndex):
        if not index.isValid():
            return Qt.NoItemFlags
        flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if index.column() == 0:
            flags |= Qt.ItemIsUserCheckable
        return Qt.ItemFlags(flags)

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        r, c = index.row(), index.column()

        # 0列はチェックボックス
        if role == Qt.CheckStateRole and c == 0:
            return Qt.Checked if self._checks[r] else Qt.Unchecked

        # 表示テキスト
        if role == Qt.DisplayRole:
            if c == 0:
                return None
            v = self._df.iat[r, c - 1]
            return "" if pd.isna(v) else str(v)

        # ツールチップ：a / b 列だけ簡易差分HTML（A上/B下）
        if role == Qt.ToolTipRole:
            if c == 0:
                return None
            if not self._col_idx:
                self._col_idx = {name: i for i, name in enumerate(self._df.columns)}
            col_name = self._df.columns[c - 1]
            if col_name in ("a", "b"):
                idx_a = self._col_idx.get("a"); idx_b = self._col_idx.get("b")
                if idx_a is None or idx_b is None:
                    return None
                a_val = self._df.iat[r, idx_a]
                b_val = self._df.iat[r, idx_b]
                a_s = "" if a_val is None else str(a_val)
                b_s = "" if b_val is None else str(b_val)
                # ★ 長過ぎる場合は生成しない（軽量化）
                if (len(a_s) + len(b_s)) > 200:
                    return None
                try:
                    return html_quick_ab_diff(a_s, b_s)
                except Exception:
                    return None
            return None

        # 行背景（gidによる着色）
        if role == Qt.BackgroundRole:
            try:
                if "gid" in self._col_idx:
                    gid_val = self._df.iat[r, self._col_idx["gid"]]
                    qc = self._row_color_for_gid(gid_val)
                    if qc:
                        return qc
            except Exception:
                pass

        # 数値系は右寄せ
        if role == Qt.TextAlignmentRole:
            col_name = self._df.columns[c - 1] if c > 0 else ""
            if col_name in ("gid", "字数", "a_count", "b_count", "score", "a数", "b数"):
                return Qt.AlignRight | Qt.AlignVCenter

        return None

    def setData(self, index: QModelIndex, value, role=Qt.EditRole):
        if not index.isValid():
            return False
        if index.column() == 0 and role == Qt.CheckStateRole:
            self._checks[index.row()] = (value == Qt.Checked)
            self.dataChanged.emit(index, index, [Qt.CheckStateRole])
            return True
        return False

    def check_all(self, checked: bool):
        if not self._checks:
            return
        for i in range(len(self._checks)):
            self._checks[i] = checked
        self.dataChanged.emit(self.index(0,0), self.index(len(self._checks)-1,0), [Qt.CheckStateRole])

class CheckBoxDelegate(QStyledItemDelegate):
    def paint(self, painter: QPainter, option, index):
        if index.column() != 0:
            return super().paint(painter, option, index)
        state = index.model().data(index, Qt.CheckStateRole)
        rect = option.rect
        size = min(rect.height(), 16)
        x = rect.x() + (rect.width() - size) // 2
        y = rect.y() + (rect.height() - size) // 2
        r = QRect(x, y, size, size)
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        pen = QPen(QColor("#111")); pen.setWidth(1); painter.setPen(pen); painter.setBrush(QColor("#fff"))
        painter.drawRoundedRect(r, 2, 2)
        if state == Qt.Checked:
            pen = QPen(QColor("#111")); pen.setWidth(2); painter.setPen(pen)
            x1 = r.x() + int(size*0.22); y1 = r.y() + int(size*0.56)
            x2 = r.x() + int(size*0.46); y2 = r.y() + int(size*0.80)
            x3 = r.x() + int(size*0.80); y3 = r.y() + int(size*0.28)
            painter.drawLine(x1,y1,x2,y2); painter.drawLine(x2,y2,x3,y3)
        painter.restore()

class HighContrastCheckboxStyle(QProxyStyle):
    def pixelMetric(self, metric, option=None, widget=None):
        if metric in (QStyle.PM_IndicatorWidth, QStyle.PM_IndicatorHeight):
            return 16
        return super().pixelMetric(metric, option, widget)

    def drawPrimitive(self, elem, opt, painter: QPainter, widget=None):
        if elem == QStyle.PE_IndicatorCheckBox:
            r = opt.rect; size = min(r.height(), 16)
            x = r.x() + (r.width() - size) // 2; y = r.y() + (r.height() - size) // 2
            box = QRect(x,y,size,size)
            painter.save(); painter.setRenderHint(QPainter.Antialiasing, True)
            pen = QPen(QColor("#111")); pen.setWidth(1); painter.setPen(pen); painter.setBrush(QColor("#fff"))
            painter.drawRoundedRect(box, 2, 2)
            if (opt.state & QStyle.State_On) or (opt.state & QStyle.State_NoChange):
                pen = QPen(QColor("#111")); pen.setWidth(2); painter.setPen(pen)
                x1 = box.x()+int(size*0.22); y1 = box.y()+int(size*0.56)
                x2 = box.x()+int(size*0.46); y2 = box.y()+int(size*0.80)
                x3 = box.x()+int(size*0.80); y3 = box.y()+int(size*0.28)
                painter.drawLine(x1,y1,x2,y2); painter.drawLine(x2,y2,x3,y3)
            painter.restore(); return
        super().drawPrimitive(elem, opt, painter, widget)

class PandasModel(QAbstractTableModel):
    def __init__(self, df=pd.DataFrame(), parent=None):
        super().__init__(parent)
        self._df = df.copy()
    def setDataFrame(self, df: pd.DataFrame, copy: bool = True):
        self.beginResetModel()
        self._df = df.copy() if copy else df
        self.endResetModel()
    def rowCount(self, parent=QModelIndex()):
        return len(self._df)
    def columnCount(self, parent=QModelIndex()):
        return len(self._df.columns)
    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            v = self._df.iat[index.row(), index.column()]
            return "" if pd.isna(v) else str(v)
        return None
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        return str(self._df.columns[section]) if orientation == Qt.Horizontal else str(section + 1)

class UnifiedFilterProxy(QSortFilterProxyModel):
    def __init__(self):
        super().__init__()
        self._text = ""
        self._reasons: Set[str] = set()
        self._scopes: Set[str] = set()
        self._hide_prefix_edge = False
        self._hide_suffix_edge = False
        self._hide_numeric_only = False
        # ▼ 新規：短文（aの文字数）フィルタ
        self._hide_short_enabled = False
        self._hide_short_len = 0

        self._collator = QCollator()
        self._collator.setNumericMode(True)
        self._collator.setCaseSensitivity(Qt.CaseInsensitive)

        # 列キャッシュ
        self._cols: Dict[str, int] = {}

    # ---- 外部API（既存） ----
    def setTextFilter(self, text: str):
        self._text = text or ""
        self.invalidateFilter()

    def setReasonFilter(self, reasons: Set[str]):
        self._reasons = set(reasons or [])
        self.invalidateFilter()

    def setScopeFilter(self, scopes: Set[str]):
        self._scopes = set(scopes or [])
        self.invalidateFilter()

    def setHidePrefixEdge(self, hide: bool):
        self._hide_prefix_edge = bool(hide)
        self.invalidateFilter()

    def setHideSuffixEdge(self, hide: bool):
        self._hide_suffix_edge = bool(hide)
        self.invalidateFilter()

    def setHideNumericOnly(self, hide: bool):
        self._hide_numeric_only = bool(hide)
        self.invalidateFilter()

    # ---- 新規API：短文フィルタ ----
    def setHideShortEnabled(self, enabled: bool):
        self._hide_short_enabled = bool(enabled)
        self.invalidateFilter()

    def setHideShortLength(self, n: int):
        try:
            n_int = int(n)
        except Exception:
            n_int = 0
        self._hide_short_len = max(0, n_int)
        self.invalidateFilter()

    # ---- 列キャッシュ構築 ----
    def setSourceModel(self, model):
        super().setSourceModel(model)
        if model:
            model.modelReset.connect(self._rebuild_cols)
            model.layoutChanged.connect(self._rebuild_cols)
        self._rebuild_cols()

    def _rebuild_cols(self):
        self._cols.clear()
        m = self.sourceModel()
        if not m:
            return
        for c in range(m.columnCount()):
            h = m.headerData(c, Qt.Horizontal, Qt.DisplayRole)
            if isinstance(h, str):
                name = h.strip()
                # ▼ “字数” もキャッシュ対象に追加
                if name in {
                    "a", "b", "一致要因", "理由", "reason", "対象", "scope", "target",
                    "端差", "数字以外一致", "字数"
                }:
                    self._cols[name] = c

    def _col(self, *names: str) -> int:
        for nm in names:
            if nm in self._cols:
                return self._cols[nm]
        return -1

    # ---- フィルタ本体 ----
    def filterAcceptsRow(self, row, parent):
        m = self.sourceModel()
        if not m:
            return True

        # 1) scope
        if self._scopes:
            c_scope = self._col("対象", "scope", "target")
            if c_scope >= 0:
                v = m.data(m.index(row, c_scope, parent), Qt.DisplayRole) or ""
                if str(v) not in self._scopes:
                    return False

        # 2) 全文フィルタ
        if self._text:
            needle = self._text.lower()
            cols = [
                self._col("a"), self._col("b"),
                self._col("一致要因", "理由", "reason"),
                self._col("対象", "scope", "target")
            ]
            cols = [c for c in cols if c >= 0]
            if not cols:
                cols = list(range(m.columnCount()))
            hit = False
            for c in cols:
                v = m.data(m.index(row, c, parent), Qt.DisplayRole)
                if v is not None and needle in str(v).lower():
                    hit = True
                    break
            if not hit:
                return False

        # 3) 端差（前/後1字）
        if self._hide_prefix_edge or self._hide_suffix_edge:
            c_edge = self._col("端差")
            edge_val = ""
            if c_edge >= 0:
                edge_val = (m.data(m.index(row, c_edge, parent), Qt.DisplayRole) or "").strip()
            else:
                ca, cb = self._col("a"), self._col("b")
                if ca >= 0 and cb >= 0:
                    a_txt = m.data(m.index(row, ca, parent), Qt.DisplayRole) or ""
                    b_txt = m.data(m.index(row, cb, parent), Qt.DisplayRole) or ""
                    try:
                        edge_val = classify_edge_diff_jp(str(a_txt), str(b_txt))
                    except Exception:
                        edge_val = ""
            if self._hide_prefix_edge and edge_val in ("前1字有無", "前1字違い"):
                return False
            if self._hide_suffix_edge and edge_val in ("後1字有無", "後1字違い"):
                return False

        # 4) 数字以外一致
        if self._hide_numeric_only:
            c_num = self._col("数字以外一致")
            is_num_only = False
            if c_num >= 0:
                v = (m.data(m.index(row, c_num, parent), Qt.DisplayRole) or "").strip()
                is_num_only = (v == "数字以外一致")
            else:
                ca, cb = self._col("a"), self._col("b")
                if ca >= 0 and cb >= 0:
                    a_txt = m.data(m.index(row, ca, parent), Qt.DisplayRole) or ""
                    b_txt = m.data(m.index(row, cb, parent), Qt.DisplayRole) or ""
                    try:
                        is_num_only = is_numeric_only_diff(str(a_txt), str(b_txt))
                    except Exception:
                        is_num_only = False
            if is_num_only:
                return False

        # 4.5) ▼ 新規：短文フィルタ（a の文字数が閾値以下なら非表示）
        if self._hide_short_enabled and self._hide_short_len > 0:
            # まず “字数” 列があればそれを使う（Workerで付与済み想定）
            c_len = self._col("字数")
            a_len = None
            if c_len >= 0:
                try:
                    v = m.data(m.index(row, c_len, parent), Qt.DisplayRole)
                    a_len = int(str(v)) if v is not None else None
                except Exception:
                    a_len = None
            # なければ a 列から長さを算出
            if a_len is None:
                ca = self._col("a")
                if ca >= 0:
                    a_txt = m.data(m.index(row, ca, parent), Qt.DisplayRole) or ""
                    a_len = len(str(a_txt))
            if a_len is not None and a_len <= self._hide_short_len:
                return False

        # 5) 一致要因
        if self._reasons:
            c_reason = self._col("一致要因", "理由", "reason")
            if c_reason >= 0:
                v = m.data(m.index(row, c_reason, parent), Qt.DisplayRole) or ""
                if str(v) not in self._reasons:
                    return False

        return True

    # ---- 並び替え（既存） ----
    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        m = self.sourceModel()
        if not m:
            return super().lessThan(left, right)

        v1 = m.data(left, Qt.DisplayRole)
        v2 = m.data(right, Qt.DisplayRole)

        def to_num(x):
            if x is None:
                return None
            s = str(x).strip()
            try:
                return float(s)
            except Exception:
                return None

        n1, n2 = to_num(v1), to_num(v2)
        if n1 is not None and n2 is not None:
            return n1 < n2

        s1 = "" if v1 is None else str(v1)
        s2 = "" if v2 is None else str(v2)
        return self._collator.compare(s1, s2) < 0

# ------------------------------------------------------------
# Worker（★進捗配分を後半重めに再設計）
# ------------------------------------------------------------
class AnalyzerWorker(QObject):
    finished = Signal(dict, str)
    progress = Signal(int)
    progress_text = Signal(str)
    def __init__(self, files, use_mecab, read_th, char_th):
        super().__init__()
        self.files = files; self.use_mecab = use_mecab
        self.read_th = read_th; self.char_th = char_th
        self.top_k_lex = 4000
        self.min_count_lex = 1
        self.min_len = 1; self.max_len = 120

    def _emit(self, v: int): self.progress.emit(max(0, min(100, int(v))))

    # ★ 進捗テキストは “作業名のみ”
    def _subprogress_factory(self, start: int, end: int, label: str):
        span = max(1, end - start)
        state = {"p": -1}
        def cb(i: int, n: int):
            try:
                n_ = max(1, int(n))
                i_int = max(0, int(i))
            except Exception:
                n_ = 1
                i_int = 0
            p = start + int((span * i_int) / n_)
            if p != state["p"]:
                self._emit(p)
                state["p"] = p
            # “％表示しない” → 作業名だけ
            if i_int == n_ or i_int % max(1, n_ // 20) == 0:
                self.progress_text.emit(f"{label}中…")
        return cb

    def run(self):
        try:
            # 0〜10%: PDF抽出
            pages = []
            n_files = max(1, len(self.files))
            for i, f in enumerate(self.files, 1):
                self.progress_text.emit("PDF抽出中…")
                pages += extract_pages(f)
                self._emit(int(10 * i / n_files))
            if not pages:
                raise RuntimeError("PDFからテキストが取得できませんでした。")

            # 12%: 細粒度トークン収集（参照）
            self.progress_text.emit("細粒度トークン収集（参照）中…")
            if self.use_mecab and HAS_MECAB:
                tokens_fine = extract_candidates_regex(pages, self.min_len, self.max_len, self.min_count_lex)
                self._emit(12)
                # 15%: 語彙構築
                self.progress_text.emit("細粒度語彙の構築中…")
                df_lex = collect_lexicon_general(pages, top_k=self.top_k_lex, min_count=self.min_count_lex)
                self._emit(15)
                # 15〜25%: 細粒度ペア生成
                df_pairs_lex_general = build_synonym_pairs_general(
                    df_lex, read_sim_th=self.read_th, char_sim_th=self.char_th, scope="語彙",
                    progress_cb=self._subprogress_factory(15, 25, "細粒度ペア生成")
                )
            else:
                raise RuntimeError("MeCab (fugashi/unidic-lite) が必要です。")

            # 27%: 複合語候補（メイン）
            tokens_compound_main = extract_candidates_compound_alljoin(
                pages, min_len=self.min_len, max_len=self.max_len,
                min_count=self.min_count_lex, top_k=0, use_mecab=True
            )
            self._emit(27)

            # 補助（MeCabなし）もマージ
            self.progress_text.emit("複合語候補抽出中…")
            tokens_compound_fb = extract_candidates_compound_alljoin(
                pages, min_len=self.min_len, max_len=self.max_len,
                min_count=self.min_count_lex, top_k=0, use_mecab=False
            )
            c_all = Counter()
            for w, c in tokens_compound_main:
                c_all[w] += c
            for w, c in tokens_compound_fb:
                c_all[w] += c
            tokens_compound = sorted(c_all.items(), key=lambda x: (-x[1], x[0]))

            # 28〜55%: 複合語ペア生成
            df_pairs_compound = build_synonym_pairs_char_only(
                tokens_compound,
                char_sim_th=self.char_th,            # ← 固定0.90を外すのがオススメ
                top_k=self.top_k_lex, scope="複合語",
                progress_cb=self._subprogress_factory(28, 55, "複合語ペア生成"),
                read_sim_th=self.read_th             # ← 追加：読みもしきい値で判定
            )

            # 57%: 文節候補抽出
            self.progress_text.emit("文節候補抽出中…")
            tokens_bunsetsu = extract_candidates_bunsetsu(
                pages, min_len=self.min_len, max_len=self.max_len,
                min_count=self.min_count_lex, top_k=0
            )
            self._emit(57)

            # 58〜80%: 文節ペア生成
            df_pairs_bunsetsu = build_synonym_pairs_char_only(
                tokens_bunsetsu,
                char_sim_th=self.char_th,            # ← 同上
                top_k=self.top_k_lex, scope="文節",
                progress_cb=self._subprogress_factory(58, 80, "文節ペア生成"),
                read_sim_th=self.read_th             # ← 追加
            )

            # 88%: 統合
            self.progress_text.emit("候補統合中…")
            df_unified = unify_pairs(df_pairs_lex_general, df_pairs_compound, df_pairs_bunsetsu)
            self._emit(88)

            # 90%: 単独仮名除去
            if not df_unified.empty and "a" in df_unified.columns:
                df_unified = df_unified[~df_unified["a"].apply(is_single_kana_char)].reset_index(drop=True)
            self._emit(90)

            # 92%: トークン一覧整形
            parts = []
            if len(tokens_fine) > 0:
                df_fine = pd.DataFrame(tokens_fine, columns=["token", "count"]); df_fine["type"] = "fine"; parts.append(df_fine)
            if len(tokens_compound) > 0:
                df_comp = pd.DataFrame(tokens_compound, columns=["token", "count"]); df_comp["type"] = "compound"; parts.append(df_comp)
            if len(tokens_bunsetsu) > 0:
                df_bun = pd.DataFrame(tokens_bunsetsu, columns=["token", "count"]); df_bun["type"] = "bunsetsu"; parts.append(df_bun)
            df_tokens = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["type", "token", "count"])
            if not df_tokens.empty:
                df_tokens = df_tokens["type token count".split()].sort_values(["type", "count", "token"], ascending=[True, False, True])
            self._emit(92)

            # 93%: 表示用ラベル変換（既存）
            df_unified = apply_reason_ja(df_unified)

            # 93.5%: 端差 / 数字以外一致（ベクトル化に置換）
            try:
                if df_unified is not None and not df_unified.empty and "a" in df_unified.columns and "b" in df_unified.columns:
                    df_unified["端差"] = _edge_labels_vectorized(df_unified)
                    df_unified["数字以外一致"] = _numeric_only_label_vectorized(df_unified)
            except Exception:
                pass

            # 94%: 字数を Worker 側で付与（UIの仕事を減らす）
            if not df_unified.empty and "a" in df_unified.columns:
                try:
                    df_unified["字数"] = df_unified["a"].astype("string").str.len().fillna(0).astype(int)
                except Exception:
                    df_unified["字数"] = df_unified["a"].astype(str).str.len().fillna(0).astype(int)

            # 95%: 表示列の概ねの順番をここで整えておく（gid は UI 付与のため除外）
            cols = list(df_unified.columns)
            pref = [c for c in ["字数", "a", "b"] if c in cols]
            rest = [c for c in cols if c not in pref]
            df_unified = df_unified[pref + rest]

            # 96%: グループ割当（既存）
            self.progress_text.emit("グループ割当中…")
            df_groups, surf2gid, gid2members = build_variation_groups(df_unified, df_lex)

            # 97%: gid 付与（UI軽量化）
            if not df_unified.empty and "a" in df_unified.columns:
                df_unified["gid"] = df_unified["a"].map(
                    lambda x: surf2gid.get(x) if isinstance(x, str) and x else None
                )

            # 98%: 完了
            self._emit(98)
            self.progress_text.emit("仕上げ中…")
            self.finished.emit(
                {"unified": df_unified, "tokens": df_tokens, "groups": df_groups,
                "surf2gid": surf2gid, "gid2members": gid2members}, ""
            )
        except Exception as e:
            self.finished.emit({}, str(e))


# ------------------------------------------------------------
# Theme / Drag helpers / GUI
# ------------------------------------------------------------
def apply_light_theme(app: QApplication, base_font_size=11):
    family = "Yu Gothic UI"
    app.setFont(QFont(family, base_font_size))

    pal = QPalette()
    pal.setColor(QPalette.Window, QColor("#fff"))
    pal.setColor(QPalette.Base, QColor("#fff"))
    pal.setColor(QPalette.AlternateBase, QColor("#f8f9fa"))
    pal.setColor(QPalette.Text, QColor("#111"))
    pal.setColor(QPalette.WindowText, QColor("#111"))
    pal.setColor(QPalette.Button, QColor("#fff"))
    pal.setColor(QPalette.ButtonText, QColor("#111"))
    pal.setColor(QPalette.Highlight, QColor("#e7f5ff"))
    pal.setColor(QPalette.HighlightedText, QColor("#0b7285"))

    # ★★★ 追加：ツールチップの白背景・文字色をパレットで固定
    pal.setColor(QPalette.ToolTipBase, QColor("#ffffff"))
    pal.setColor(QPalette.ToolTipText, QColor("#111111"))

    app.setPalette(pal)
    app.setStyle(HighContrastCheckboxStyle(app.style()))

    # ★★★ 追加：QToolTip の背景・枠線・文字色をスタイルで固定
    # 既存の app.setStyleSheet に QToolTip のルールを追記しています
    app.setStyleSheet("""
    QMainWindow, QWidget { background:#fff; color:#111; }
    QLabel, QGroupBox, QCheckBox { color:#111; }

    /* GroupBoxの余白を詰める */
    QGroupBox {
        border:1px solid #e9ecef; border-radius:8px;
        margin-top:12px;
    }
    QGroupBox::title {
        subcontrol-origin: margin; subcontrol-position: top left;
        padding:2px 6px; font-weight:600;
    }

    /* 入力のパディング縮小 */
    QLineEdit {
        background:#fff; color:#111;
        border:1px solid #dee2e6; border-radius:6px;
        padding:4px 6px; min-height: 24px;
    }
    QLineEdit:focus { border:1px solid #4dabf7; }

    QPushButton {
        background:#4dabf7; color:#fff; border-radius:8px;
        padding:6px 10px; border:1px solid #339af0;
    }
    QPushButton:hover { background:#339af0; }
    QPushButton:disabled { background:#cfd4da; color:#888; border-color:#cfd4da; }
    #btnRun {
        background:#f59f00; border-color:#f08c00;
        font-weight:700; font-size:13px; padding:8px 12px; min-height:36px;
    }
    #btnRun:hover { background:#f08c00; }

    QTabWidget::pane { border:1px solid #dee2e6; background:#fff; }
    QTabBar::tab {
        background:#f1f3f5; color:#111; padding:4px 8px; margin:2px; border-radius:6px;
    }
    QTabBar::tab:selected { background:#4dabf7; color:#fff; }

    QTableView {
        background:#fff; color:#111; gridline-color:#dee2e6;
        alternate-background-color:#f8f9fa;
        selection-background-color:#e7f5ff; selection-color:#0b7285;
    }
    QHeaderView::section {
        background:#f1f3f5; color:#212529;
        padding:4px 6px; border:1px solid #dee2e6; font-weight:600;
    }
    QTableView QTableCornerButton::section { background:#f1f3f5; border:1px solid #dee2e6; }

    QProgressBar { border:1px solid #dee2e6; border-radius:6px; text-align:center; height:16px; }
    QProgressBar::chunk { background:#4dabf7; }

    /* コンパクト化用：行間を詰める */
    QCheckBox { spacing:6px; }

    /* ★★★ ここが重要：ツールチップを白背景＋淡い枠に固定（ダークテーマでも白に）
       必要なら背景を #f8f9fa に変えると淡いグレーになります。 */
    QToolTip {
        background-color: #ffffff;
        color: #111111;
        border: 1px solid #dee2e6;
        padding: 6px 8px;
        border-radius: 6px;
        font-family: 'Yu Gothic UI','Noto Sans JP',sans-serif;
        font-size: 12px;
    }
    """)

def extract_pdf_paths_from_urls(urls) -> List[str]:
    out = []
    for u in urls:
        p = u.toLocalFile()
        if not p:
            continue
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for fn in files:
                    if fn.lower().endswith(".pdf"):
                        out.append(os.path.join(root, fn))
        elif p.lower().endswith(".pdf"):
            out.append(p)
    return sorted(dict.fromkeys(out))

class DropArea(QTextEdit):
    filesChanged = Signal(list)
    def __init__(self):
        super().__init__(); self.setAcceptDrops(True); self.setReadOnly(True)
        self.setPlaceholderText("ここにPDFをドラッグ＆ドロップしてください（複数可）")
        self.setStyleSheet("QTextEdit{border:2px dashed #4dabf7; border-radius:10px; background:#fff; padding:10px; color:#111; min-height:80px;}")
    def dragEnterEvent(self, e):
        e.acceptProposedAction() if e.mimeData().hasUrls() else e.ignore()
    def dragMoveEvent(self, e):
        e.acceptProposedAction() if e.mimeData().hasUrls() else e.ignore()
    def dropEvent(self, e):
        if not e.mimeData().hasUrls():
            e.ignore(); return
        files = extract_pdf_paths_from_urls(e.mimeData().urls())
        if files:
            self.setText("\n".join(files)); self.filesChanged.emit(files)
        e.acceptProposedAction()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("PDF 表記ゆれチェック [ver.1.20]"); self.resize(1180, 860)
        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # ---- 左パネル（固定幅 280px）----
        left_panel = QWidget(); left_panel.setFixedWidth(280)
        left = QVBoxLayout(left_panel)
        left.setContentsMargins(8, 8, 8, 8)
        left.setSpacing(8)

        gb_files = QGroupBox("入力PDF"); v_files = QVBoxLayout(gb_files)
        self.drop = DropArea(); v_files.addWidget(self.drop)
        row = QHBoxLayout(); self.btn_browse = QPushButton("ファイルを選択"); self.btn_clear = QPushButton("クリア")
        row.addWidget(self.btn_browse); row.addWidget(self.btn_clear); v_files.addLayout(row)
        self.list_files = QListWidget(); self.lbl_count = QLabel("選択ファイル：0 件")
        v_files.addWidget(self.list_files); v_files.addWidget(self.lbl_count)

        gb_params = QGroupBox("設定"); form = QFormLayout(gb_params)
        self.dsb_read = QDoubleSpinBox(); self.dsb_read.setRange(0.0,1.0); self.dsb_read.setSingleStep(0.05); self.dsb_read.setValue(0.90)
        self.dsb_char = QDoubleSpinBox(); self.dsb_char.setRange(0.0,1.0); self.dsb_char.setSingleStep(0.05); self.dsb_char.setValue(0.85)
        form.addRow("読み 類似しきい値", self.dsb_read); form.addRow("文字 類似しきい値", self.dsb_char)

        # 下揃えブロック
        self.btn_run = QPushButton("解析開始"); self.btn_run.setObjectName("btnRun")
        self.progress = QProgressBar(); self.progress.setRange(0,100); self.progress.setValue(0)
        self.lbl_stage = QLabel("待機中")
        self.lbl_elapsed = QLabel("経過 00:00")

        left.addWidget(gb_files)
        left.addWidget(gb_params)
        left.addStretch(1)
        left.addWidget(self.btn_run)
        left.addWidget(self.progress)
        left.addWidget(self.lbl_stage)
        left.addWidget(self.lbl_elapsed)

        # --- 右パネル（コンパクト配置） ---
        right_panel = QWidget(); right = QVBoxLayout(right_panel)
        right.setContentsMargins(6, 6, 6, 6)
        right.setSpacing(6)

        # ★統合グループ：フィルタ（全文 / 対象 / 一致要因 / 端差 を1つにまとめる）
        gb_filters = QGroupBox("フィルタ")
        v_filters = QVBoxLayout(gb_filters)
        v_filters.setContentsMargins(8, 8, 8, 8)
        v_filters.setSpacing(4)

        # 1) 全文フィルタ行
        row_full = QHBoxLayout()
        row_full.setSpacing(8)
        row_full.addWidget(QLabel("全文:"))
        self.ed_filter = QLineEdit()
        self.ed_filter.setPlaceholderText("絞り込みたい語や表現を入力")
        row_full.addWidget(self.ed_filter, 1)

        # ▼ 追加：短文フィルタ（aの文字数で絞り込み）
        self.chk_shortlen = QCheckBox("字数")
        self.sb_shortlen = QSpinBox()
        self.sb_shortlen.setRange(1, 120)
        self.sb_shortlen.setValue(3)  # 初期表示
        lbl_short_suffix = QLabel("以下を隠す")

        self.chk_shortlen.setToolTip("a 列の文字数が指定値以下の行を非表示にします。")
        self.sb_shortlen.setToolTip("非表示にする最大文字数（a 列の長さ、1〜120）")

        # 行内に追加（「全文：」と同じ行に並べる）
        row_full.addSpacing(6)
        row_full.addWidget(self.chk_shortlen)
        row_full.addWidget(self.sb_shortlen)
        row_full.addWidget(lbl_short_suffix)

        v_filters.addLayout(row_full)

        # 区切り罫
        sep1 = QFrame(); sep1.setFrameShape(QFrame.HLine); sep1.setFrameShadow(QFrame.Sunken)
        v_filters.addWidget(sep1)

        # 2) 対象フィルタ行（語彙/複合語/文節）
        row_scope = QHBoxLayout(); row_scope.setSpacing(8)
        row_scope.addWidget(QLabel("対　　象:"))
        self.scope_labels = ["語彙", "複合語", "文節"]
        self.chk_scopes: Dict[str, QCheckBox] = {}
        for s in self.scope_labels:
            cb = QCheckBox(s); cb.setChecked(True); self.chk_scopes[s] = cb
            row_scope.addWidget(cb)
        row_scope.addStretch(1)
        v_filters.addLayout(row_scope)

        # 区切り罫
        sep2 = QFrame(); sep2.setFrameShape(QFrame.HLine); sep2.setFrameShadow(QFrame.Sunken)
        v_filters.addWidget(sep2)

        # 3) 一致要因フィルタ行
        row_reason = QHBoxLayout(); row_reason.setSpacing(8)
        row_reason.addWidget(QLabel("一致要因:"))
        self.reason_labels = ["基本形一致","基本形の読み一致","読み類似","文字類似","活用違い"]
        self.chk_reasons: Dict[str, QCheckBox] = {}
        for r in self.reason_labels:
            cb = QCheckBox(r); cb.setChecked(True); self.chk_reasons[r] = cb
            row_reason.addWidget(cb)
        row_reason.addStretch(1)
        v_filters.addLayout(row_reason)

        # 区切り罫
        sep3 = QFrame(); sep3.setFrameShape(QFrame.HLine); sep3.setFrameShadow(QFrame.Sunken)
        v_filters.addWidget(sep3)

        # 4) 端差（前/後1字）フィルタ行（★初期OFF）
        row_edge = QHBoxLayout()
        row_edge.setSpacing(8)
        row_edge.addWidget(QLabel("特殊絞込:"))
        self.chk_edge_prefix = QCheckBox("前1文字差を非表示")
        self.chk_edge_suffix = QCheckBox("後1文字差を非表示")
        # ★追加：数字以外一致
        self.chk_num_only = QCheckBox("数字以外一致を非表示")

        self.chk_edge_prefix.setChecked(False)  # ★初期OFF
        self.chk_edge_suffix.setChecked(False)  # ★初期OFF
        self.chk_num_only.setChecked(False)     # ★初期OFF

        self.chk_edge_prefix.setToolTip("「前1字有無」「前1字違い」の候補を表から除外します。")
        self.chk_edge_suffix.setToolTip("「後1字有無」「後1字違い」の候補を表から除外します。")
        self.chk_num_only.setToolTip("数字だけが異なる候補（例: 1時間30分 ↔ 1時間5分）を表から除外します。")

        row_edge.addWidget(self.chk_edge_prefix)
        row_edge.addWidget(self.chk_edge_suffix)
        row_edge.addWidget(self.chk_num_only)   # ★追加
        row_edge.addStretch(1)
        v_filters.addLayout(row_edge)


        right.addWidget(gb_filters)

        # テーブルタブ（そのまま）
        self.tabs = QTabWidget()
        self.view_unified = self._make_table()
        # --- 表の描画を軽くする設定（QTableView 版・行高さ一定） ---
        from PySide6.QtWidgets import QHeaderView

        # ❶ 行高さを一定に（ユーザーの自動リサイズや内容依存を止める）
        vh = self.view_unified.verticalHeader()
        vh.setSectionResizeMode(QHeaderView.Fixed)  # ← 行の高さを固定
        vh.setDefaultSectionSize(28)  # ← お好みで 28〜36 など

        # ❷ 折り返しを無効化（1行表示）
        self.view_unified.setWordWrap(False)  # ← ★ここを False に変更
        self.view_unified.setTextElideMode(Qt.ElideRight)  # 長い場合は「…」で省略

        # ❸ 列ヘッダ側も極端な自動拡張を抑えるなら Interactive を使う
        # hh = self.view_unified.horizontalHeader()
        # hh.setSectionResizeMode(QHeaderView.Interactive)

        # ❹ ツールチップ遅延（既存の ToolTipDebouncer を利用）
        self._tt_debouncer = ToolTipDebouncer(self.view_unified, delay_ms=200, parent=self)
        self.view_unified.viewport().installEventFilter(self._tt_debouncer)


        # ❸ ツールチップ遅延（9-A の ToolTipDebouncer を利用）
        self._tt_debouncer = ToolTipDebouncer(self.view_unified, delay_ms=200, parent=self)
        self.view_unified.viewport().installEventFilter(self._tt_debouncer)
        # --- ここまで ---

        self.view_unified.setColumnWidth(0, 32)
        self.view_unified.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.view_unified.setItemDelegateForColumn(0, CheckBoxDelegate(self.view_unified))
        self.view_unified.setTextElideMode(Qt.ElideNone)

        # 右クリックメニュー（差分表示のみ）
        self.view_unified.setContextMenuPolicy(Qt.CustomContextMenu)
        self.view_unified.customContextMenuRequested.connect(self.on_unified_context_menu)

        self.view_tokens = self._make_table_simple()
        self.tabs.addTab(self.view_unified, "表記ゆれ候補")
        self.tabs.addTab(self.view_tokens, "候補（トークン）")

        # ★テーブルを優先的に広げる
        right.addWidget(self.tabs, 1)

        # 操作行（下部のボタン類）
        ops = QHBoxLayout()
        self.btn_select_all = QPushButton("すべて選択")
        self.btn_clear_sel = QPushButton("すべて解除")
        self.btn_export = QPushButton("Excelエクスポート")
        self.btn_mark = QPushButton("PDFにマーキング")

        ops.addWidget(self.btn_select_all)
        ops.addWidget(self.btn_clear_sel)
        ops.addStretch(1)

        # ▼ 置き換え：リンク風の「ヘルプ」ボタン（CSVエクスポートの左に配置）
        help_url = (
            "https://itpcojp-my.sharepoint.com/:b:/g/personal/masahiro_tanaka_itp_co_jp/"
            "Ec84iGbfkvRFl4ZHhK-XR9YBcjDRoWvi6cf3XX59l2sBzg?e=qvFOZP"
        )
        self.btn_help = QPushButton("ヘルプ")
        self.btn_help.setFlat(True)
        self.btn_help.setCursor(Qt.PointingHandCursor)
        # リンク風の見た目（青＋ホバーで下線）
        self.btn_help.setStyleSheet("""
            QPushButton { color:#1c7ed6; background:transparent; border:none; padding:0 4px; }
            QPushButton:hover { text-decoration: underline; }
        """)
        self.btn_help.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(help_url)))
        ops.addWidget(self.btn_help)

        ops.addSpacing(8)
        ops.addWidget(self.btn_export)
        ops.addWidget(self.btn_mark)

        right.addLayout(ops)

        root.addWidget(left_panel)
        root.addWidget(right_panel)
        root.setStretch(0, 0)
        root.setStretch(1, 1)

        # models & proxies
        self.model_unified = UnifiedModel(); self.model_tokens = PandasModel()
        self.proxy_unified = UnifiedFilterProxy(); self.proxy_unified.setSourceModel(self.model_unified)
        self.proxy_tokens = QSortFilterProxyModel(); self.proxy_tokens.setSourceModel(self.model_tokens)
        self.view_unified.setModel(self.proxy_unified); self.view_tokens.setModel(self.proxy_tokens)
        self.proxy_tokens.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.proxy_tokens.setFilterKeyColumn(1) # 0:type, 1:token, 2:count

        # signals
        self.drop.filesChanged.connect(self.on_files_changed)
        self.btn_browse.clicked.connect(self.on_browse); self.btn_clear.clicked.connect(self.on_clear_files)
        self.btn_run.clicked.connect(self.on_run)
        self.ed_filter.textChanged.connect(self.on_text_filter_changed)

        # ... 既存のシグナル接続の後ろに追記 ...
        self.chk_shortlen.stateChanged.connect(self.on_shortlen_filter_changed)
        self.sb_shortlen.valueChanged.connect(self.on_shortlen_filter_changed)

        # 初期反映（未チェック／値はUIに合わせる）
        self.proxy_unified.setHideShortEnabled(self.chk_shortlen.isChecked())
        self.proxy_unified.setHideShortLength(self.sb_shortlen.value())


        for cb in self.chk_reasons.values():
            cb.stateChanged.connect(self.on_reason_changed)
        for cb in self.chk_scopes.values():
            cb.stateChanged.connect(self.on_scope_changed)

        # ★統合フィルタ：端差（前/後）変更
        self.chk_edge_prefix.stateChanged.connect(self.on_edge_filter_changed)
        self.chk_edge_suffix.stateChanged.connect(self.on_edge_filter_changed)
        self.chk_num_only.stateChanged.connect(self.on_edge_filter_changed)  # ★追加

        # ★端差初期状態（OFF）を明示反映
        self.proxy_unified.setHidePrefixEdge(self.chk_edge_prefix.isChecked())  # False
        self.proxy_unified.setHideSuffixEdge(self.chk_edge_suffix.isChecked())  # False
        self.proxy_unified.setHideNumericOnly(self.chk_num_only.isChecked())    # ★False
        self._t0: Optional[float] = None
        self._timer = QTimer(self); self._timer.setInterval(250); self._timer.timeout.connect(self._update_elapsed)
        self.df_groups = pd.DataFrame()
        self.surf2gid: Dict[str, int] = {}
        self.gid2members: Dict[int, List[str]] = {}
        self._refresh_candidate_count()

        self.btn_select_all.clicked.connect(lambda: self.model_unified.check_all(True))
        self.btn_clear_sel.clicked.connect(lambda: self.model_unified.check_all(False))
        self.view_unified.clicked.connect(self.on_unified_clicked)
        self.btn_mark.clicked.connect(self.on_mark_pdf)
        self.btn_export.clicked.connect(self.on_export)

        self.files: List[str] = []
        self.proxy_unified.setScopeFilter(self._selected_scopes())
        self.proxy_unified.setReasonFilter(self._selected_reasons())

        # ★新規：端差（前/後）既定状態をProxyへ反映
        self.proxy_unified.setHidePrefixEdge(self.chk_edge_prefix.isChecked())
        self.proxy_unified.setHideSuffixEdge(self.chk_edge_suffix.isChecked())

        self._t0: Optional[float] = None
        self._timer = QTimer(self); self._timer.setInterval(250); self._timer.timeout.connect(self._update_elapsed)
        self.df_groups = pd.DataFrame()
        self.surf2gid: Dict[str, int] = {}
        self.gid2members: Dict[int, List[str]] = {}
        self._refresh_candidate_count()

    def _post_show_tweaks(self):
        # 最初の描画完了後にごく軽い微調整だけ行う
        try:
            self._ensure_ab_visible(min_a=220, min_b=220)
        except Exception:
            pass

    # ---- ユーティリティ ----
    def _make_table(self):
        tv = QTableView()
        tv.setSortingEnabled(True); tv.setAlternatingRowColors(True)
        tv.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        tv.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        tv.verticalHeader().setVisible(False); tv.verticalHeader().setDefaultSectionSize(28)
        tv.horizontalHeader().setStretchLastSection(True); tv.horizontalHeader().setHighlightSections(False)
        tv.setShowGrid(True)
        return tv
    def _make_table_simple(self):
        return self._make_table()

    def _ensure_ab_visible(self, min_a=220, min_b=220):
        df = getattr(self.model_unified, "_df", pd.DataFrame())
        if df.empty: return
        cols = list(df.columns)
        try:
            idx_a = cols.index("a"); idx_b = cols.index("b")
        except ValueError:
            return
        view_idx_a = 1 + idx_a  # 先頭にチェック列があるため +1
        view_idx_b = 1 + idx_b
        self.view_unified.setColumnWidth(view_idx_a, max(self.view_unified.columnWidth(view_idx_a), min_a))
        self.view_unified.setColumnWidth(view_idx_b, max(self.view_unified.columnWidth(view_idx_b), min_b))

    # ---- 進捗・件数 ----
    def _update_elapsed(self):
        if self._t0 is None:
            self.lbl_elapsed.setText("経過 00:00"); return
        sec = max(0, int(time.monotonic() - self._t0))
        h, rem = divmod(sec, 3600); m, s = divmod(rem, 60)
        fmt = f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
        self.lbl_elapsed.setText(f"経過 {fmt}")

    def _refresh_candidate_count(self):
        try:
            df_unified = getattr(self.model_unified, "_df", pd.DataFrame())
            total = int(df_unified.shape[0]) if isinstance(df_unified, pd.DataFrame) else 0
            visible = int(self.proxy_unified.rowCount())
            self.tabs.setTabText(0, f"表記ゆれ候補[{visible:,}/{total:,}]")
        except Exception:
            self.tabs.setTabText(0, "表記ゆれ候補[0/0]")

    # ---- DnD & 入出力 ----
    def on_files_changed(self, files: List[str]): self.set_files(files)
    def set_files(self, files: List[str]):
        self.files = files; self.list_files.clear()
        for f in files: self.list_files.addItem(f)
        self.lbl_count.setText(f"選択ファイル：{len(files)} 件")
    def on_browse(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "PDFを選択", "", "PDF Files (*.pdf)")
        if paths: self.set_files(paths)
    def on_clear_files(self):
        self.set_files([]); self.drop.clear()

    def on_run(self):
        if not self.files:
            QMessageBox.warning(self, "注意", "PDFを指定してください。"); return
        if not HAS_MECAB:
            QMessageBox.critical(self, "エラー",
                'MeCab (fugashi/unidic-lite) が必要です。\n'
                'インストール例: pip install "fugashi[unidic-lite]"')
            return
        self.btn_run.setEnabled(False); self.progress.setValue(0)
        self._t0 = time.monotonic(); self._update_elapsed(); self._timer.start()
        self.lbl_stage.setText("開始")
        self.thread = QThread()
        self.worker = AnalyzerWorker(self.files, True, self.dsb_read.value(), self.dsb_char.value())
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.progress_text.connect(self.lbl_stage.setText)
        self.worker.finished.connect(self.on_finished)
        self.worker.finished.connect(lambda *_: self.thread.quit())
        self.thread.finished.connect(lambda: self.btn_run.setEnabled(True))
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_finished(self, results: dict, error: str):
        if error:
            QMessageBox.critical(self, "エラー", error)
            self.lbl_stage.setText("エラー")
            if self._timer.isActive(): self._timer.stop()
            return

        self.lbl_stage.setText("集計完了（表示中…）")
        df_unified = results.get("unified", pd.DataFrame())
        self.df_groups = results.get("groups", pd.DataFrame())
        self.surf2gid = results.get("surf2gid", {}) or {}
        self.gid2members = results.get("gid2members", {}) or {}

        if not df_unified.empty:
            # Worker 側で 'gid' と '字数' を付与済み想定
            df_u = df_unified.copy()

            cols = list(df_u.columns)
            pref = [c for c in ["gid", "字数", "a", "b"] if c in cols]
            rest = [c for c in cols if c not in pref]
            df_u = df_u[pref + rest]
        else:
            df_u = df_unified

        self.model_unified.setDataFrame(df_u)
        self.model_tokens.setDataFrame(results.get("tokens", pd.DataFrame()))
        # 反映中は再描画＆並び替えを止める
        self.view_unified.setUpdatesEnabled(False)
        self.view_unified.setSortingEnabled(False)

        # 列幅は「a/b だけ軽く自動→すぐに上限クリップ」
        # self.view_unified.resizeColumnsToContents()  # ← 1回でOK（tokens 側はスキップで可）
        # self.view_tokens.resizeColumnsToContents() は呼ばない or 後回し

        # a/b 列の最小・最大幅で固定
        self._cap_ab_widths(min_a=220, max_a=320, min_b=220, max_b=320)

        # 補助列の非表示
        self._hide_aux_columns()
        self._ensure_ab_visible(min_a=220, min_b=220)

        self._compact_columns()
        self.view_unified.setSortingEnabled(True)
        self.view_unified.setUpdatesEnabled(True)

        self._refresh_candidate_count()
        QApplication.processEvents()

        self.progress.setValue(100)
        self.lbl_stage.setText("完了")
        if self._timer.isActive(): self._timer.stop()

    def _compact_columns(self):
        view = self.view_unified
        model = view.model()  # Proxy
        if not model:
            return
        hdr = view.horizontalHeader()
        hdr.setMinimumSectionSize(40)  # 小さめOKに

        NUMERIC_NAMES = {"gid", "字数", "a_count", "b_count", "a数", "b数", "score", "類似度"}
        LABEL_NAMES   = {"一致要因", "理由", "対象", "scope", "target"}

        for c in range(model.columnCount()):
            name = model.headerData(c, Qt.Horizontal, Qt.DisplayRole)
            name = (name or "").strip()

            # 目安の最大幅
            if name in NUMERIC_NAMES:
                max_w = 88 if name not in {"gid", "字数"} else 72
            elif name in LABEL_NAMES:
                max_w = 120
            elif name in {"a", "b"}:
                # a/b は専用の _cap_ab_widths で処理するのでスキップ
                hdr.setSectionResizeMode(c, QHeaderView.Interactive)
                continue
            else:
                max_w = 160

            # 現在幅を上限でクリップ
            w = view.columnWidth(c)
            view.setColumnWidth(c, min(max_w, w if w > 0 else max_w))
            hdr.setSectionResizeMode(c, QHeaderView.Interactive)

    def on_unified_clicked(self, proxy_index: QModelIndex):
        if not proxy_index.isValid() or proxy_index.column() != 0: return
        src_index = self.proxy_unified.mapToSource(proxy_index)
        current = self.model_unified.data(src_index, Qt.CheckStateRole)
        new_state = Qt.Unchecked if current == Qt.Checked else Qt.Checked
        self.model_unified.setData(src_index, new_state, Qt.CheckStateRole)

    # ---- 右クリックメニュー（差分表示）----
    def on_unified_context_menu(self, pos: QPoint):
        index = self.view_unified.indexAt(pos)
        if not index.isValid():
            return
        proxy_row = index.row()
        src_index = self.proxy_unified.mapToSource(self.proxy_unified.index(proxy_row, 1))
        row = src_index.row()
        df = getattr(self.model_unified, "_df", pd.DataFrame())
        if df.empty or row < 0 or row >= len(df):
            return
        a = df.at[row, "a"] if "a" in df.columns else ""
        b = df.at[row, "b"] if "b" in df.columns else ""

        menu = QMenu(self)

        # 差分表示のみ
        act_diff = QAction("差分を表示…", self)
        def _show():
            dlg = DiffDialog(str(a) if a is not None else "", str(b) if b is not None else "", self)
            dlg.exec()
        act_diff.triggered.connect(_show)
        menu.addAction(act_diff)

        menu.exec(self.view_unified.viewport().mapToGlobal(pos))

    def on_export(self):
        """
        Excel（全件データ＋初期フィルタ見た目）を出力。
        - データは全件
        - 開いた直後の見た目はGUIと同じ（オートフィルタ宣言＋非該当行を非表示）
        - xlsxwriter 優先。無ければ openpyxl でも近い見た目
        - ※「フィルタ設定」シートは出しません
        """
        import pandas as pd
        import json, math as _math

        # ---------- 0) 全件DFの取得（「選択」列付き） ----------
        try:
            df_all = self._build_unified_df_with_selection_all()
        except Exception:
            src_df = getattr(self.model_unified, "_df", pd.DataFrame()).copy()
            checks = list(getattr(self.model_unified, "_checks", []))
            if src_df.empty:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(self, "情報", "エクスポートするデータがありません。")
                return
            if len(checks) != len(src_df):
                checks = [False] * len(src_df)
            df_all = src_df.reset_index(drop=True)
            df_all.insert(0, "選択", checks)
            if "字数" not in df_all.columns and "a" in df_all.columns:
                df_all["字数"] = df_all["a"].map(lambda x: len(str(x)) if x is not None else 0).astype(int)

        # ---------- 1) Excel安全化（dict/list → JSON、NaN→空） ----------
        def _sanitize_df_for_excel(df: pd.DataFrame) -> pd.DataFrame:
            def coerce(v):
                if v is None: return ""
                if isinstance(v, float) and _math.isnan(v): return ""
                if isinstance(v, (dict, list, tuple, set)):
                    try: return json.dumps(v, ensure_ascii=False)
                    except Exception: return str(v)
                return v
            df = df.copy()
            for c in df.columns:
                if df[c].dtype == "object":
                    df[c] = df[c].map(coerce)
            return df

        df_all = _sanitize_df_for_excel(df_all)

        # ---------- 2) 保存パス ----------
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        default_name = "variants_unified.xlsx"
        path, _ = QFileDialog.getSaveFileName(
            self, "Excelの保存先", default_name, "Excel ファイル (*.xlsx)"
        )
        if not path:
            return

        # ---------- 3) エンジン選択（xlsxwriter 優先） ----------
        try:
            import xlsxwriter  # noqa
            engine = "xlsxwriter"
        except Exception:
            engine = "openpyxl"

        # ---------- 4) GUIフィルタ状態 ----------
        scopes_on  = sorted(list(self._selected_scopes()))  if hasattr(self, "_selected_scopes")  else []
        reasons_on = sorted(list(self._selected_reasons())) if hasattr(self, "_selected_reasons") else []
        hide_prefix = bool(getattr(self, "chk_edge_prefix", None) and self.chk_edge_prefix.isChecked())
        hide_suffix = bool(getattr(self, "chk_edge_suffix", None) and self.chk_edge_suffix.isChecked())
        short_on    = bool(getattr(self, "chk_shortlen", None) and self.chk_shortlen.isChecked())
        short_n     = int(self.sb_shortlen.value()) if short_on else None
        fulltext    = (getattr(self, "ed_filter", None).text() if hasattr(self, "ed_filter") else "") or ""
        fulltext    = fulltext.strip()
        need_full   = bool(fulltext)

        # ---------- 5) 許可リスト（Blanks対応）＆ 補助列 ----------
        BLANK_TOKEN = "Blanks"  # XlsxWriter の特別トークン（"" ではなくこれを使う）

        # 端差
        edge_allowed_list, edge_allow_blank = None, False
        if "端差" in df_all.columns:
            series = df_all["端差"].astype(str)
            uniq = set(s.strip() for s in series.fillna(""))
            deny = set()
            if hide_prefix: deny |= {"前1字有無", "前1字違い"}
            if hide_suffix: deny |= {"後1字有無", "後1字違い"}
            edge_allow_blank = ("" in uniq)  # 空セルも見せる（必要に応じて調整）
            allowed = [v for v in sorted(uniq) if v and v not in deny]
            if edge_allow_blank:
                allowed = [BLANK_TOKEN] + allowed
            edge_allowed_list = allowed

        # 数字以外一致
        num_allowed_list, num_allow_blank = None, False
        if "数字以外一致" in df_all.columns and getattr(self, "chk_num_only", None) and self.chk_num_only.isChecked():
            series = df_all["数字以外一致"].astype(str)
            uniq = set(s.strip() for s in series.fillna(""))
            num_allow_blank = ("" in uniq)
            allowed = [v for v in sorted(uniq) if v and v != "数字以外一致"]
            if num_allow_blank:
                allowed = [BLANK_TOKEN] + allowed
            num_allowed_list = allowed

        # 補助列：全文__concat（contains）
        if need_full:
            cols_for_full = [c for c in ["a", "b", "一致要因", "対象"] if c in df_all.columns]
            df_all["全文__concat"] = (
                df_all[cols_for_full].astype(str).fillna("").agg(" / ".join, axis=1)
            ) if cols_for_full else ""

        # ---------- 6) 初期表示マスク（True=表示） ----------
        mask = pd.Series(True, index=df_all.index)
        if scopes_on and "対象" in df_all.columns:
            mask &= df_all["対象"].astype(str).isin(scopes_on)
        if reasons_on and "一致要因" in df_all.columns:
            mask &= df_all["一致要因"].astype(str).isin(reasons_on)

        if edge_allowed_list is not None and "端差" in df_all.columns:
            vals = df_all["端差"].astype(str).fillna("")
            nonblank_set = set(v for v in edge_allowed_list if v != BLANK_TOKEN)
            cond_nonblank = vals.str.strip().isin(nonblank_set) if nonblank_set else pd.Series(False, index=vals.index)
            cond_blank    = vals.str.strip().eq("") if edge_allow_blank else pd.Series(False, index=vals.index)
            mask &= (cond_nonblank | cond_blank)

        if num_allowed_list is not None and "数字以外一致" in df_all.columns:
            vals = df_all["数字以外一致"].astype(str).fillna("")
            nonblank_set = set(v for v in num_allowed_list if v != BLANK_TOKEN)
            cond_nonblank = vals.str.strip().isin(nonblank_set) if nonblank_set else pd.Series(False, index=vals.index)
            cond_blank    = vals.str.strip().eq("") if num_allow_blank else pd.Series(False, index=vals.index)
            mask &= (cond_nonblank | cond_blank)

        if short_on and "字数" in df_all.columns:
            mask &= pd.to_numeric(df_all["字数"], errors="coerce").fillna(0) > short_n
        if need_full and "全文__concat" in df_all.columns:
            mask &= df_all["全文__concat"].astype(str).str.contains(fulltext, case=False, na=False)

        # ---------- 7) 書き出し ----------
        try:
            with pd.ExcelWriter(path, engine=engine) as writer:
                sheet_main = "表記ゆれ候補"
                df_all.to_excel(writer, index=False, sheet_name=sheet_main)

                nrows, ncols = df_all.shape

                # 列インデックス
                def col_idx(col_name):
                    try: return df_all.columns.get_loc(col_name)
                    except Exception: return None
                idx_len   = col_idx("字数")
                idx_reason= col_idx("一致要因")
                idx_scope = col_idx("対象")
                idx_edge  = col_idx("端差")
                idx_num   = col_idx("数字以外一致")
                idx_full  = col_idx("全文__concat") if need_full else None

                if engine == "xlsxwriter":
                    wb = writer.book
                    ws = writer.sheets[sheet_main]

                    # ❶ フリーズ＋オートフィルタ枠（全件）
                    ws.freeze_panes(1, 1)
                    ws.autofilter(0, 0, nrows, max(0, ncols - 1))

                    # ❷ フィルタ条件の宣言（Blanksは特別トークン）
                    if idx_scope is not None and scopes_on:
                        uniq_scopes = set(str(x) for x in df_all["対象"].dropna().astype(str))
                        if len(scopes_on) < len(uniq_scopes):
                            ws.filter_column_list(idx_scope, [str(v) for v in scopes_on])

                    if idx_reason is not None and reasons_on:
                        uniq_reasons = set(str(x) for x in df_all["一致要因"].dropna().astype(str))
                        if len(reasons_on) < len(uniq_reasons):
                            ws.filter_column_list(idx_reason, [str(v) for v in reasons_on])

                    if idx_edge is not None and edge_allowed_list is not None:
                        uniq_edge = set(str(x).strip() for x in df_all["端差"].fillna(""))
                        if 0 < len(edge_allowed_list) < len(uniq_edge) + (1 if edge_allow_blank else 0):
                            ws.filter_column_list(idx_edge, edge_allowed_list)

                    if idx_num is not None and num_allowed_list is not None:
                        uniq_num = set(str(x).strip() for x in df_all["数字以外一致"].fillna(""))
                        if 0 < len(num_allowed_list) < len(uniq_num) + (1 if num_allow_blank else 0):
                            ws.filter_column_list(idx_num, num_allowed_list)

                    if idx_len is not None and short_on:
                        ws.filter_column(idx_len, f'x > {short_n}')

                    if idx_full is not None and fulltext:
                        def _xf_escape(s: str) -> str:
                            return s.replace('~', '~~').replace('*', '~*').replace('?', '~?')
                        ws.filter_column(idx_full, f'x == *{_xf_escape(fulltext)}*')
                        ws.set_column(idx_full, idx_full, None, None, {"hidden": True})

                    # ❸ 初期表示で非該当行を非表示（ヘッダ0を除外）
                    for i, ok in enumerate(mask.tolist()):
                        if not ok:
                            ws.set_row(i + 1, None, None, {'hidden': True})

                    # ❹ 列幅
                    num_fmt = wb.add_format({"align": "right"})
                    def set_w(name, width, fmt=None):
                        ci = col_idx(name)
                        if ci is not None:
                            ws.set_column(ci, ci, width, fmt)
                    set_w("選択", 7)
                    set_w("gid", 7, num_fmt)
                    set_w("字数", 6, num_fmt)
                    set_w("a", 32); set_w("b", 32)
                    set_w("一致要因", 12); set_w("対象", 10)
                    set_w("端差", 10); set_w("数字以外一致", 12)

                else:
                    # openpyxl：枠＋B2固定＋非該当行の非表示
                    ws = writer.sheets[sheet_main]
                    try:
                        from openpyxl.utils import get_column_letter
                        last_row = max(1, nrows + 1)      # ヘッダ込み
                        last_col = max(1, ncols)
                        ws.auto_filter.ref = f"A1:{get_column_letter(last_col)}{last_row}"
                        ws.freeze_panes = "B2"
                        for i, ok in enumerate(mask.tolist(), start=2):
                            if not ok:
                                ws.row_dimensions[i].hidden = True
                    except Exception:
                        pass

            QMessageBox.information(self, "完了", "Excel（全件＋初期フィルタ）を出力しました。")
        except Exception as e:
            QMessageBox.critical(self, "エラー", str(e))

    def on_mark_pdf(self):
        if not self.files:
            QMessageBox.warning(self, "注意", "PDFを指定してください。"); return
        df = getattr(self.model_unified, "_df", pd.DataFrame())
        if df.empty:
            QMessageBox.information(self, "情報", "マーキング対象がありません。"); return

        col_a = df.columns.get_indexer(["a"])[0] if "a" in df.columns else -1
        col_b = df.columns.get_indexer(["b"])[0] if "b" in df.columns else -1

        raw_targets = []
        for r_proxy in range(self.proxy_unified.rowCount()):
            src_idx = self.proxy_unified.mapToSource(self.proxy_unified.index(r_proxy, 0))
            i = src_idx.row()
            if i < 0 or i >= len(self.model_unified._checks):
                continue
            if self.model_unified._checks[i]:
                if col_a >= 0:
                    a = df.iat[i, col_a]
                    if isinstance(a, str) and a:
                        raw_targets.append(("a", a))
                if col_b >= 0:
                    b = df.iat[i, col_b]
                    if isinstance(b, str) and b:
                        raw_targets.append(("b", b))
        if not raw_targets:
            QMessageBox.information(self, "情報", "マーキング対象が選択されていません。"); return

        by_text = {}
        for kind, text in raw_targets:
            if text not in by_text:
                by_text[text] = kind
            else:
                if by_text[text] == "b" and kind == "a":
                    by_text[text] = "a"
        items = [(kind, text) for text, kind in by_text.items()]
        items.sort(key=lambda kt: (kt[0], kt[1]))

        COLOR_YELLOW = (0.98, 0.90, 0.25)
        COLOR_CYAN   = (0.00, 0.90, 1.00)
        HIGHLIGHT_COLORS = {"a": COLOR_YELLOW, "b": COLOR_CYAN}

        out_dir = QFileDialog.getExistingDirectory(self, "出力フォルダを選択")
        if not out_dir:
            return
        CONN_CHARS = "ー-－–—・/／_＿"
        SPACE_CHARS = {" ", "\u00A0", "\u3000"}

        def kana_norm_variants(s: str) -> Set[str]:
            base = nfkc(s or ""); k = hira_to_kata(base)
            return {k, k.replace("ー", "")}

        def base_variants(s: str) -> Set[str]:
            vs = set(); s0 = s or ""; s1 = nfkc(s0)
            for cand in (s0, s1):
                if not cand: continue
                vs.add(cand)
                vs.add(cand.replace(" ", ""))
                vs.add(cand.replace(" ", "\u00A0"))
                vs.add(cand.replace(" ", "\u3000"))
            for c in list(vs):
                for ch in CONN_CHARS:
                    if ch in c:
                        vs.add(c.replace(ch, ""))
                        vs.add(c.replace(ch, " "))
                        vs.add(c.replace(ch, "\u00A0"))
            add = set()
            for c in vs:
                add |= kana_norm_variants(c)
            vs |= add
            return {v for v in vs if v}

        def norm_stream_chars(page):
            rd = page.get_text("rawdict")
            chars = []
            line_counter = 0
            used_equal_split = False
            blocks = rd.get("blocks", []) if isinstance(rd, dict) else []
            for b in blocks:
                for line in b.get("lines", []):
                    for span in line.get("spans", []):
                        if "chars" in span and isinstance(span["chars"], list) and span["chars"]:
                            for ch in span["chars"]:
                                c = ch.get("c", ""); bbox = ch.get("bbox", None)
                                if not c or not bbox: continue
                                r = fitz.Rect(bbox)
                                chars.append({"orig": c, "rect": r, "line": line_counter})
                        else:
                            text = span.get("text", ""); bbox = span.get("bbox", None)
                            if not text or not bbox: continue
                            used_equal_split = True
                            r = fitz.Rect(bbox); n = len(text)
                            if n <= 0: continue
                            x0, y0, x1, y1 = r
                            dx, dy = 1.0, 0.0
                            d = span.get("dir", None)
                            if isinstance(d, (list, tuple)) and len(d) >= 2:
                                try:
                                    dx, dy = float(d[0] or 0.0), float(d[1] or 0.0)
                                except Exception:
                                    dx, dy = 1.0, 0.0
                            is_vertical = abs(dy) > abs(dx)
                            if is_vertical:
                                ch_h = max(1e-3, (y1 - y0) / n)
                                for idx, c in enumerate(text):
                                    rr = fitz.Rect(x0, y0 + idx*ch_h, x1, y0 + (idx+1)*ch_h)
                                    chars.append({"orig": c, "rect": rr, "line": line_counter})
                            else:
                                ch_w = max(1e-3, (x1 - x0) / n)
                                for idx, c in enumerate(text):
                                    rr = fitz.Rect(x0 + idx*ch_w, y0, x0 + (idx+1)*ch_w, y1)
                                    chars.append({"orig": c, "rect": rr, "line": line_counter})
                    line_counter += 1

            def norm_keep(c: str) -> str:
                if c in SPACE_CHARS or c in CONN_CHARS: return ""
                c = nfkc(c); c = hira_to_kata(c)
                if c in SPACE_CHARS or c in CONN_CHARS: return ""
                return c
            def norm_nochoon(c: str) -> str:
                s = norm_keep(c)
                if s == "ー": return ""
                return s

            keep_list, keep_idxmap = [], []
            noch_list, noch_idxmap = [], []
            for i, it in enumerate(chars):
                c = it["orig"]
                nk = norm_keep(c)
                if nk: keep_list.append(nk); keep_idxmap.append(i)
                nn = norm_nochoon(c)
                if nn: noch_list.append(nn); noch_idxmap.append(i)
            stream_keep = "".join(keep_list)
            stream_noch = "".join(noch_list)
            return chars, (stream_keep, keep_idxmap), (stream_noch, noch_idxmap), used_equal_split

        def search_stream(stream_text: str, idxmap: list, qnorm: str):
            pos = 0; spans = []; Lq = len(qnorm)
            if not qnorm or not stream_text: return spans
            while True:
                k = stream_text.find(qnorm, pos)
                if k < 0: break
                s_idx = k; e_idx = k + Lq - 1
                spans.append((idxmap[s_idx], idxmap[e_idx]))
                pos = k + 1
            return spans

        def merge_to_line_rects(chars, idx_s: int, idx_e: int):
            items = chars[idx_s:idx_e+1]
            if not items: return []
            by_line = defaultdict(list)
            for it in items:
                by_line[it["line"]].append(it["rect"])
            rects = []
            for ln in sorted(by_line.keys()):
                rs = by_line[ln]
                x0 = min(r.x0 for r in rs); y0 = min(r.y0 for r in rs)
                x1 = max(r.x1 for r in rs); y1 = max(r.y1 for r in rs)
                rects.append(fitz.Rect(x0, y0, x1, y1))
            return rects

        def dedup(rects):
            uniq, seen = [], set()
            for rr in rects:
                key = (round(rr.x0, 2), round(rr.y0, 2), round(rr.x1, 2), round(rr.y1, 2))
                if key not in seen:
                    seen.add(key); uniq.append(rr)
            return uniq

        def robust_find_on_page(page, queries: list):
            chars, (stream_keep, keep_idxmap), (stream_noch, noch_idxmap), used_equal_split = norm_stream_chars(page)

            qnorm_keep_set, qnorm_noch_set = set(), set()
            for q in queries:
                for v in base_variants(q):
                    v_keep = hira_to_kata(nfkc(v))
                    v_keep = "".join(ch for ch in v_keep if ch not in SPACE_CHARS and ch not in CONN_CHARS)
                    if v_keep: qnorm_keep_set.add(v_keep)
                    v_noch = v_keep.replace("ー", "")
                    if v_noch: qnorm_noch_set.add(v_noch)

            stream_hits = []
            for qk in qnorm_keep_set:
                for s_idx, e_idx in search_stream(stream_keep, keep_idxmap, qk):
                    stream_hits.extend(merge_to_line_rects(chars, s_idx, e_idx))
            for qn in qnorm_noch_set:
                for s_idx, e_idx in search_stream(stream_noch, noch_idxmap, qn):
                    stream_hits.extend(merge_to_line_rects(chars, s_idx, e_idx))
            stream_hits = dedup(stream_hits)

            search_hits = []
            vset = set()
            for q in queries:
                vset |= base_variants(q)
            for cand in vset:
                try:
                    search_hits += page.search_for(cand)
                except Exception:
                    pass
            search_hits = dedup(search_hits)

            rect_hits = search_hits if search_hits else stream_hits
            return dedup(rect_hits)

        def _suppress_overlap_hits(hits, iou_th=0.60):
            def area(r): return max(0.0, (r.x1 - r.x0) * (r.y1 - r.y0))
            def inter(r1, r2):
                x0 = max(r1.x0, r2.x0); y0 = max(r1.y0, r2.y0)
                x1 = min(r1.x1, r2.x1); y1 = min(r1.y1, r2.y1)
                if x1 <= x0 or y1 <= y0: return 0.0
                return (x1 - x0) * (y1 - y0)
            kept = []
            hits_sorted = sorted(hits, key=lambda r: (-area(r), r.y0, r.x0))
            for r in hits_sorted:
                drop = False
                for rk in kept:
                    inter_a = inter(r, rk)
                    if inter_a <= 0: continue
                    iou = inter_a / (area(r) + area(rk) - inter_a + 1e-6)
                    if iou >= iou_th: drop = True; break
                    if (r.x0 >= rk.x0 - 0.5 and r.y0 >= rk.y0 - 0.5 and
                        r.x1 <= rk.x1 + 0.5 and r.y1 <= rk.y1 + 0.5):
                        drop = True; break
                if not drop:
                    kept.append(r)
            return kept

        try:
            for src in self.files:
                doc = fitz.open(src)
                for page in doc:
                    page_rects = []
                    for v_idx, (kind, s) in enumerate(items):
                        label = chr(ord('a') + (v_idx % 26))
                        gid = self.surf2gid.get(s, 0)
                        rects = robust_find_on_page(page, [s])
                        rects = _suppress_overlap_hits(rects)
                        for r in rects:
                            page_rects.append((r, s, gid, label, kind))
                    for r, text, gid, var, kind in page_rects:
                        ann = page.add_highlight_annot(r)
                        ann.set_info({"title": f"{gid}-{var}", "subject": f"#{gid}", "content": text})
                        color = HIGHLIGHT_COLORS.get(kind, COLOR_YELLOW)
                        ann.set_colors(stroke=color)
                        ann.update()
                base = os.path.splitext(os.path.basename(src))[0]
                out_path = os.path.join(out_dir, f"{base}_marked.pdf")
                doc.save(out_path); doc.close()
            QMessageBox.information(self, "完了", "PDFへマーキングしました。")
        except Exception as e:
            QMessageBox.critical(self, "エラー", str(e))

    def on_text_filter_changed(self, text: str):
        self.proxy_unified.setTextFilter(text)
        self.proxy_tokens.setFilterFixedString(text)
        self._refresh_candidate_count()
    def on_shortlen_filter_changed(self, *_):
        """短文（aの文字数）フィルタのUI変更ハンドラ"""
        self.proxy_unified.setHideShortEnabled(self.chk_shortlen.isChecked())
        self.proxy_unified.setHideShortLength(self.sb_shortlen.value())
        self._refresh_candidate_count()

    def on_reason_changed(self, *_):
        self.proxy_unified.setReasonFilter(self._selected_reasons())
        self._refresh_candidate_count()
    def on_scope_changed(self, *_):
        self.proxy_unified.setScopeFilter(self._selected_scopes())
        self._refresh_candidate_count()
    def on_edge_filter_changed(self, *_):
        """端差（前/後1字）＋ 数字以外一致 のチェックボックス変更ハンドラ"""
        self.proxy_unified.setHidePrefixEdge(self.chk_edge_prefix.isChecked())
        self.proxy_unified.setHideSuffixEdge(self.chk_edge_suffix.isChecked())
        self.proxy_unified.setHideNumericOnly(self.chk_num_only.isChecked())  # ★追加
        self._refresh_candidate_count()
    def _selected_reasons(self) -> Set[str]:
        return {r for r, cb in self.chk_reasons.items() if cb.isChecked()}
    def _selected_scopes(self) -> Set[str]:
        return {s for s, cb in self.chk_scopes.items() if cb.isChecked()}

    # 追加：'端差' と '数字以外一致' の列をビュー上だけ非表示
    def _hide_aux_columns(self) -> None:
        """QTableView（表記ゆれ候補）から『端差』『数字以外一致』列を非表示にします。"""
        view = self.view_unified
        model = view.model()  # Proxy（UnifiedFilterProxy）
        if not model:
            return
        for name in ("端差", "数字以外一致"):
            for c in range(model.columnCount()):
                h = model.headerData(c, Qt.Horizontal, Qt.DisplayRole)
                if isinstance(h, str) and h.strip() == name:
                    view.setColumnHidden(c, True)
                    break


# ヘッダ名から「ビュー上の列インデックス」を探す
    def _find_view_column(self, header_name: str) -> int:
        model = self.view_unified.model()  # Proxy（UnifiedFilterProxy）
        if not model:
            return -1
        for c in range(model.columnCount()):
            h = model.headerData(c, Qt.Horizontal, Qt.DisplayRole)
            if isinstance(h, str) and h.strip() == header_name:
                return c
        return -1

    # a/b 列の幅を min～max でクリップ（自動調整の“あと”に呼ぶ）
    def _cap_ab_widths(self, min_a=220, max_a=420, min_b=220, max_b=420):
        view = self.view_unified
        hdr  = view.horizontalHeader()
        idx_a = self._find_view_column("a")
        idx_b = self._find_view_column("b")
        for idx, min_w, max_w in ((idx_a, min_a, max_a), (idx_b, min_b, max_b)):
            if idx >= 0:
                w = view.columnWidth(idx)
                w = max(min_w, min(w, max_w))
                view.setColumnWidth(idx, w)
                # 以後は自動リサイズではなく“手動ベース”にしておく
                hdr.setSectionResizeMode(idx, QHeaderView.Interactive)

    def _build_filtered_unified_df(self) -> pd.DataFrame:
        """現在のフィルタ結果（表示中の行のみ）をDataFrameとして返す。先頭に「選択」列を付与。"""
        src_df = getattr(self.model_unified, "_df", pd.DataFrame())
        if src_df.empty:
            # 「選択」列だけを持つ空表（ヘッダ付きでExcelに出せる）
            return pd.DataFrame(columns=["選択"] + list(src_df.columns))

        rows_src = []
        sel_flags = []
        for r_proxy in range(self.proxy_unified.rowCount()):
            sidx = self.proxy_unified.mapToSource(self.proxy_unified.index(r_proxy, 0))
            i = sidx.row()
            if i < 0 or i >= len(self.model_unified._checks):
                continue
            rows_src.append(i)
            sel_flags.append(bool(self.model_unified._checks[i]))

        if not rows_src:
            return pd.DataFrame(columns=["選択"] + list(src_df.columns))

        out = src_df.iloc[rows_src].copy().reset_index(drop=True)
        out.insert(0, "選択", sel_flags)

        # 列の並びを軽く整える（見やすさ重視）
        cols = list(out.columns)
        prefer = [c for c in ["選択", "gid", "字数", "a", "b", "一致要因", "対象", "端差", "数字以外一致"] if c in cols]
        rest = [c for c in cols if c not in prefer]
        out = out[prefer + rest]
        return out

    def _current_filter_state(self) -> dict:
        """GUIのフィルタ状態を辞書で返す（Excelの「フィルタ設定」シート用）。"""
        return {
            "全文": self.ed_filter.text() or "",
            "対象（表示）": "、".join(sorted(self._selected_scopes())),
            "一致要因（表示）": "、".join(sorted(self._selected_reasons())),
            "前1文字差を非表示": "ON" if self.chk_edge_prefix.isChecked() else "OFF",
            "後1文字差を非表示": "ON" if self.chk_edge_suffix.isChecked() else "OFF",
            "数字以外一致を非表示": "ON" if self.chk_num_only.isChecked() else "OFF",
            "字数フィルタ 有効": "ON" if self.chk_shortlen.isChecked() else "OFF",
            "字数フィルタ（n以下を隠す）": self.sb_shortlen.value() if self.chk_shortlen.isChecked() else "",
            "読み 類似しきい値": self.dsb_read.value(),
            "文字 類似しきい値": self.dsb_char.value(),
            "解析ファイル数": len(self.files),
            "解析ファイル一覧": "\n".join(self.files),
            "エクスポート日時": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _build_unified_df_with_selection_all(self) -> pd.DataFrame:
        """
        フィルタ前の全件（model_unified._df 全行）に「選択」列を付けて返す。
        Excel出力で“データは全件、見た目はフィルタ適用”を実現するための基礎DF。
        """
        src_df = getattr(self.model_unified, "_df", pd.DataFrame())
        if src_df.empty:
            return pd.DataFrame(columns=["選択"] + list(src_df.columns))

        # 現在のチェック状態を先頭列に
        checks = list(getattr(self.model_unified, "_checks", []))
        if len(checks) != len(src_df):
            # サイズが違う（何かの操作直後等）は全Falseで揃える
            checks = [False] * len(src_df)

        out = src_df.copy().reset_index(drop=True)
        out.insert(0, "選択", checks)

        # 補助：'字数' 列が無ければ a の長さから作る（Excelの数値フィルタ用）
        if "字数" not in out.columns and "a" in out.columns:
            out["字数"] = out["a"].map(lambda x: len(str(x)) if x is not None else 0).astype(int)

        return out

    def _sanitize_df_for_excel(self, df: pd.DataFrame) -> pd.DataFrame:
        """Excel出力前の安全化:
        - dict / list / tuple / set は JSON 文字列化
        - None/NaN は空文字
        - それ以外はそのまま（数値は数値のまま）
        """
        import json
        import math
        def coerce(v):
            # NaN も空へ
            if v is None:
                return ""
            if isinstance(v, float) and math.isnan(v):
                return ""
            # コンテナ系はJSON化（ensure_ascii=Falseで日本語もOK）
            if isinstance(v, (dict, list, tuple, set)):
                try:
                    return json.dumps(v, ensure_ascii=False)
                except Exception:
                    return str(v)
            return v

        for col in df.columns:
            # object列のみmap（数値列は触らない）
            if df[col].dtype == "object":
                df[col] = df[col].map(coerce)
        return df

def main():
    app = QApplication(sys.argv)
    apply_light_theme(app, base_font_size=10)
    win = MainWindow(); win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()  # ★ 追加：凍結exeでの子プロセス起動対策
    main()

