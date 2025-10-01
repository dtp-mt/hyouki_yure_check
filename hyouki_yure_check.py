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
    QTimer, QCollator, QPoint, QSize   # ← これを追加
)

from PySide6.QtGui import (
    QFont, QPalette, QColor, QPainter, QPen, QAction
)
# ===== [PATCH] import QSizePolicy =====
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTableView, QTabWidget, QLineEdit, QDoubleSpinBox, QSpinBox,
    QGroupBox, QFormLayout, QMessageBox, QTextEdit, QTextBrowser, QProgressBar, QAbstractItemView,
    QListWidget, QCheckBox, QStyledItemDelegate, QStyle, QProxyStyle,
    QMenu, QDialog, QDialogButtonBox, QFrame, QHeaderView, QSizePolicy
)
# ===== [/PATCH] ======================

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

# ==== 読み類似スコア（数・記号は読み対象外／満点抑制の強化版） ==================
KANJI_RE = re.compile(r"[\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF]")

def _has_kanji(s: str) -> bool:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = nfkc(s)
    return bool(KANJI_RE.search(s))

def _norm_len_for_surface(s: str) -> int:
    """表層長（比較用）。NFKC→空白除去。"""
    if s is None:
        s = ""
    s = nfkc(str(s))
    s = re.sub(r"\s+", "", s)
    return len(s)

# ===== [Patch] 読み一致（reading_eq）のスコアを 1.0 固定にしない軽量スコアラー =====
def _numeric_chunk_count(s: str) -> int:
    """
    NFKC 後の文字列に _NUMERIC_CHUNK_RE を当てて、数値“まとまり”の個数を数える。
    例: '630W[30°C]' -> 2（630, 30）
    """
    try:
        nf = unicodedata.normalize("NFKC", "" if s is None else str(s))
    except Exception:
        nf = "" if s is None else str(s)
    return sum(1 for _ in _NUMERIC_CHUNK_RE.finditer(nf))

def reading_eq_score(a: str, b: str) -> float:
    """
    'reading_eq' に分類されたペアの “類似度” を 1.0 固定にせず、
    a/b の差に応じた小さな減点を与えて 0.90〜0.995 の範囲に収める。

    減点規則（初期チューニング）：
      - P0: a != b                   …… -0.01
      - P1: 表層長差（NFKC/空白除去）…… -min(0.05, 0.01 * |len_a - len_b|)
      - P2: 数値チャンク個数差             …… -min(0.02, 0.005 * |numA - numB|)
      - P3: 片側だけ漢字を含む                …… -0.01

    最終クランプ：0.90 <= score <= 0.995
    """
    a_s = "" if a is None else str(a)
    b_s = "" if b is None else str(b)

    score = 1.0

    # P0: 表層が同一でない（reading_eq は基本 a!=b の想定）
    if a_s != b_s:
        score -= 0.01

    # P1: NFKC + 空白除去の長さ差
    try:
        la = _norm_len_for_surface(a_s)  # 既存：nfkc -> 空白除去 -> len
        lb = _norm_len_for_surface(b_s)
        diff_len = abs(la - lb)
        if diff_len > 0:
            score -= min(0.05, 0.01 * diff_len)
    except Exception:
        pass

    # P2: 数値“まとまり”個数差
    try:
        na = _numeric_chunk_count(a_s)
        nb = _numeric_chunk_count(b_s)
        diff_num = abs(na - nb)
        if diff_num > 0:
            score -= min(0.02, 0.005 * diff_num)
    except Exception:
        pass

    # P3: 片方だけ漢字を含む
    try:
        if _has_kanji(a_s) != _has_kanji(b_s):
            score -= 0.01
    except Exception:
        pass

    # 取りすぎない・取りなさすぎない
    score = max(0.90, min(0.995, score))

    # UI 側の丸めと整合（内部でも丸めておく）
    return round(float(score), 3)
# ===== [Patch end] ============================================================

def _temp_marker_core_after_number_strip(s: str) -> str:
    """
    NFKC → 数字まとまり除去（既存の _strip_numbers_like）後、
    温度記号に関係する最小核だけを取り出す。
    - '°' と 'C/c' がともに残っていれば '°C' を返す
    - それ以外は空文字
    """
    if s is None:
        return ""
    # 既存の数字まとまり除去（NFKC 内部で実施される）
    t = _strip_numbers_like(s)  # 例: "630W[30°C]" -> "W[°C]"
    if not t:
        return ""
    # 温度記号に関係ない文字は落とし、'°' と 'C/c' だけ残す
    keep = []
    for ch in t:
        if ch == "°" or ch in ("C", "c"):
            keep.append(ch)
    has_deg = "°" in keep
    has_c   = ("C" in keep) or ("c" in keep)
    return "°C" if has_deg and has_c else ""

def is_symbol_only_surface_diff(a: str, b: str) -> bool:
    """
    表層 a, b のうち、ひらがな/カタカナ/漢字だけを取り出した “和字コア” が
    一致するかどうか（= 記号・英数・単位の差だけか）を判定する。

    変更点（℃専用の救済）：
      - 和字コアが両方 空 の場合に限り、
        「数字を除いた後」に温度記号の最小核（° + C/c）が
        双方で満たされる（= '°C'）なら True を返す。
      - 単位一般は扱わず、あくまで「℃」相当のケースに限定。

    これにより、以下が True（= reading_eq, score=1.0 対象）になります：
      - "20℃" ↔ "30℃"
      - "30℃" ↔ "630W［30℃"
    反対に、和字コアが非対称（例: "℃" ↔ "たび"）は False のまま。
    """
    ca = _surface_core_for_reading(a)
    cb = _surface_core_for_reading(b)

    # 既存ルール：和字コアが非空で一致 → True
    if ca and cb:
        return ca == cb

    # 追加ルール（℃専用）：
    # 両方とも和字コアが空 かつ 「数字を除いた後」の温度核が双方 '°C'
    if not ca and not cb:
        ta = _temp_marker_core_after_number_strip(a)
        tb = _temp_marker_core_after_number_strip(b)
        if ta and tb and (ta == tb == "°C"):
            return True

    return False

def reading_sim_with_penalty(a_surface: str, b_surface: str, ra: str, rb: str, th: float) -> Optional[float]:
    """
    読み同士の類似度（帯付きLevenshtein）を返すが、以下の調整を行う：

    [A] “和字コア”が一致する（= 数字・記号・空白など読み対象外だけが違う）場合は
        読み類似を強制で 1.0（満点）にする。
        例: 「000円」vs「250円」→ コアはどちらも「円」 ⇒ 1.0

    [B] それ以外で満点(≈1.0)になった場合は、満点抑制を適用：
        [P0] 表層が異なるだけで 1.0 は避ける       → -0.01
        [P1] 片方だけ漢字を含む（かな/漢字表記差）→ さらに -0.01（計 -0.02）
        [P2] 表層長が異なる（読み対象文字数差）   → さらに -0.05

    例：
      「電気」vs「電機」  → P0: 1.00→0.99
      「とき」vs「冬季」  → P0+P1: 1.00→0.98
      「離し」vs「話」    → P0+P2: 1.00→0.94
    """
    ra = "" if ra is None else str(ra)
    rb = "" if rb is None else str(rb)

    # --- [A] 数字・記号・空白などを除いた“和字コア”が一致するなら、読み=1.0 扱い ---
    core_a = _surface_core_for_reading(a_surface)
    core_b = _surface_core_for_reading(b_surface)
    if core_a and core_b and core_a == core_b:
        # 読み生成（ra/rb）が数字読みなどで異なっても、読み対象はコアのみとみなす
        return 1.0

    # --- 読みが両方空なら、読み類似は使わない ---
    if len(ra) == 0 and len(rb) == 0:
        return None

    sim = sim_with_threshold(ra, rb, th)
    if sim is None:
        return None

    # --- [B] 満点(≈1.0)のときだけ抑制ロジックを適用 ---
    if sim >= 0.9999:
        penalized = 1.0

        # [P0] 表層差があるなら -0.01
        if str(a_surface) != str(b_surface):
            penalized -= 0.01

        # [P1] 片方だけ漢字を含むなら -0.01（合計 -0.02）
        if _has_kanji(a_surface) != _has_kanji(b_surface):
            penalized -= 0.01

        # [P2] 表層長（NFKC/空白無視）が異なるなら -0.05
        if _norm_len_for_surface(a_surface) != _norm_len_for_surface(b_surface):
            penalized -= 0.05

        sim = penalized

    return sim

# ==== 読み類似のスコア再採点（lemma/活用も“読み”基準で減点 + P4 追加） ============
READING_LIKE_REASONS = {"lemma", "lemma_read", "inflect", "reading"}

# 追加：読み不一致の追加減点（調整しやすいように定数化）
READ_MISMATCH_PENALTY = 0.03  # ←「0.99より下げたい」ニュアンスに最適化

def _reading_for_surface_cached(s: str) -> str:
    """表層 s の“読み（骨格）”。MeCab があればそれ、無ければ簡易フォールバック。"""
    ok, _ = ensure_mecab()
    if ok:
        return phrase_reading_norm_cached(s or "")
    # フォールバック（NFKC→ひら->カナ→カナ以外除去→長音除去）
    return normalize_kana(hira_to_kata(nfkc(s or "")), drop_choon=True)

def _lemma_joined_for_surface(s: str) -> str:
    """
    表層 s の lemma を連結して返す（活用差の検出に使う）。
    例: 「あわせ」「合わせる」→ ともに '合わせる'（期待）
    """
    ok, _ = ensure_mecab()
    if not ok or MECAB_TAGGER is None:
        # MeCab 無しなら NFKC 表層を代用（厳密でなくてOK）
        return nfkc(s or "")
    toks = tokenize_mecab(s or "", MECAB_TAGGER)
    # lemma が無ければ surface を採用
    return "".join((lemma if (lemma and isinstance(lemma, str)) else surf)
                   for (surf, pos1, lemma, reading, ct, cf) in toks)

def _surface_core_for_reading(s: str) -> str:
    """読み対象の“和字コア”（漢字・ひらがな・カタカナのみ）を抽出。"""
    nf = nfkc(str(s or ""))
    out = []
    for ch in nf:
        o = ord(ch)
        if (0x3040 <= o <= 0x309F) or (0x30A0 <= o <= 0x30FF) or \
           (0x3400 <= o <= 0x4DBF) or (0x4E00 <= o <= 0x9FFF) or \
           (0xF900 <= o <= 0xFAFF):
            out.append(ch)
    return "".join(out)

# ===== 追加: 囲み数字などの「リストマーカー」を判定 =====
def _is_enclosed_list_marker(ch: str) -> bool:
    """
    '①' '②' ... '㉑' や '❶' '❷' ... などの囲み数字系は
    「リストマーカー」とみなし、ノイズ扱いしないための判定。
    """
    if not ch:
        return False
    o = ord(ch)
    # Enclosed Alphanumerics (一部): ⓪(0x24EA), ①..⑳(0x2460..0x2473), ⑴..⒇ 等
    if 0x2460 <= o <= 0x24FF or o == 0x24EA:
        return True
    # Dingbats (Negative circled digits 等): ❶..➓(0x2776..0x2793)
    if 0x2776 <= o <= 0x2793:
        return True
    # Enclosed Alphanumeric Supplement (拡張の囲み文字): U+1F100..U+1F1FF
    if 0x1F100 <= o <= 0x1F1FF:
        return True

# ===== 置換: 非日本語(英字・数字・記号)ノイズの有無を判定 =====
def _has_non_japanese_noise(s: str) -> bool:
    """
    '英数字や記号によるノイズ' が含まれるか。
    - 日本語スクリプト(漢字/ひら/カタカナ)と空白は無視
    - 囲み数字などの「リストマーカー」(①, ❶ など)はノイズに数えない
    - それ以外の ASCII/全角英字・数字・各種記号はノイズとみなす
    """
    if s is None:
        return False
    # 原字のまま評価（NFKCにすると ①→"1" に潰れてしまうため）
    for ch in str(s):
        if not ch or ch.isspace():
            continue
        o = ord(ch)
        # 日本語の主要スクリプト → ノイズではない
        if (
            0x3040 <= o <= 0x309F   # ひらがな
            or 0x30A0 <= o <= 0x30FF  # カタカナ
            or 0x3400 <= o <= 0x4DBF  # CJK統合漢字拡張A
            or 0x4E00 <= o <= 0x9FFF  # CJK統合漢字
            or 0xF900 <= o <= 0xFAFF  # 互換漢字
        ):
            continue
        # 囲み数字などの「リストマーカー」 → ノイズに数えない
        if _is_enclosed_list_marker(ch):
            continue
        # ここからはノイズ候補
        # ・英数字（ASCII/全角問わず）
        # ・各種記号（Unicode Category 'S*'）や句読点等（'P*'）
        cat = unicodedata.category(ch)  # 'Nd' 数字, 'Ll' 小文字, 'Lu' 大文字, 'S*' 記号, 'P*' 句読点など
        if ch.isdigit() or ('A' <= ch <= 'Z') or ('a' <= ch <= 'z'):
            return True
        if cat.startswith('S') or cat.startswith('P'):
            return True
        # 全角英字（NFKCせずともカテゴリで拾えない場合があるため補助）
        ch_nfkc = nfkc(ch)
        if ch_nfkc != ch:
            if ch_nfkc.isdigit() or ('A' <= ch_nfkc <= 'Z') or ('a' <= ch_nfkc <= 'z'):
                return True
            cat2 = unicodedata.category(ch_nfkc)
            if cat2.startswith('S') or cat2.startswith('P'):
                return True


def _script_pattern(s: str) -> str:
    """
    文字ごとのスクリプト種別列（K:漢字/H:ひら/C:カナ/O:その他）を返す。
    スペースは除外。NFKC 後のコードポイントに基づく。
    """
    def tag(ch: str) -> str:
        o = ord(ch)
        if 0x3400 <= o <= 0x4DBF or 0x4E00 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF:
            return "K"
        if 0x3040 <= o <= 0x309F:
            return "H"
        if 0x30A0 <= o <= 0x30FF:
            return "C"
        return "O"
    nf = nfkc(str(s or ""))
    return "".join(tag(ch) for ch in nf if not ch.isspace())

# --- REPLACE: recalibrate_reading_like_scores (重複ヘルパ排除) ---
def recalibrate_reading_like_scores(df: pd.DataFrame, read_th: float) -> pd.DataFrame:
    """
    {lemma, lemma_read, inflect, reading} を読みルールで再採点し reason='reading' に揃える。
    和字コア一致は強制1.0維持。満点時の抑制・P3/P4は現状ロジックを踏襲。
    """
    if df is None or df.empty:
        return df
    need_cols = {"a", "b", "reason", "score"}
    if not need_cols.issubset(df.columns):
        return df

    mask = df["reason"].isin(READING_LIKE_REASONS)
    if not mask.any():
        return df

    df = df.copy()

    # ---- 前計算（surface単位）
    surfaces = set(df.loc[mask, "a"].astype(str)) | set(df.loc[mask, "b"].astype(str))
    surf2read = {s: _reading_for_surface_cached(s) for s in surfaces}
    surf2lemma = {s: _lemma_joined_for_surface(s) for s in surfaces}
    surf2core  = {s: _surface_core_for_reading(s) for s in surfaces}
    surf2script = {s: _script_pattern(s) for s in surfaces}

    def _calc(row):
        a = str(row["a"]); b = str(row["b"])

        # 1) 和字コア一致 → 読み 1.0 維持
        ca, cb = surf2core.get(a, ""), surf2core.get(b, "")
        if ca and cb and ca == cb:
            return 1.0

        ra, rb = surf2read.get(a, ""), surf2read.get(b, "")
        pattern_equal = (surf2script.get(a, "") == surf2script.get(b, ""))
        read_mismatch = (bool(ra) and bool(rb) and ra != rb)

        # 2) 読み類似（満点抑制は reading_sim_with_penalty 内で処理）
        sim = reading_sim_with_penalty(a, b, ra, rb, read_th)
        if sim is not None:
            val = float(sim)
            # P3: lemma一致 & 表層差
            if val >= 0.9999 and a != b and (surf2lemma.get(a, "") == surf2lemma.get(b, "")):
                val -= 0.02
            # P4: 同スクリプト構成 & 読みが異なる
            if pattern_equal and read_mismatch:
                val -= READ_MISMATCH_PENALTY
            return max(val, 0.0)

        # 3) 読みが取れなかった場合でも、満点のままにしない（P0/P1/P2/P3/P4 の合算）
        penal = 1.0
        if a != b: penal -= 0.01                                        # P0
        if _has_kanji(a) != _has_kanji(b): penal -= 0.01                 # P1（グローバルの _has_kanji を利用）
        if _norm_len_for_surface(a) != _norm_len_for_surface(b): penal -= 0.05  # P2（グローバル利用）
        if a != b and (surf2lemma.get(a, "") == surf2lemma.get(b, "")): penal -= 0.02  # P3
        if pattern_equal and read_mismatch: penal -= READ_MISMATCH_PENALTY      # P4
        return max(penal, 0.0)

    df.loc[mask, "score"] = df.loc[mask].apply(_calc, axis=1)
    df.loc[mask, "reason"] = "reading"
    return df
# --- /REPLACE ---

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

# === 最終スコアを combined_char_similarity で強制（score_reasonは作らない既定） ===
def enforce_combined_similarity_score(
    df: pd.DataFrame,
    keep_backup: bool = False,   # 既定で退避しない
    drop_existing_backup: bool = True,  # 既にある 'score_reason' は落とす
) -> pd.DataFrame:
    """
    最終表示用に score を combined_char_similarity に差し替える。
    keep_backup=True のときだけ元スコアを score_reason に退避（.3f）。
    drop_existing_backup=True のとき、既にある score_reason を削除。
    """
    if df is None or df.empty:
        return df
    if not {"a", "b"}.issubset(df.columns):
        return df

    df = df.copy()

    # バックアップ列の扱い
    if drop_existing_backup:
        df.drop(columns=["score_reason"], errors="ignore", inplace=True)
    if keep_backup and "score" in df.columns and "score_reason" not in df.columns:
        df["score_reason"] = pd.to_numeric(df["score"], errors="coerce").round(3)

    # 表層キャッシュ（NFKC と 2-gram）
    try:
        uniq = pd.unique(pd.concat([df["a"], df["b"]]).astype("string"))
    except Exception:
        uniq = pd.unique(pd.concat([df["a"].astype(str), df["b"].astype(str)]))

    nf = {}
    grams = {}
    for s in uniq:
        s = "" if pd.isna(s) else str(s)
        ns = nfkc(s)
        nf[s] = ns
        grams[s] = frozenset(_ngram_set(ns, 2))

    def _calc(row):
        a = "" if pd.isna(row["a"]) else str(row["a"])
        b = "" if pd.isna(row["b"]) else str(row["b"])
        try:
            val = combined_char_similarity(
                a, b,
                sa=nf.get(a), sb=nf.get(b),
                grams_a=grams.get(a), grams_b=grams.get(b),
            )
        except Exception:
            # フォールバック：最悪でも 0.0〜1.0
            try:
                val = max(0.0, min(1.0, float(levenshtein_sim(nfkc(a), nfkc(b)))))
            except Exception:
                val = 0.0
        return round(float(val), 3)

    df["score"] = df.apply(_calc, axis=1)
    return df
# =====================================================================

# === 読み一致( reading_eq )で最終スコア1.0のものを 'basic' に付け替え ==========
def reclassify_basic_for_reading_eq(df: pd.DataFrame, eps: float = 0.0005) -> pd.DataFrame:
    """
    df に対し、reason=='reading_eq' かつ score が 1.0（丸め誤差±eps）を 'basic' へ変更。
    """
    if df is None or df.empty or "reason" not in df.columns or "score" not in df.columns:
        return df
    df = df.copy()
    # 丸め誤差込みで 1.0 判定（例: 0.9996 以上は 1.000 とみなす）
    m = (df["reason"] == "reading_eq") & (pd.to_numeric(df["score"], errors="coerce").fillna(0.0) >= 1.0 - eps)
    if m.any():
        df.loc[m, "reason"] = "basic"
        df.loc[m, "score"] = 1.0  # 表示上も 1.0 にそろえる
    return df
# =============================================================================

# =============================================================================
# 読み一致の厳密判定（前置/後置による包含を除外）
def _readings_equal_strict(a: str, b: str) -> bool:
    """
    ① 読み（長音保持）が完全一致（非空）であること
    ② ただし、和字コア（漢字・ひら・カナのみ）を比較して、
       一方がもう一方を strict な prefix/suffix として「包含」している場合は除外する
       （= 前置/後置で語が付加された可能性が高いケースを除く）
    例：還気温度センサ vs 温度センサ
        読み: （欠落により）どちらも 'オンドセンサ' になり得るが、
        和字コアは '還気温度センサ' と '温度センサ' で「後方一致」（suffix包含）→ NG
    """
    a_s = "" if a is None else str(a)
    b_s = "" if b is None else str(b)

    # 読み（長音保持）
    try:
        ra = phrase_reading_norm_keepchoon(a_s) or ""
        rb = phrase_reading_norm_keepchoon(b_s) or ""
    except Exception:
        ra, rb = "", ""

    if not (ra and rb):
        return False
    if ra != rb:
        return False

    # 和字コア（漢字・ひら・カナのみ）
    try:
        ca = _surface_core_for_reading(a_s) or ""
        cb = _surface_core_for_reading(b_s) or ""
    except Exception:
        ca, cb = "", ""

    if ca and cb and ca != cb:
        # strict な prefix/suffix 包含か？（完全一致は除外済み）
        if ca.endswith(cb) or cb.endswith(ca) or ca.startswith(cb) or cb.startswith(ca):
            # 前置/後置の付加（= 語の増減）とみなし、厳密一致から除外
            return False

    return True
# =============================================================================

# =============================================================================
# === [REPLACE] 読み一致（表記違い）: 厳格読み一致のみ昇格
#      ※ ただし「英数・記号ノイズ」が片側に含まれる場合は reading_eq を優先
# =============================================================================
def reclassify_reading_equal_formdiff(df: pd.DataFrame) -> pd.DataFrame:
    """
    'reading' の行から、以下の優先規則で再分類します：

    [優先1] 厳格読み一致(_readings_equal_strict) かつ
            片側にでも英数・記号ノイズ(_has_non_japanese_noise)が含まれる
            → reason='reading_eq', score=reading_eq_score(a,b)

    [優先2] それ以外で厳格読み一致(_readings_equal_strict) かつ a!=b
            → reason='reading_same'（従来昇格）

    [優先3] レマ一致 & 「名詞のみ」&「単一形態素」& 厳格読み一致
            → reason='reading_same'（従来昇格）

    なお、英数・記号だけの差は 'reading_eq' 領域（is_symbol_only_surface_diff True）
    とみなし本関数では扱いません（従来どおり）。
    """
    if (
        df is None or df.empty
        or "reason" not in df.columns
        or "a" not in df.columns or "b" not in df.columns
    ):
        return df

    # まず 'reading' 行のみ対象
    mask = df["reason"].eq("reading")
    if not mask.any():
        return df

    df = df.copy()

    # 表層キャッシュ
    sub = df.loc[mask, ["a", "b"]].astype("string")
    surfaces = pd.unique(pd.concat([sub["a"], sub["b"]], ignore_index=True))

    # 読み（長音保持）と レマ のキャッシュ
    surf2read_keep: Dict[str, str] = {}
    surf2lemma: Dict[str, str] = {}
    for s in surfaces:
        s_str = str(s)
        try:
            surf2read_keep[s_str] = phrase_reading_norm_keepchoon(s_str) or ""
        except Exception:
            surf2read_keep[s_str] = ""
        try:
            surf2lemma[s_str] = _lemma_joined_for_surface(s_str) or ""
        except Exception:
            surf2lemma[s_str] = ""

    # MeCab ユーティリティ
    ok_mecab, _ = ensure_mecab()
    tagger = MECAB_TAGGER if ok_mecab else None

    def _tokens(s: str):
        if not tagger:
            return []
        try:
            return tokenize_mecab(s, tagger)
        except Exception:
            return []

    def _is_noun_only(s: str) -> bool:
        toks = _tokens(s)
        if not toks:
            return False
        for _, pos1, *_ in toks:
            if not (isinstance(pos1, str) and pos1.startswith("名詞")):
                return False
        return True

    def _is_single_morpheme(s: str) -> bool:
        toks = _tokens(s)
        return len(toks) == 1

    # ここから再分類
    idx = df.index[mask]

    # 行ごとに適用
    for i in idx:
        a = "" if pd.isna(df.at[i, "a"]) else str(df.at[i, "a"])
        b = "" if pd.isna(df.at[i, "b"]) else str(df.at[i, "b"])
        if not a or not b or a == b:
            continue

        # 英数・記号だけの差は 'reading_eq' 領域（ここでは扱わない＝従来）
        try:
            if is_symbol_only_surface_diff(a, b):
                continue
        except Exception:
            pass

        # 厳格読み一致？
        try:
            eq_strict = _readings_equal_strict(a, b)
        except Exception:
            eq_strict = False
        if not eq_strict:
            continue

        # --- 優先1：片側にでも英数・記号ノイズが含まれるなら reading_eq 優先 ---
        try:
            if _has_non_japanese_noise(a) or _has_non_japanese_noise(b):
                df.at[i, "reason"] = "reading_eq"
                df.at[i, "score"] = reading_eq_score(a, b)
                continue
        except Exception:
            # 失敗時は従来どおりの昇格判断へフォールバック
            pass

        # --- 優先2：厳格読み一致のみでも 'reading_same' へ昇格 ---
        # （この分岐に来るのは、上のノイズ判定に該当しない純粋な表記差と考える）
        df.at[i, "reason"] = "reading_same"
        # score は既存のままでも良いが、読みの厳格一致なので 1.0 にしておく手もある
        # ただし最終 enforce_combined_similarity_score() が上書きする前提なら不要
        # df.at[i, "score"] = 1.0

        # --- 優先3：レマ一致 & 名詞のみ & 単一形態素 & 厳格読み一致 も reading_same ---
        # （上で 'reading_same' 済みなので実質そのまま。明示的に残しておくなら以下）
        # la, lb = surf2lemma.get(a, ""), surf2lemma.get(b, "")
        # if la and lb and (la == lb):
        #     if _is_noun_only(a) and _is_noun_only(b) and _is_single_morpheme(a) and _is_single_morpheme(b):
        #         df.at[i, "reason"] = "reading_same"

    return df


# =============================================================================
# 'reading_same' の最終サニタイズ（厳格読み一致を満たさなければ 'reading' に戻す）
def sanitize_reading_same(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "reason" not in df.columns:
        return df
    m = df["reason"].eq("reading_same")
    if not m.any():
        return df
    df = df.copy()
    bad = df.loc[m].apply(
        lambda r: not _readings_equal_strict(str(r.get("a", "")), str(r.get("b", ""))),
        axis=1
    )
    if bad.any():
        df.loc[bad[bad].index, "reason"] = "reading"  # 取りやめて 'reading' に戻す
    return df
# =============================================================================

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

# ====== ここから差し替え（堅めのフォールバック入り） ======
def _compose_text_from_rawdict(
    page,
    *,
    vertical_strategy: str = "auto",   # "auto" | "force_v" | "force_h"
    drop_marginal: bool = False,
    margin_ratio: float = 0.045,

    # カラム検出
    col_bin_ratio_h: float = 0.08,     # 横: ビン幅（ページ幅比）
    col_bin_ratio_v: float = 0.06,     # 縦: ビン幅（ページ幅比; やや細かめ）

    # ギャップ閾値
    span_space_ratio_h: float = 0.006, # 横: スペース
    cell_tab_ratio_h: float   = 0.018, # 横: タブ
    char_space_ratio_v: float = 0.004, # 縦: charベース スペース
    char_tab_ratio_v: float   = 0.012, # 縦: charベース タブ

    # ルビ抑制
    drop_ruby: bool = True,
    ruby_size_ratio: float = 0.60,     # 行の中央値サイズ × 0.60 未満をルビ候補
):
    """
    縦組み強化版のブロック整形:
      - 行ごとに縦横判定（wmode / dir / char分布のIQR比）
      - 縦行は char 単位で連結（スペース/タブ判定は Δy）
      - 縦カラムは x中心でクラスタ → 右→左に結合
      - 横は従来の spanベース（Δx）で連結
    """
    rd = page.get_text("rawdict")
    blocks = rd.get("blocks", []) if isinstance(rd, dict) else []
    if not blocks:
        return ""

    width  = float(page.rect.width)
    height = float(page.rect.height)

    # 絶対値へ
    span_space_h = max(1.5, width  * span_space_ratio_h)
    cell_tab_h   = max(3.0, width  * cell_tab_ratio_h)
    char_space_v = max(1.0, height * char_space_ratio_v)
    char_tab_v   = max(2.0, height * char_tab_ratio_v)

    col_bin_w_h = max(16.0, width * col_bin_ratio_h)
    col_bin_w_v = max(12.0, width * col_bin_ratio_v)

    top_margin = height * margin_ratio
    bot_margin = height * (1.0 - margin_ratio)

    from collections import defaultdict
    import statistics as stats

    def _span_text(sp) -> str:
        t = sp.get("text") or ""
        if t.strip():
            return t
        chs = sp.get("chars")
        if isinstance(chs, list) and chs:
            return "".join(ch.get("c", "") for ch in chs)
        return ""

    def _line_chars(ln):
        """行内の char 配列を平坦化して返す。各要素: (c, x0,y0,x1,y1, size)"""
        out = []
        for sp in ln.get("spans", []):
            size = float(sp.get("size", 0.0) or 0.0)
            chs = sp.get("chars")
            if isinstance(chs, list) and chs:
                for ch in chs:
                    c = ch.get("c", "")
                    x0,y0,x1,y1 = map(float, ch.get("bbox", (0,0,0,0)))
                    out.append((c,x0,y0,x1,y1,size))
            else:
                # charsが無い場合はspan.textを1文字ずつ（近似）
                t = sp.get("text") or ""
                if not t:
                    continue
                x0,y0,x1,y1 = map(float, sp.get("bbox", (0,0,0,0)))
                # 幅/文字数で荒く分割（均等割り）
                n = max(1, len(t))
                w = (x1 - x0) / n if n else 0
                for i, c in enumerate(t):
                    cx0 = x0 + i*w
                    cx1 = cx0 + w
                    out.append((c, cx0, y0, cx1, y1, size))
        return out

    def _is_vertical_line(ln) -> bool:
        # 1) wmode / dir を優先
        wmode = ln.get("wmode")
        if isinstance(wmode, int) and wmode == 1:
            return True
        dirv = ln.get("dir")
        if isinstance(dirv, (list, tuple)) and len(dirv) >= 2:
            try:
                dx, dy = float(dirv[0]), float(dirv[1])
                if abs(dy) > abs(dx):
                    return True
            except Exception:
                pass

        # 2) char 分布で判定（IQR 比でロバスト）
        chars = _line_chars(ln)
        if len(chars) < 2:
            return False  # 情報不足→横扱い
        xs = [(x0+x1)/2 for _,x0,y0,x1,y1,_ in chars]
        ys = [(y0+y1)/2 for _,x0,y0,x1,y1,_ in chars]
        try:
            iqr_x = stats.quantiles(xs, n=4)[2] - stats.quantiles(xs, n=4)[0]
            iqr_y = stats.quantiles(ys, n=4)[2] - stats.quantiles(ys, n=4)[0]
        except Exception:
            # フォールバック：分散比
            import math
            mean_x = sum(xs)/len(xs); mean_y = sum(ys)/len(ys)
            iqr_x = sum((x-mean_x)**2 for x in xs)/len(xs)
            iqr_y = sum((y-mean_y)**2 for y in ys)/len(ys)
        # y広がりが明確に大きければ縦
        return (iqr_y > iqr_x * 1.6)

    def _build_line_text_vertical(ln) -> str:
        """縦行を char 単位で復元（ルビ抑制・Δyでスペ/タブ判定）"""
        chars = _line_chars(ln)
        if not chars:
            return ""
        # ルビ抑制のため、行のフォントサイズ中央値
        if drop_ruby:
            sizes = [sz for *_, sz in chars if sz > 0]
            med = stats.median(sizes) if sizes else 0.0
        # 読みは上→下
        chars.sort(key=lambda t: t[2])  # y0

        parts = []
        prev_y1 = None
        for c, x0, y0, x1, y1, sz in chars:
            if drop_ruby and med and sz and sz < med * ruby_size_ratio:
                # ルビはスキップ
                continue
            if prev_y1 is None:
                parts.append(c)
            else:
                gap = y0 - prev_y1
                if gap >= char_tab_v:
                    parts.append("\t"); parts.append(c)
                elif gap >= char_space_v:
                    parts.append(" ");  parts.append(c)
                else:
                    parts.append(c)
            prev_y1 = y1
        return "".join(parts).strip()

    def _build_line_text_horizontal(ln, span_space=span_space_h, cell_tab=cell_tab_h) -> str:
        spans = ln.get("spans", [])
        if not spans:
            return ""
        # xで並べ替え
        spans = sorted(spans, key=lambda sp: float(sp.get("bbox", [0,0,0,0])[0]))
        parts = []
        prev_x1 = None
        for sp in spans:
            txt = (_span_text(sp) or "").replace("\u00A0", " ").strip()
            if not txt:
                continue
            x0, y0, x1, y1 = map(float, sp.get("bbox", (0,0,0,0)))
            if prev_x1 is None:
                parts.append(txt)
            else:
                gap = x0 - prev_x1
                if gap >= cell_tab:
                    parts.append("\t"); parts.append(txt)
                elif gap >= span_space:
                    parts.append(" ");  parts.append(txt)
                else:
                    parts.append(txt)
            prev_x1 = x1
        return "".join(parts).strip()

    # ---- ブロック収集＆行ごとに縦横付与 ----
    blist = []
    for b in blocks:
        if b.get("type", 1) != 0:
            continue
        x0,y0,x1,y1 = map(float, b.get("bbox", (0,0,0,0)))
        if drop_marginal and (y1 <= top_margin or y0 >= bot_margin):
            continue
        lines = b.get("lines", [])
        if not lines:
            continue

        # 行ごとの縦横
        ln_items = []
        v_cnt = 0; h_cnt = 0
        for ln in lines:
            is_v = _is_vertical_line(ln)
            ln_items.append((ln, is_v))
            if is_v: v_cnt += 1
            else:    h_cnt += 1

        orient = "v" if v_cnt > h_cnt else "h"
        blist.append({
            "bbox": (x0,y0,x1,y1),
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "lines_oriented": ln_items,
            "orient": orient
        })

    if not blist:
        return ""

    # ページ優勢方向
    v_total = sum(1 for tb in blist if tb["orient"] == "v")
    h_total = len(blist) - v_total
    if vertical_strategy == "force_v":
        page_mode = "v"
    elif vertical_strategy == "force_h":
        page_mode = "h"
    else:
        page_mode = "v" if v_total > h_total else "h"

    # カラムバスケット
    cols_h = defaultdict(list)
    cols_v = defaultdict(list)

    for tb in blist:
        if tb["orient"] == "v":
            # 縦：x中心でビン
            cx = (tb["x0"] + tb["x1"]) / 2
            key = int(cx // col_bin_w_v)
            cols_v[key].append(tb)
        else:
            key = int(tb["x0"] // col_bin_w_h)
            cols_h[key].append(tb)

    def _build_block_text(tb, vertical_block: bool) -> str:
        texts = []
        for ln, is_v in tb["lines_oriented"]:
            if vertical_block or is_v:
                t = _build_line_text_vertical(ln)
            else:
                t = _build_line_text_horizontal(ln)
            if t:
                texts.append(t)
        return "\n".join(texts)

    # 横: 左→右
    parts_h = []
    for ck in sorted(cols_h.keys()):
        col_blocks = sorted(cols_h[ck], key=lambda b: (b["y0"], b["x0"]))
        col_texts = []
        for b in col_blocks:
            t = _build_block_text(b, vertical_block=False)
            if t:
                col_texts.append(t)
        if col_texts:
            parts_h.append("\n".join(col_texts))

    # 縦: 右→左（ck降順）
    parts_v = []
    for ck in sorted(cols_v.keys(), reverse=True):
        col_blocks = sorted(cols_v[ck], key=lambda b: (b["y0"], b["x0"]))
        col_texts = []
        for b in col_blocks:
            t = _build_block_text_vertical_charcloud(
                b,
                char_space_v=char_space_v,
                char_tab_v=char_tab_v,
                drop_ruby=True,
                ruby_size_ratio=0.60
            )
            if t:
                col_texts.append(t)
        if col_texts:
            parts_v.append("\n".join(col_texts))


    if page_mode == "v":
        chunks = []
        if parts_v: chunks.append("\n".join(parts_v))
        if parts_h: chunks.append("\n".join(parts_h))
        return "\n\n".join([c for c in chunks if c])
    else:
        chunks = []
        if parts_h: chunks.append("\n".join(parts_h))
        if parts_v: chunks.append("\n".join(parts_v))
        return "\n\n".join([c for c in chunks if c])

def _build_block_text_vertical_charcloud(tb,
                                         *,
                                         char_space_v: float,
                                         char_tab_v: float,
                                         drop_ruby: bool = True,
                                         ruby_size_ratio: float = 0.60) -> str:
    """
    縦ブロックを、行を信用せずブロック内の全Charから再構成。
    - X中心でカラム化（右→左の順）
    - 各カラムは Y 昇順で連結、ΔYでスペース/タブを挿入
    """
    lines_oriented = tb.get("lines_oriented") or []
    # ---- 1) 全Char収集 ----
    chars = []
    for ln, _is_v in lines_oriented:
        for sp in ln.get("spans", []):
            size = float(sp.get("size", 0.0) or 0.0)
            chs = sp.get("chars")
            if isinstance(chs, list) and chs:
                for ch in chs:
                    c  = ch.get("c", "")
                    x0,y0,x1,y1 = map(float, ch.get("bbox", (0,0,0,0)))
                    cx = (x0 + x1) / 2.0
                    cy = (y0 + y1) / 2.0
                    chars.append((c, cx, cy, y0, y1, size))
            else:
                # charsが無い場合のフォールバック: span.text を均等割り（精度は低下）
                t = (sp.get("text") or "")
                if not t:
                    continue
                x0,y0,x1,y1 = map(float, sp.get("bbox", (0,0,0,0)))
                n = max(1, len(t))
                w = (x1 - x0) / n if n else 0
                for i, c in enumerate(t):
                    cx = x0 + (i + 0.5) * w
                    cy = (y0 + y1) / 2.0
                    chars.append((c, cx, cy, y0, y1, size))

    if not chars:
        return ""

    # ルビ抑制（サイズ中央値の比で判定）
    if drop_ruby:
        import statistics as stats
        sizes = [sz for *_, sz in chars if sz > 0]
        med = stats.median(sizes) if sizes else 0.0
        if med > 0:
            chars = [t for t in chars if not (t[-1] and t[-1] < med * ruby_size_ratio)]
        if not chars:
            return ""

    # ---- 2) カラム化（X 量子化 → 右→左）----
    # カラムのビン幅はブロック幅の数%程度にしておくと安定
    x0, y0, x1, y1 = tb["x0"], tb["y0"], tb["x1"], tb["y1"]
    bw = max(1.0, (x1 - x0))
    col_bin_w = max(8.0, bw * 0.06)   # 必要なら外から渡してもOK

    from collections import defaultdict
    cols = defaultdict(list)
    for c, cx, cy, ly0, ly1, sz in chars:
        key = int((cx - x0) // col_bin_w)
        cols[key].append((c, cx, cy, ly0, ly1, sz))

    # 右→左（キー降順）で処理
    texts = []
    for k in sorted(cols.keys(), reverse=True):
        col = cols[k]
        # カラム内は上→下（cy昇順）
        col.sort(key=lambda t: t[2])
        parts = []
        prev_y1 = None
        for (c, cx, cy, ly0, ly1, sz) in col:
            if prev_y1 is None:
                parts.append(c)
            else:
                gap = ly0 - prev_y1
                if gap >= char_tab_v:
                    parts.append("\t"); parts.append(c)
                elif gap >= char_space_v:
                    parts.append(" ");  parts.append(c)
                else:
                    parts.append(c)
            prev_y1 = ly1
        t = "".join(parts).strip()
        if t:
            texts.append(t)

    # カラム間は空行で分離
    return "\n\n".join(texts)

def extract_pages(pdf_path: str, *,
                  flatten_for_nlp: bool = False) -> List[PageData]:
    """
    1) rawdict 座標ベースで復元
    2) 空なら 'blocks' でフォールバック
    3) それでも空なら 'text' で最後のフォールバック
    """
    pages: List[PageData] = []
    doc = fitz.open(pdf_path)
    try:
        for i in range(len(doc)):
            page = doc[i]

            # ① rawdict
            text_blocked = _compose_text_from_rawdict(
                page,
                vertical_strategy="auto",   # ← 縦横自動
                drop_marginal=False,        # 必要に応じて True に
                margin_ratio=0.05
            )

            # ② blocks フォールバック
            if not text_blocked.strip():
                try:
                    blks = page.get_text("blocks")
                except Exception:
                    blks = []
                if blks:
                    # (x0,y0,x1,y1, text, block_no, block_type) 形式を想定
                    blks_sorted = sorted(blks, key=lambda t: (t[0], t[1]))
                    parts = []
                    for b in blks_sorted:
                        if len(b) >= 5:
                            txt = (b[4] or "").rstrip()
                            if txt:
                                parts.append(txt)
                    text_blocked = "\n".join(parts)

            # ③ text 最終フォールバック
            if not text_blocked.strip():
                text_blocked = page.get_text("text") or ""

            # NLP用に改行を潰したい場合は既存関数で
            text_final = soft_join_lines_lang_aware(text_blocked) if flatten_for_nlp else text_blocked

            pages.append(PageData(os.path.basename(pdf_path), i + 1, text_final))
    finally:
        doc.close()
    return pages

# ====== ここまで差し替え ======

# ===== 追加: レイアウト由来の“強制区切り”でセグメント化 =====
_HARD_BREAK_RE = re.compile(r"[\t\r\n]+")

def iter_nlp_segments(text: str):
    """
    レイアウト由来の区切り（改行・タブ）でテキストを分割し、
    空セグメントを除いて順に返す。
    """
    if not isinstance(text, str) or not text:
        return
    for seg in _HARD_BREAK_RE.split(text):
        seg = seg.strip()
        if seg:
            yield seg

# ===== GiNZA: 軽量モデル + 文節専用で初期化（差し替え） =====
HAS_GINZA = False
GINZA_NLP = None

# 文節抽出に必要な最小構成（parser + bunsetu_recognizer）
GINZA_ENABLED_COMPONENTS = ("parser", "bunsetu_recognizer")

try:
    import spacy, ginza
    # ▼ モデル選択：環境変数 GINZA_MODEL で上書き可（既定は軽量の ja_ginza）
    #   - ja_ginza: CPU向けの標準モデル（軽量）[1](https://pypi.org/project/ja-ginza/)
    #   - ja_ginza_electra: Transformerベースで重い（必要時のみ）[2](https://pypi.org/project/ja-ginza-electra/)[3](https://huggingface.co/megagonlabs/transformers-ud-japanese-electra-base-ginza)
    model_name = os.environ.get("GINZA_MODEL", "ja_ginza").strip()

    # ▼ 不要コンポーネントをロード時に無効化（文節には不要）
    #   - ner, attribute_ruler, compound_splitter をデフォルトで停止
    #     （parser と bunsetu_recognizer は動かす）[1](https://pypi.org/project/ja-ginza/)
    disable = os.environ.get(
        "GINZA_DISABLE", "ner,attribute_ruler,compound_splitter"
    ).split(",")
    disable = [d.strip() for d in disable if d.strip()]

    GINZA_NLP = spacy.load(model_name, disable=disable)

    # （任意）Sudachi の分割モードをCに固定：複合語優先
    try:
        ginza.set_split_mode(GINZA_NLP, "C")
    except Exception:
        pass

    HAS_GINZA = True
except Exception:
    HAS_GINZA = False
    GINZA_NLP = None
# ===== GiNZA 初期化ここまで =====

# ===== 差し替え：GiNZA 文節抽出（並列 nlp.pipe + 文節専用） =====
def ginza_bunsetsu_chunks(text: str) -> list[str]:
    """
    GiNZA を '文節だけ' に限定して高速に抽出する:
      - nlp.select_pipes(enable=['parser','bunsetu_recognizer']) で実行時も絞る
      - nlp.pipe(..., n_process, batch_size) で並列＆バッチ処理
      - 改行/タブは空白1つに正規化、長さ 1..120 のみ返す（従来仕様を踏襲）
    環境変数:
      GINZA_N_PROCESS : 並列プロセス数（既定 1）
      GINZA_BATCH_SIZE: バッチサイズ    （既定 64）
    """
    ok = HAS_GINZA and (GINZA_NLP is not None)
    if not ok or not isinstance(text, str) or not text:
        return []

    # レイアウト由来のセグメントに分けてからまとめて処理
    segs = [seg for seg in iter_nlp_segments(text)]
    if not segs:
        return []

    # 並列・バッチの設定（必要に応じて環境変数で調整）
    try:
        n_process = max(1, int(os.environ.get("GINZA_N_PROCESS", "1")))
    except Exception:
        n_process = 1
    try:
        batch_size = max(1, int(os.environ.get("GINZA_BATCH_SIZE", "64")))
    except Exception:
        batch_size = 64

    chunks: list[str] = []
    # 実行時も 'parser' と 'bunsetu_recognizer' のみに限定して回す（高速化）[4](https://spacy.io/usage/processing-pipelines/)
    pipes_to_enable = [p for p in GINZA_ENABLED_COMPONENTS if p in GINZA_NLP.pipe_names]
    with GINZA_NLP.select_pipes(enable=pipes_to_enable):
        # nlp.pipe による高速ストリーミング処理（並列/バッチ）[4](https://spacy.io/usage/processing-pipelines/)[6](https://spacy.io/api/language/)
        for doc in GINZA_NLP.pipe(segs, n_process=n_process, batch_size=batch_size):
            # 文単位に分け、各文の文節Spanを取得（GiNZAの文節API）[5](https://github.com/megagonlabs/ginza/blob/develop/docs/bunsetu_api.md)
            for sent in doc.sents:
                for sp in ginza.bunsetu_spans(sent):
                    s = re.sub(r"[\t\r\n]+", " ", sp.text).strip()
                    if 1 <= len(s) <= 120:
                        chunks.append(s)
    return chunks
# ===== 差し替えここまで =====

# ===== 追加: GiNZA 文章（文）抽出 =====
def ginza_sentence_units(text: str, *, max_len: int = 300) -> list[str]:
    """
    レイアウト由来のセグメント毎に GiNZA を流し、doc.sents で文単位に抽出。
    改行・タブは空白1個に正規化し、1..max_len だけ採用。
    GiNZA が無い場合は句読点（。！？）ベースの簡易分割でフォールバック。
    """
    if not isinstance(text, str) or not text:
        return []

    # --- GiNZA あり: spaCy doc.sents で文抽出 ---
    ok = HAS_GINZA and (GINZA_NLP is not None)
    if ok:
        try:
            # 実行時も parser / bunsetu_recognizer を有効（parser があれば sents が出ます）
            pipes_to_enable = [p for p in GINZA_ENABLED_COMPONENTS if p in GINZA_NLP.pipe_names]
            chunks: list[str] = []
            segs = [seg for seg in iter_nlp_segments(text)]
            if not segs:
                return []

            # 並列・バッチ設定は bunsetsu と揃える
            try:
                n_process = max(1, int(os.environ.get("GINZA_N_PROCESS", "1")))
            except Exception:
                n_process = 1
            try:
                batch_size = max(1, int(os.environ.get("GINZA_BATCH_SIZE", "64")))
            except Exception:
                batch_size = 64

            with GINZA_NLP.select_pipes(enable=pipes_to_enable):
                for doc in GINZA_NLP.pipe(segs, n_process=n_process, batch_size=batch_size):
                    for sent in doc.sents:
                        s = re.sub(r"[\t\r\n]+", " ", sent.text).strip()
                        if 1 <= len(s) <= max_len:
                            chunks.append(s)
            return chunks
        except Exception:
            # 下のフォールバックへ
            pass

    # --- フォールバック（GiNZA なし／失敗時）: 句読点ベースの簡易文分割 ---
    out: list[str] = []
    for seg in iter_nlp_segments(text):
        for s in re.split(r"(?<=[。！？!?.])\s*", seg):
            s = re.sub(r"[\t\r\n]+", " ", s).strip()
            if 1 <= len(s) <= max_len:
                out.append(s)
    return out
# ===== 追加ここまで =====

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

# === [NEW] 長音「ー」を保持する読み（骨格） ================================

def phrase_reading_norm_keepchoon(s: str) -> str:
    """
    文字列 s の '読み（骨格）' を生成するが、長音「ー」を保持する版。
    - ひらがな→カタカナ
    - カタカナ以外を除去
    - 長音「ー」は残す（drop_choon=False）
    MeCab があれば各語の reading を使い、なければ表層の簡易変換でフォールバック。
    """
    if not isinstance(s, str) or not s:
        return ""
    ok, _ = ensure_mecab()
    if ok and MECAB_TAGGER is not None:
        toks = tokenize_mecab(s, MECAB_TAGGER)
        parts: List[str] = []
        for surf, pos1, lemma, reading, ct, cf in toks:
            r = reading or hira_to_kata(nfkc(surf))
            parts.append(r)
        joined = "".join(parts)
        return normalize_kana(joined, drop_choon=False)
    # fallback: NFKC → カタカナ化 → カタカナ以外除去（長音は残す）
    return normalize_kana(hira_to_kata(nfkc(s or "")), drop_choon=False)


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

# ===== （任意）置き換え: 複合語もセグメント単位で =====
def mecab_compound_tokens_alljoin(text: str) -> List[str]:
    ok, msg = ensure_mecab()
    if not ok:
        raise RuntimeError(msg)
    if not isinstance(text, str) or not text:
        return []

    out: List[str] = []

    def worker(seg: str):
        toks = tokenize_mecab(seg, MECAB_TAGGER)
        parts: List[str] = []
        has_conn = False
        prev_type: Optional[str] = None

        def flush_local():
            nonlocal parts, has_conn, prev_type
            if parts and has_conn:
                token = "".join(parts)
                if 1 <= len(token) <= 120:
                    out.append(token)
            parts = []
            has_conn = False
            prev_type = None

        for surf, pos1, lemma, reading, ct, cf in toks:
            if not surf:
                continue
            is_word = pos1.startswith(("名詞", "動詞", "形容詞", "助動詞"))
            is_conn = surf and all(ch in CONNECT_CHARS for ch in surf)

            if is_conn:
                if parts and prev_type == 'word':
                    parts.append(surf); has_conn = True; prev_type = 'conn'
                continue
            elif is_word:
                if not parts:
                    parts = [surf]; prev_type = 'word'
                else:
                    if prev_type == 'conn':
                        parts.append(surf); prev_type = 'word'
                    else:
                        flush_local()
                        parts = [surf]; prev_type = 'word'
            else:
                flush_local()

        flush_local()

    for seg in iter_nlp_segments(text):
        worker(seg)

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

# ===== 置き換え: MeCab 文節（セグメント単位 + 改行/タブの空白化） =====
def mecab_bunsetsu_chunks(text: str) -> List[str]:
    ok, msg = ensure_mecab()
    if not ok:
        raise RuntimeError(msg)
    if not isinstance(text, str) or not text:
        return []

    chunks: List[str] = []

    def flush(cur: List[str]):
        if not cur:
            return
        s = "".join(cur)
        # 改行・タブは空白に、連続空白は 1 個に
        s = re.sub(r"[\t\r\n]+", " ", s)
        s = re.sub(r"[ ]{2,}", " ", s).strip()
        if 1 <= len(s) <= 120:
            chunks.append(s)
        cur.clear()

    # ★ 重要：セグメント単位に MeCab を回す
    for seg in iter_nlp_segments(text):
        toks = tokenize_mecab(seg, MECAB_TAGGER)
        cur: List[str] = []
        for surf, pos1, lemma, reading, ct, cf in toks:
            if not surf:
                continue
            # 句読点などの記号で文節を打ち切り
            if surf in PUNCT_SET or pos1.startswith("記号"):
                flush(cur)
                continue

            cur.append(surf)
            # 助詞・助動詞の直後で文節を区切る（元のロジック）
            if pos1 in ("助詞", "助動詞"):
                flush(cur)

        flush(cur)  # セグメント終端で flush
    return chunks

def extract_candidates_bunsetsu(
    pages: List[PageData], min_len=1, max_len=120, min_count=1, top_k=0
) -> List[Tuple[str,int]]:
    cnt = Counter()
    # GiNZA があれば最優先、なければ従来の簡易ルール
    for p in pages:
        chunks = ginza_bunsetsu_chunks(p.text_join) if HAS_GINZA else mecab_bunsetsu_chunks(p.text_join)
        for ch in chunks:
            ch = strip_leading_control_chars(ch)
            if ch and (min_len <= len(ch) <= max_len):
                cnt[ch] += 1
    arr = [(w, c) for w, c in cnt.items() if c >= min_count]
    arr.sort(key=lambda x: (-x[1], x[0]))
    return arr if (not top_k or top_k <= 0) else arr[:top_k]

# ===== 追加: 文章（文）候補抽出 =====
def extract_candidates_sentence(
    pages: List[PageData],
    min_len: int = 1,
    max_len: int = 300,
    min_count: int = 1,
    top_k: int = 0
) -> List[Tuple[str, int]]:
    cnt = Counter()
    for p in pages:
        # GiNZA 優先、無ければ簡易フォールバック
        sents = ginza_sentence_units(p.text_join, max_len=max_len) if HAS_GINZA else ginza_sentence_units(p.text_join, max_len=max_len)
        for s in sents:
            s = strip_leading_control_chars(s)
            if s and (min_len <= len(s) <= max_len):
                cnt[s] += 1
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

# === 合成 2-gram Jaccard × Levenshtein 類似 + 軽ペナルティ ===================
# 重み（お好みで微調整）
CHAR_SIM_W_LEV  = 0.60   # Levenshtein 類似の寄与
CHAR_SIM_W_JACC = 0.40   # 2-gram Jaccard の寄与

# ペナルティ係数（“軽め”）
P_LEN_PER_CHAR  = 0.01   # 表層長差 1 文字あたりの減点
P_LEN_CAP       = 0.05   # 表層長差ペナルティの上限
P_NUM_PER_CHUNK = 0.01   # 数値“まとまり”個数差 1 つあたりの減点
P_NUM_CAP       = 0.03   # 数値まとまりペナルティの上限
P_CORE_ASYM     = 0.02   # 片側だけ和字コア（かな/カナ/漢字）が存在する時の一括減点

def _jaccard_2gram(sa: str, sb: str, A=None, B=None) -> float:
    """
    2-gram Jaccard（かぶり具合）。A/B に事前計算済みの集合を渡せます。
    """
    sa = nfkc(sa or "")
    sb = nfkc(sb or "")
    A = _ngram_set(sa, 2) if A is None else set(A)
    B = _ngram_set(sb, 2) if B is None else set(B)
    inter = len(A & B)
    union = len(A | B)
    return 0.0 if union == 0 else inter / union

def _lev_sim(sa: str, sb: str) -> float:
    """
    正規化 Levenshtein 類似（1 - dist/maxlen）。既存の levenshtein_sim を利用。
    """
    return levenshtein_sim(nfkc(sa or ""), nfkc(sb or ""))

def _char_penalty(a: str, b: str) -> float:
    """
    軽い機械的ペナルティの合算：
      - 表層長差（NFKC + 空白除去後の len 差）
      - 数値“まとまり”個数差（例: '630W[30°C]' -> 2）
      - 片側だけ和字コア（かな/カナ/漢字）がある
    """
    try:
        la = _norm_len_for_surface(a); lb = _norm_len_for_surface(b)
        p_len = min(P_LEN_CAP, P_LEN_PER_CHAR * abs(la - lb))
    except Exception:
        p_len = 0.0

    try:
        na = _numeric_chunk_count(a); nb = _numeric_chunk_count(b)
        p_num = min(P_NUM_CAP, P_NUM_PER_CHUNK * abs(na - nb))
    except Exception:
        p_num = 0.0

    try:
        core_a = _surface_core_for_reading(a)
        core_b = _surface_core_for_reading(b)
        p_core = P_CORE_ASYM if bool(core_a) != bool(core_b) else 0.0
    except Exception:
        p_core = 0.0

    return float(p_len + p_num + p_core)

def combined_char_similarity(
    a: str,
    b: str,
    *,
    sa: str | None = None,
    sb: str | None = None,
    grams_a=None,
    grams_b=None,
    w_lev: float = CHAR_SIM_W_LEV,
    w_jacc: float = CHAR_SIM_W_JACC,
    clamp: tuple[float, float] = (0.0, 1.0),
) -> float:
    """
    文字ベースの最終スコア：
        raw = w_lev * Levenshtein_sim  +  w_jacc * Jaccard2
        score = clamp(raw - penalty)

    - sa/sb: NFKC 済み表層を渡すと再正規化を避けられます
    - grams_a/grams_b: 2-gram 集合（frozenset など）を渡すと再構築を避けられます
    - 戻り値は [0.000, 1.000] に丸め（小数第3位）
    """
    # NFKC 済みを尊重（未指定なら正規化）
    sa = nfkc(sa if sa is not None else (a or ""))
    sb = nfkc(sb if sb is not None else (b or ""))

    # 完全一致は 1.0（basic）
    if sa == sb:
        return 1.0

    # 成分スコア
    jacc = _jaccard_2gram(sa, sb, grams_a, grams_b)
    levs = _lev_sim(sa, sb)
    raw  = (w_lev * levs) + (w_jacc * jacc)
    raw  = max(clamp[0], min(clamp[1], raw))

    # 軽い機械ペナルティ
    pen = _char_penalty(a, b)
    score = max(clamp[0], min(clamp[1], raw - pen))

    return round(float(score), 3)

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

# ============================================================================
# 数字以外一致（数字「まとまり」を無視して比較）
_NUMERIC_CHUNK_RE = re.compile(r"[+\-]?\d(?:[\d,.\s]*\d)?")

# 空白の正規化（半角スペース/全角スペースを1個の半角スペースに）
_WS_RE = re.compile(r"[\s\u3000]+")

def _strip_numbers_like(s: str) -> str:
    """文字列から『数字のまとまり』をすべて除去し、残りを空白正規化して返す。"""
    if s is None:
        s = ""
    s = str(s)
    # 内部のみ NFKC（全角→半角、全角記号→半角記号 等）
    nf = unicodedata.normalize("NFKC", s)
    # 数字まとまりを空文字へ（桁区切り・小数点・連続ドット等を含めて除去）
    t = _NUMERIC_CHUNK_RE.sub("", nf)
    # 余計な空白を整理（比較を安定化）
    t = _WS_RE.sub(" ", t).strip()
    return t

def is_numeric_only_diff(a: str, b: str) -> bool:
    if a is None and b is None:
        return False
    a_s, b_s = "" if a is None else str(a), "" if b is None else str(b)
    if a_s == b_s:
        return False
    # 少なくとも一方に数字があるか（NFKC後に判定）
    a_has = bool(re.search(r"\d", unicodedata.normalize("NFKC", a_s)))
    b_has = bool(re.search(r"\d", unicodedata.normalize("NFKC", b_s)))
    if not (a_has or b_has):
        return False
    # 数字のまとまりを除去して比較
    return _strip_numbers_like(a_s) == _strip_numbers_like(b_s)

def _numeric_only_label_vectorized(df: pd.DataFrame) -> pd.Series:
    a = df["a"].astype("string")
    b = df["b"].astype("string")

    # 事前条件: 元が同一は対象外
    diff_orig = (a != b)

    # “数字が含まれるか” の判定は NFKC 後に実施
    a_nf = a.map(lambda x: unicodedata.normalize("NFKC", "" if pd.isna(x) else str(x)))
    b_nf = b.map(lambda x: unicodedata.normalize("NFKC", "" if pd.isna(x) else str(x)))
    has_digit = a_nf.str.contains(r"\d") | b_nf.str.contains(r"\d")

    # 数字まとまりを除去して比較
    a_stripped = a.map(_strip_numbers_like)
    b_stripped = b.map(_strip_numbers_like)

    mask = diff_orig & has_digit & (a_stripped == b_stripped)
    out = pd.Series("", index=df.index, dtype="string")
    out[mask] = "数字以外一致"
    return out

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

# ===== [REPLACE] build_synonym_pairs_general: 語彙を「名詞」「助・動詞」に分離 =====
def build_synonym_pairs_general(
    df_lex: pd.DataFrame,
    read_sim_th=0.90,
    char_sim_th=0.85,
    scope="語彙",          # ← 呼び出し互換のため残します（内部では無視）
    progress_cb=None
) -> pd.DataFrame:
    """
    語彙ペアの生成。scope を固定の「語彙」にせず、
    表層ごとの最頻 pos1 を用いて「名詞」「助・動詞」に二分して付与します。
    ・名詞 … pos1.startswith("名詞")
    ・助・動詞 … pos1.startswith(("動詞","助動詞","形容詞"))  # 形容詞は用言としてこちらに含める
    ペアの scope は「両方が名詞なら '名詞'、それ以外は '助・動詞'」に統一。
    """
    if df_lex.empty:
        return empty_pairs_df()

    # --- 表層ごとの粗カテゴリを作成（名詞 / 助・動詞） -------------------------
    def _coarse_group(pos1: str) -> str:
        p = str(pos1 or "")
        if p.startswith("名詞"):
            return "名詞"
        if p.startswith(("動詞", "助動詞", "形容詞")):
            return "助・動詞"
        # 未判定は名詞扱い（多くの語彙が名詞のため安全側）
        return "名詞"

    surf_series = df_lex["surface"].astype(str)
    pos1_series = df_lex.get("pos1", pd.Series([""] * len(df_lex)))
    surf2group: Dict[str, str] = {s: _coarse_group(p) for s, p in zip(surf_series, pos1_series)}

    # --- 既存ロジック（読み/文字類似によるペア化）は極力そのまま -------------------
    words = df_lex["surface"].tolist()
    counts = df_lex["count"].tolist()
    lemmas = df_lex["lemma"].fillna("").tolist()
    readings = df_lex["reading"].fillna("").tolist()

    n = len(words)
    norm_surface = [nfkc(w) for w in words]
    norm_read = [normalize_kana(r or "", True) for r in readings]
    ng_char = [_ngram_set(s) for s in norm_surface]

    rows = []
    step = max(1, n // 100)

    def try_reading_like(a, b, ra, rb, th):
        # 第1パス
        sim = reading_sim_with_penalty(a, b, ra, rb, th)
        if sim is not None:
            return sim
        # 第2パス（漢字↔カナなどは少し緩める）
        if _has_kanji(a) != _has_kanji(b):
            th2 = max(0.75, float(th) - 0.15)
            return reading_sim_with_penalty(a, b, ra, rb, th2)
        return None

    for i in range(n):
        for j in range(i + 1, n):
            a, b = words[i], words[j]
            ca, cb = int(counts[i]), int(counts[j])
            la, lb = lemmas[i], lemmas[j]
            ra, rb = norm_read[i], norm_read[j]
            sa, sb = norm_surface[i], norm_surface[j]

            reason = None
            score = 0.0

            # 1) lemma 優先（既存）
            if la and lb and la == lb and a != b:
                if (a == la) or (b == la):
                    reason, score = "lemma", 1.0
                else:
                    reason, score = "inflect", 0.95
            else:
                # 1.5) 和字コア一致のみ（英数記号差）は reading_eq
                if is_symbol_only_surface_diff(a, b) and a != b:
                    reason, score = "reading_eq", reading_eq_score(a, b)
                else:
                    # 2) 読み類似
                    sim_r = try_reading_like(a, b, ra, rb, read_sim_th)
                    if sim_r is not None:
                        reason, score = "reading", round(float(sim_r), 3)
                    else:
                        # 3) 文字類似（合成：2-gram Jaccard × Levenshtein + 軽ペナルティ）
                        inter = len(ng_char[i] & ng_char[j])
                        union = len(ng_char[i] | ng_char[j])
                        if union > 0 and (inter / union) >= 0.30:
                            sim_val = combined_char_similarity(
                                a, b,
                                sa=sa, sb=sb,
                                grams_a=ng_char[i], grams_b=ng_char[j],
                            )
                            if sim_val >= 0.999:
                                reason, score = "basic", 1.0
                            elif sim_val is not None and sim_val >= float(char_sim_th or 0.0):
                                reason, score = "char", sim_val

            if reason:
                # ★ scope を「名詞」「助・動詞」に二分
                ga = surf2group.get(a, "名詞")
                gb = surf2group.get(b, "名詞")
                scope_ab = "名詞" if (ga == "名詞" and gb == "名詞") else "活用語"

                rows.append({
                    "a": a, "b": b,
                    "a_count": ca, "b_count": cb,
                    "reason": reason, "score": score,
                    "scope": scope_ab,
                })

        if progress_cb and (i % step == 0 or i == n - 1):
            progress_cb(i + 1, n)

    if not rows:
        return empty_pairs_df()

    return pd.DataFrame(rows).sort_values(
        ["reason", "score", "a_count", "b_count"],
        ascending=[True, False, False, False]
    )
# ===== [/REPLACE] ==============================================================

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
    文字類似は「2-gram Jaccard × Levenshtein の合成 + 軽ペナルティ」を用いる。
    """
    (i_start, i_end, words, counts, norm_char, read_norm,
     ngram_list, char_sim_th, read_sim_th, scope) = args

    rows = []
    n = len(words)
    JACC_TH = 0.30  # 粗フィルタ用（前段の足切り）

    # 逆引きインデックス（ローカル構築）
    from collections import defaultdict
    inv = defaultdict(list)
    for idx, grams in enumerate(ngram_list):
        for g in grams:
            inv[g].append(idx)
    for g, lst in inv.items():
        inv[g] = sorted(set(lst))

    # 読み類似（第1/第2パス）
    def try_reading_like(a, b, ra, rb, th):
        sim = reading_sim_with_penalty(a, b, ra, rb, th)
        if sim is not None:
            return sim
        if _has_kanji(a) != _has_kanji(b):
            th2 = max(0.75, float(th) - 0.15)
            return reading_sim_with_penalty(a, b, ra, rb, th2)
        return None

    for i in range(i_start, i_end):
        a  = words[i]
        sa = norm_char[i]
        Ai = ngram_list[i]

        # まず 2-gram の共起から候補集合
        pool = set()
        for g in Ai:
            pool.update(inv.get(g, ()))
        cand_js = [j for j in pool if j > i and j < n]

        for j in cand_js:
            b  = words[j]

            # 【読み一致（和字コア一致のみ）→ 既存ロジック】
            if read_sim_th is not None and is_symbol_only_surface_diff(a, b) and a != b:
                rows.append({
                    "a": a, "b": b,
                    "a_count": counts[a], "b_count": counts[b],
                    "reason": "reading_eq",
                    "score": reading_eq_score(a, b),
                    "scope": scope,
                })
                continue

            # 【読み類似】（既存）
            if read_sim_th is not None:
                ra = read_norm[i] if read_norm[i] is not None else ""
                rb = read_norm[j] if read_norm[j] is not None else ""
                sim_r = try_reading_like(a, b, ra, rb, read_sim_th)
                if sim_r is not None:
                    rows.append({
                        "a": a, "b": b,
                        "a_count": counts[a], "b_count": counts[b],
                        "reason": "reading",
                        "score": round(float(sim_r), 3),
                        "scope": scope,
                    })
                    continue

            # 【文字類似】ここを “合成スコア” に差し替え
            Aj = ngram_list[j]
            inter = len(Ai & Aj)
            union = len(Ai | Aj)
            if union == 0:
                continue
            jacc = inter / union
            if jacc < JACC_TH:  # 粗フィルタ
                continue

            sb = norm_char[j]
            sim_val = combined_char_similarity(
                a, b,
                sa=sa, sb=sb,
                grams_a=Ai, grams_b=Aj,
            )

            # 完全一致は "basic"、それ以外は "char"
            if sim_val >= 0.999:
                rows.append({
                    "a": a, "b": b,
                    "a_count": counts[a], "b_count": counts[b],
                    "reason": "basic", "score": 1.0,
                    "scope": scope,
                })
            elif sim_val >= float(char_sim_th or 0.0):
                rows.append({
                    "a": a, "b": b,
                    "a_count": counts[a], "b_count": counts[b],
                    "reason": "char", "score": sim_val,
                    "scope": scope,
                })

    return rows

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
REASON_PRIORITY = {
    "basic": 7.2,          # 基本一致（NFKC同一）
    "reading_eq": 6.8,     # 読み一致（英数記号除く）
    "reading_same": 6.5,   # ★ 追加：読み一致（表記違い）
    "reading": 6.0,        # 読み類似
    "lemma": 5.0,
    "inflect": 4.5,
    "lemma_read": 4.0,
    "char": 2.0,
}

# --- REPLACE: unify_pairs (簡素化) ---
def unify_pairs(*dfs: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for d in dfs:
        if d is None or d.empty:
            continue
        t = d.copy()
        for col in PAIR_COLUMNS:
            if col not in t.columns:
                t[col] = pd.NA
        frames.append(t[PAIR_COLUMNS].copy())
    if not frames:
        return empty_pairs_df()

    df = pd.concat(frames, ignore_index=True)

    # 正規化キー（a<=b で並べ替えたペア）
    def _norm_row(r):
        a, b = r["a"], r["b"]
        if pd.isna(a) or pd.isna(b):
            return a, b
        return (a, b) if str(a) <= str(b) else (b, a)

    df[["a_n", "b_n"]] = df.apply(_norm_row, axis=1, result_type="expand")

    # 優先度と合算頻度を付与
    df["prio"] = df["reason"].map(REASON_PRIORITY).fillna(1.0)
    df["sum_count"] = (
        pd.to_numeric(df["a_count"], errors="coerce").fillna(0).astype(int) +
        pd.to_numeric(df["b_count"], errors="coerce").fillna(0).astype(int)
    )
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)

    # （a_n,b_n）グループで最良（prio→score→sum_count の降順）を1件選抜
    df = df.sort_values(
        ["a_n", "b_n", "prio", "score", "sum_count"],
        ascending=[True, True, False, False, False]
    ).drop_duplicates(["a_n", "b_n"], keep="first")

    # 表示用の最終順
    df = df.drop(columns=["a_n", "b_n", "prio", "sum_count"], errors="ignore")
    return df.sort_values(["reason", "score", "a", "b"], ascending=[True, False, True, True]).reset_index(drop=True)
# --- /REPLACE ---

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

# ==== ここから差し替え：2→3分類に拡張（新カテゴリを表示側に追加） ====
REASON_TO_GROUP = {
    "basic": "basic",
    "reading_eq": "reading_eq",
    "reading_same": "reading_same",  # ★ 追加
    "lemma": "reading",
    "lemma_read": "reading",
    "reading": "reading",
    "inflect": "reading",
    "char": "char",
}

REASON_GROUP_JA = {
    "basic": "基本一致",
    "reading_eq": "読み一致（英数記号除く）",
    "reading_same": "読み一致（表記違い）",  # ★ 追加
    "reading": "読み類似",
    "char": "文字類似",
}

def apply_reason_ja(df: pd.DataFrame) -> pd.DataFrame:
    """
    reason列（英）→ 「一致要因」（日本語）の3分類に変換。
    - 未知のreasonは安全側で「文字類似」にフォールバック。
    - 最終的に「reason」列は削除。
    """
    if df is None or df.empty or "reason" not in df.columns:
        return df
    df = df.copy()
    grp_key = df["reason"].map(REASON_TO_GROUP).fillna("char")
    df["一致要因"] = grp_key.map(REASON_GROUP_JA).fillna("文字類似")
    df = df.drop(columns=["reason"])
    return df

# ------------------------------------------------------------
# Diff（インライン差分のみ）
# ------------------------------------------------------------
import html

def _esc_html(s: str) -> str:
    return "" if s is None else html.escape(str(s), quote=False)

# ===== [REPLACE] html_inline_diff（クラス付与版：eq/tok-del/tok-ins） =====
def html_inline_diff(a: str, b: str) -> str:
    """
    インライン差分を HTML で返す。
    等しい部分: <span class='eq'>…</span>
    削除      : <span class='tok-del'>…</span>
    追加      : <span class='tok-ins'>…</span>
    """
    sm = difflib.SequenceMatcher(a=a or "", b=b or "")
    out = []
    A = a or ""; B = b or ""
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            seg = _esc_html(A[i1:i2])
            if seg:
                out.append(f"<span class='eq'>{seg}</span>")
        elif op == "delete":
            seg = _esc_html(A[i1:i2])
            if seg:
                out.append(f"<span class='tok-del'>{seg}</span>")
        elif op == "insert":
            seg = _esc_html(B[j1:j2])
            if seg:
                out.append(f"<span class='tok-ins'>{seg}</span>")
        elif op == "replace":
            del_seg = _esc_html(A[i1:i2])
            ins_seg = _esc_html(B[j1:j2])
            if del_seg:
                out.append(f"<span class='tok-del'>{del_seg}</span>")
            if ins_seg:
                out.append(f"<span class='tok-ins'>{ins_seg}</span>")
    return "".join(out)
# ===== [/REPLACE] =====

# ===== [REPLACE] 左プレビュー: 統合インライン差分の見た目（薄い未変更＋淡背景） =====
def build_unified_inline_diff_embed(a: str, b: str) -> str:
    a = "" if a is None else str(a)
    b = "" if b is None else str(b)
    body = html_inline_diff(a, b)

    css = """
    <style>
      html, body { margin:0; padding:0; background:#fff; }
      body {
        font-family: 'Yu Gothic UI','Noto Sans JP',sans-serif;
        font-size: 16px;                /* お好みで 15〜18px */
        line-height: 1.5; letter-spacing: .2px;
      }
      .box {
        border: 1px solid #dee2e6; border-radius: 6px;
        padding: 8px 10px; background: #fff;
        white-space: pre-wrap; word-break: break-word;
      }
      /* 等しい部分：薄めのグレー（前回よりさらに薄く） */
      .eq       { color:#94a3b8; }       /* 例: #94a3b8 / #9aa0a6 / #a0a7b1 */
      /* 追加：青字＋淡い青背景＋下線を少し太く */
      .tok-ins  {
        color:#1c7ed6; background:#d0ebff;
        text-decoration: underline;
        text-decoration-thickness: 2px; text-underline-offset: 1px;
        border-radius: 2px; padding: 0 1px;
      }
      /* 削除：赤字＋淡い赤背景＋取り消し線を少し太く */
      .tok-del  {
        color:#c92a2a; background:#ffe3e3;
        text-decoration: line-through;
        text-decoration-thickness: 2px;
        border-radius: 2px; padding: 0 1px;
      }
    </style>
    """
    return f"<html><head>{css}</head><body><div class='box'>{body}</div></body></html>"


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

# ===== [REPLACE] 差分プレビュー用：縦並び（見出しなし・A/B表記なし） =====
def build_vertical_diff_html_embed(a: str, b: str) -> str:
    """
    簡易プレビュー向けの縦並び差分（見出しなし/A・B表記なし）。
    上段=旧テキスト相当、下段=新テキスト相当 ですが文言は出しません。
    """
    import difflib, html as _html

    def esc(s: str) -> str:
        return _html.escape(s or "", quote=False)

    def tok_a(a_line: str, b_line: str) -> str:
        sm2 = difflib.SequenceMatcher(a=a_line or "", b=b_line or "")
        parts = []
        for op, i1, i2, j1, j2 in sm2.get_opcodes():
            a_seg = (a_line or "")[i1:i2]
            if op == "equal":
                parts.append(esc(a_seg))
            elif op in ("delete", "replace"):
                if a_seg:
                    parts.append(f"<span class='tok-del'>{esc(a_seg)}</span>")
        return "".join(parts)

    def tok_b(a_line: str, b_line: str) -> str:
        sm2 = difflib.SequenceMatcher(a=a_line or "", b=b_line or "")
        parts = []
        for op, i1, i2, j1, j2 in sm2.get_opcodes():
            b_seg = (b_line or "")[j1:j2]
            if op == "equal":
                parts.append(esc(b_seg))
            elif op in ("insert", "replace"):
                if b_seg:
                    parts.append(f"<span class='tok-ins'>{esc(b_seg)}</span>")
        return "".join(parts)

    a_lines = (a or "").splitlines()
    b_lines = (b or "").splitlines()
    sm = difflib.SequenceMatcher(a=a_lines, b=b_lines)
    ops = sm.get_opcodes()

    css = """
    <style>
      body { background:#fff; color:#111; font-family:'Yu Gothic UI','Noto Sans JP',sans-serif;
             font-size:13px; line-height:1.6; margin:0; }
      .sec { margin:0 0 6px 0; }
      .box { border:1px solid #dee2e6; border-radius:6px; padding:6px 8px; background:#fff; }
      .line { white-space:pre-wrap; padding:1px 0; }
      .eq  {}
      .add { background:#e7f5ff; }
      .del { background:#fff0f0; }
      .rep { background:#fff7e6; }
      .tok-ins { color:#1c7ed6; background:#d0ebff; border-radius:2px; }
      .tok-del { color:#c92a2a; background:#ffe3e3; border-radius:2px; }
    </style>
    """

    a_rows = ["<div class='sec'><div class='box'>"]
    b_rows = ["<div class='sec'><div class='box'>"]

    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            for k in range(i2 - i1):
                text = esc(a_lines[i1 + k])
                a_rows.append(f"<div class='line eq'>{text}</div>")
                b_rows.append(f"<div class='line eq'>{text}</div>")
        elif tag == "delete":
            for k in range(i2 - i1):
                a_line = a_lines[i1 + k]
                a_rows.append(f"<div class='line del'>{tok_a(a_line, '')}</div>")
            for _ in range(i2 - i1):
                b_rows.append("<div class='line del'></div>")
        elif tag == "insert":
            for k in range(j2 - j1):
                b_line = b_lines[j1 + k]
                b_rows.append(f"<div class='line add'>{tok_b('', b_line)}</div>")
            for _ in range(j2 - j1):
                a_rows.append("<div class='line add'></div>")
        elif tag == "replace":
            h = max(i2 - i1, j2 - j1)
            for k in range(h):
                a_line = a_lines[i1 + k] if (i1 + k) < i2 else ""
                b_line = b_lines[j1 + k] if (j1 + k) < j2 else ""
                a_rows.append(f"<div class='line rep'>{tok_a(a_line, b_line) if a_line != '' else ''}</div>")
                b_rows.append(f"<div class='line rep'>{tok_b(a_line, b_line) if b_line != '' else ''}</div>")

    a_rows.append("</div></div>")
    b_rows.append("</div></div>")

    return f"<html><head>{css}</head><body>{''.join(a_rows)}{''.join(b_rows)}</body></html>"
# ===== [/REPLACE] ==============================================================

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

    # --- REPLACE: DiffDialog._build_vertical_html (HTMLエスケープ修正 & 最小化) ---
    def _build_vertical_html(self, a: str, b: str) -> str:
        import difflib, html
        a_lines = (a or "").splitlines()
        b_lines = (b or "").splitlines()
        sm = difflib.SequenceMatcher(a=a_lines, b=b_lines)
        ops = sm.get_opcodes()

        css = """
        <style>
        html, body, div, p { margin:0; padding:0; }
        body {
        font-family:'Yu Gothic UI','Noto Sans JP',sans-serif;
        color:#111; font-size:16px; line-height:1.80; letter-spacing:0.2px;
        }
        .page { padding-top:6px; padding-bottom:0; }
        .sec { margin:0 0 6px 0; } .sec:last-child { margin-bottom:0; }
        .sec-b { margin-top:12px; }
        .sec-head { font-weight:600; margin:0; line-height:0; }
        .sec-a .sec-head { color:#d9480f; }
        .sec-b .sec-head { color:#1c7ed6; }
        .diff { line-height:1.00; }
        table,.wrap { border-collapse:collapse; border-spacing:0; margin:0; }
        .wrap { width:100%; }
        .rail { width:18px; }
        .rail-a { background:#fa5252; } .rail-b { background:#4dabf7; }
        .body { padding:4px 10px 6px 10px; }
        .line { white-space:pre-wrap; padding:2px 4px; }
        .eq{} .add{ background:#e7f5ff; } .del{ background:#fff0f0; } .rep{ background:#fff7e6; }
        .tok-ins{ color:#1c7ed6; background:#d0ebff; border-radius:2px; }
        .tok-del{ color:#c92a2a; background:#ffe3e3; border-radius:2px; }
        </style>
        """

        def esc(s: str) -> str:
            return html.escape(s or "", quote=False)

        def tok_a(a_line: str, b_line: str) -> str:
            sm2 = difflib.SequenceMatcher(a=a_line or "", b=b_line or "")
            parts = []
            for op, i1, i2, j1, j2 in sm2.get_opcodes():
                a_seg = (a_line or "")[i1:i2]
                b_seg = (b_line or "")[j1:j2]
                if op == "equal":
                    parts.append(esc(a_seg))
                elif op in ("delete", "replace"):
                    if a_seg:
                        parts.append(f"<span class='tok-del'>{esc(a_seg)}</span>")
            return "".join(parts)

        def tok_b(a_line: str, b_line: str) -> str:
            sm2 = difflib.SequenceMatcher(a=a_line or "", b=b_line or "")
            parts = []
            for op, i1, i2, j1, j2 in sm2.get_opcodes():
                a_seg = (a_line or "")[i1:i2]
                b_seg = (b_line or "")[j1:j2]
                if op == "equal":
                    parts.append(esc(b_seg))
                elif op in ("insert", "replace"):
                    if b_seg:
                        parts.append(f"<span class='tok-ins'>{esc(b_seg)}</span>")
            return "".join(parts)

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
                    a_rows.append(f"<div class='line del'>{tok_a(a_line, '')}</div>")
                for _ in range(i2 - i1):
                    b_rows.append("<div class='line del'></div>")
            elif tag == "insert":
                for k in range(j2 - j1):
                    b_line = b_lines[j1 + k]
                    b_rows.append(f"<div class='line add'>{tok_b('', b_line)}</div>")
                for _ in range(j2 - j1):
                    a_rows.append("<div class='line add'></div>")
            elif tag == "replace":
                h = max(i2 - i1, j2 - j1)
                for k in range(h):
                    a_line = a_lines[i1 + k] if (i1 + k) < i2 else ""
                    b_line = b_lines[j1 + k] if (j1 + k) < j2 else ""
                    a_rows.append(f"<div class='line rep'>{tok_a(a_line, b_line) if a_line != '' else ''}</div>")
                    b_rows.append(f"<div class='line rep'>{tok_b(a_line, b_line) if b_line != '' else ''}</div>")

        a_rows += ["</td></tr></table>", "</div>", "</div>"]
        b_rows += ["</td></tr></table>", "</div>", "</div>"]
        return f"<html><head>{css}</head><body>{''.join(a_rows)}{''.join(b_rows)}</body></html>"
    # --- /REPLACE ---
# ------------------------------------------------------------
# Qt Models / Delegate / Style / Proxy
# ------------------------------------------------------------
# ===== ツールチップ遅延表示フィルタ（完全版：差し替え用） =====
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
            if pd.isna(v):
                return ""
            # ★ score 列だけは小数点第3位で固定表示
            try:
                col_name = self._df.columns[c - 1]
            except Exception:
                col_name = ""
            if col_name == "score":
                try:
                    return f"{float(v):.3f}"
                except Exception:
                    return str(v)
            return str(v)

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

# ===== [REPLACE] CheckBoxDelegate（センター描画 + sizeHint + __init__） =====
class CheckBoxDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)

    def sizeHint(self, option, index):
        # セルの推奨サイズ（高さに追随しつつ 16px 基準）
        h = max(16, option.rect.height())
        return QSize(h, h)

    def paint(self, painter: QPainter, option, index):
        if index.column() != 0:
            return super().paint(painter, option, index)

        state = index.model().data(index, Qt.CheckStateRole)
        rect = option.rect
        size = min(rect.height(), 16)  # 16px を目安
        x = rect.x() + (rect.width() - size) // 2
        y = rect.y() + (rect.height() - size) // 2
        r = QRect(x, y, size, size)

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        # 枠
        pen = QPen(QColor("#111")); pen.setWidth(1)
        painter.setPen(pen); painter.setBrush(QColor("#fff"))
        painter.drawRoundedRect(r, 2, 2)
        # チェック
        if state == Qt.Checked:
            pen = QPen(QColor("#111")); pen.setWidth(2)
            painter.setPen(pen)
            x1 = r.x() + int(size*0.22); y1 = r.y() + int(size*0.56)
            x2 = r.x() + int(size*0.46); y2 = r.y() + int(size*0.80)
            x3 = r.x() + int(size*0.80); y3 = r.y() + int(size*0.28)
            painter.drawLine(x1,y1,x2,y2); painter.drawLine(x2,y2,x3,y3)
        painter.restore()
# ===== [/REPLACE] ===========================================================

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
        self._hide_contains = False   # 内包関係除外
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

        # 4.7) ▼ 新規：内包関係除外（a ∈ b もしくは b ∈ a）
        if self._hide_contains:
            ca, cb = self._col("a"), self._col("b")
            if ca >= 0 and cb >= 0:
                a_txt = m.data(m.index(row, ca, parent), Qt.DisplayRole) or ""
                b_txt = m.data(m.index(row, cb, parent), Qt.DisplayRole) or ""
                # NFKC正規化 + 小文字化で安定判定（全角/半角・ケース差を吸収）
                sa = nfkc(str(a_txt)).lower()
                sb = nfkc(str(b_txt)).lower()
                # 完全一致は“内包”から除外（== は隠さない）
                if sa and sb and sa != sb and (sa.find(sb) != -1 or sb.find(sa) != -1):
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

    # --- 追加: 内包関係除外のオン/オフ ---
    def setHideContainment(self, hide: bool):
        self._hide_contains = bool(hide)
        self.invalidateFilter()


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

            # ===== 挿入: 文章（文）候補抽出 → ペア生成 =====
            # 80%: 文章候補抽出
            self.progress_text.emit("文章候補抽出中…")
            tokens_sentence = extract_candidates_sentence(
                pages, min_len=self.min_len, max_len=max(self.max_len, 300),  # 文は少し長めも許容
                min_count=self.min_count_lex, top_k=0
            )
            self._emit(80)

            # 81〜88%: 文章ペア生成
            df_pairs_sentence = build_synonym_pairs_char_only(
                tokens_sentence,
                char_sim_th=self.char_th,
                top_k=self.top_k_lex,
                scope="文章",
                progress_cb=self._subprogress_factory(81, 88, "文章ペア生成"),
                read_sim_th=self.read_th  # 読みもしきい値で判定（bunsetsu と同様）
            )

            # ===== 差し替え: 統合（文章を追加） =====
            # 88%: 統合
            self.progress_text.emit("候補統合中…")
            df_unified = unify_pairs(df_pairs_lex_general, df_pairs_compound, df_pairs_bunsetsu, df_pairs_sentence)
            self._emit(88)
            # ===== 差し替えここまで =====

            # 88.5%: 読み系（lemma/inflect/reading...）を再採点（既存）
            df_unified = recalibrate_reading_like_scores(df_unified, read_th=self.read_th)

            # 88.6%: ★最終スコアを combined に強制（score_reasonは作らない・既存も捨てる）
            df_unified = enforce_combined_similarity_score(
                df_unified,
                keep_backup=False,
                drop_existing_backup=True,
            )

            # 88.7%: ★「読み一致で 1.0」は 'basic' に付け替え
            df_unified = reclassify_basic_for_reading_eq(df_unified, eps=0.0005)

            # ★ 追加: 読み一致（表記違い）へ再分類
            df_unified = reclassify_reading_equal_formdiff(df_unified)
            df_unified = sanitize_reading_same(df_unified)

            # 89%: 小数第3位に丸め（念のため整合）
            if df_unified is not None and not df_unified.empty and "score" in df_unified.columns:
                df_unified["score"] = pd.to_numeric(df_unified["score"], errors="coerce").round(3)

            # 90%: 単独仮名除去
            if not df_unified.empty and "a" in df_unified.columns:
                df_unified = df_unified[~df_unified["a"].apply(is_single_kana_char)].reset_index(drop=True)
            self._emit(90)

            # ===== 差し替え: 92% トークン一覧整形（sentence を追加） =====
            # 92%: トークン一覧整形
            parts = []
            if len(tokens_fine) > 0:
                df_fine = pd.DataFrame(tokens_fine, columns=["token", "count"]); df_fine["type"] = "fine"; parts.append(df_fine)
            if len(tokens_compound) > 0:
                df_comp = pd.DataFrame(tokens_compound, columns=["token", "count"]); df_comp["type"] = "compound"; parts.append(df_comp)
            if len(tokens_bunsetsu) > 0:
                df_bun = pd.DataFrame(tokens_bunsetsu, columns=["token", "count"]); df_bun["type"] = "bunsetsu"; parts.append(df_bun)
            if len(tokens_sentence) > 0:
                df_sent = pd.DataFrame(tokens_sentence, columns=["token", "count"]); df_sent["type"] = "sentence"; parts.append(df_sent)

            df_tokens = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["type", "token", "count"])
            if not df_tokens.empty:
                df_tokens = df_tokens["type token count".split()].sort_values(["type", "count", "token"], ascending=[True, False, True])
            self._emit(92)
            # ===== 差し替えここまで =====


            # 93%: 表示用ラベル変換（2分類）
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

# ===== [REPLACE] DropArea（高さ固定を撤回：固定高さ指定なし） =====
class DropArea(QTextEdit):
    filesChanged = Signal(list)
    def __init__(self):
        super().__init__(); self.setAcceptDrops(True); self.setReadOnly(True)
        self.setPlaceholderText("ここにPDFをドラッグ＆ドロップしてください（複数可）")
        # 見た目はそのまま／高さは固定しない（必要なら最小高さだけ）
        self.setStyleSheet(
            "QTextEdit{border:2px dashed #4dabf7; border-radius:10px; "
            "background:#fff; padding:10px; color:#111; min-height:100px;}"
        )

    def dragEnterEvent(self, e):
        e.acceptProposedAction() if e.mimeData().hasUrls() else e.ignore()
    def dragMoveEvent(self, e):
        e.acceptProposedAction() if e.mimeData().hasUrls() else e.ignore()
    def dropEvent(self, e):
        if not e.mimeData().hasUrls():
            e.ignore(); return
        files = extract_pdf_paths_from_urls(e.mimeData().urls())
        if files:
            self.setText("\n".join(files))
            self.filesChanged.emit(files)
        e.acceptProposedAction()
# ===== [/REPLACE] =====================================================

# =========================================================
# MainWindow（まるごと差し替え）
# =========================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDF 表記ゆれチェック [ver.1.50]")
        self.resize(1180, 860)

        # ---- レイアウト骨格 ----
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ========== 左パネル ==========
        left_panel = QWidget()
        left_panel.setFixedWidth(280)
        left = QVBoxLayout(left_panel)
        left.setContentsMargins(8, 8, 8, 8)
        left.setSpacing(8)

        # 入力PDF
        gb_files = QGroupBox("入力PDF")
        v_files = QVBoxLayout(gb_files)
        # ===== [PATCH] 入力PDFレイアウトの下余白を小さく =====
        v_files.setContentsMargins(8, 8, 8, 4)  # 左,上,右,下
        v_files.setSpacing(6)
        # ===== [/PATCH] =================================
        self.drop = DropArea()
        v_files.addWidget(self.drop)
        row = QHBoxLayout()
        self.btn_browse = QPushButton("ファイルを選択")
        self.btn_clear = QPushButton("クリア")
        row.addWidget(self.btn_browse)
        row.addWidget(self.btn_clear)
        v_files.addLayout(row)
        # ===== [PATCH] ファイル一覧リストを “3行表示” に固定して縦幅を狭く =====
        self.list_files = QListWidget()
        self.list_files.setAlternatingRowColors(True)
        self.list_files.setUniformItemSizes(True)  # 行高の計算を安定化
        self.list_files.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # 3行ぶんの高さに抑える（フォントに応じて自動計算）
        visible_rows = 3
        fm = self.list_files.fontMetrics()
        row_h = max(18, fm.height() + 6)  # 行の目安（字高 + パディング）
        frame = self.list_files.frameWidth() * 2
        # 上下の余白を少なめに見積もって調整
        self.list_files.setFixedHeight(visible_rows * row_h + frame + 2)

        # 件数ラベルも余白を詰めてコンパクトに
        self.lbl_count = QLabel("選択ファイル：0 件")
        self.lbl_count.setStyleSheet("color:#666; margin-top:2px;")
        # ===== [/PATCH] =======================================================
        v_files.addWidget(self.list_files)
        v_files.addWidget(self.lbl_count)

        # 設定
        gb_params = QGroupBox("設定")
        form = QFormLayout(gb_params)
        self.dsb_read = QDoubleSpinBox()
        self.dsb_read.setRange(0.0, 1.0)
        self.dsb_read.setSingleStep(0.05)
        self.dsb_read.setValue(0.90)
        self.dsb_char = QDoubleSpinBox()
        self.dsb_char.setRange(0.0, 1.0)
        self.dsb_char.setSingleStep(0.05)
        self.dsb_char.setValue(0.85)
        form.addRow("読み 類似しきい値", self.dsb_read)
        form.addRow("文字 類似しきい値", self.dsb_char)

        # 実行・進捗
        self.btn_run = QPushButton("解析開始")
        self.btn_run.setObjectName("btnRun")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.lbl_stage = QLabel("待機中")
        self.lbl_elapsed = QLabel("経過 00:00")

        left.addWidget(gb_files)
        left.addWidget(gb_params)
        # ===== [PATCH] 差分プレビュー領域を少し狭く（高さ/余白/パディング） =====
        gb_diff = QGroupBox("簡易差分")
        v_diff = QVBoxLayout(gb_diff)
        # 余白と間隔を詰める（下方向をやや強め）
        v_diff.setContentsMargins(8, 4, 8, 6)  # L, T, R, B
        v_diff.setSpacing(4)

        self.diff_view = QTextBrowser()
        self.diff_view.setOpenExternalLinks(False)
        self.diff_view.setReadOnly(True)

        # パディングを小さく・枠は維持、背景は白
        self.diff_view.setStyleSheet(
            "QTextBrowser{background:#fff; border:0px solid #dee2e6; "
            "border-radius:6px; padding:10px 6px;}"  # ← 6px/8px → 4px/6px に縮小
        )

        # 高さは“上限”を付けてコンパクトに（必要なら数値を微調整）
        self.diff_view.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.diff_view.setMaximumHeight(120)  # ← 140〜180でお好み調整

        # スクロールバーは必要時のみ表示（高さ上限内でスクロール）
        self.diff_view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.diff_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # 初期メッセージ（そのまま or 省略済み）
        self.diff_view.setHtml(
            "<html><body style='font-family:Yu Gothic UI, Noto Sans JP, sans-serif; "
            "color:#666; font-size:12px; margin:4px 6px;'>"
            "表のセルを選択すると、簡易差分をここに表示します。"
            "</body></html>"
        )

        v_diff.addWidget(self.diff_view)

        # GroupBox 自体も“高さを取りすぎない”ように最大方針
        gb_diff.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        # 追加位置は従来どおり
        left.addWidget(gb_diff)
        # ===== [/PATCH] ======================================================

        left.addStretch(1)
        left.addWidget(self.btn_run)
        left.addWidget(self.progress)
        left.addWidget(self.lbl_stage)
        left.addWidget(self.lbl_elapsed)

        # ========== 右パネル ==========
        right_panel = QWidget()
        right = QVBoxLayout(right_panel)
        right.setContentsMargins(6, 6, 6, 6)
        right.setSpacing(6)

        # --- フィルタ群（統合） ---
        gb_filters = QGroupBox("フィルタ")
        v_filters = QVBoxLayout(gb_filters)
        v_filters.setContentsMargins(8, 8, 8, 8)
        v_filters.setSpacing(4)

        # 1) 全文 ＋ 短文フィルタ
        row_full = QHBoxLayout()
        row_full.setSpacing(8)
        row_full.addWidget(QLabel("全文:"))
        self.ed_filter = QLineEdit()
        self.ed_filter.setPlaceholderText("絞り込みたい語や表現を入力")
        row_full.addWidget(self.ed_filter, 1)

        self.chk_shortlen = QCheckBox("字数")
        self.sb_shortlen = QSpinBox()
        self.sb_shortlen.setRange(1, 120)
        self.sb_shortlen.setValue(3)
        lbl_short_suffix = QLabel("以下を隠す")
        self.chk_shortlen.setToolTip("a 列の文字数が指定値以下の行を非表示にします。")
        self.sb_shortlen.setToolTip("非表示にする最大文字数（a 列の長さ、1〜120）")

        row_full.addSpacing(6)
        row_full.addWidget(self.chk_shortlen)
        row_full.addWidget(self.sb_shortlen)
        row_full.addWidget(lbl_short_suffix)
        v_filters.addLayout(row_full)

        sep1 = QFrame(); sep1.setFrameShape(QFrame.HLine); sep1.setFrameShadow(QFrame.Sunken)
        v_filters.addWidget(sep1)

        # 2) 対象（語彙/複合語/文節）
        row_scope = QHBoxLayout(); row_scope.setSpacing(8)
        row_scope.addWidget(QLabel("対　　象:"))
        self.scope_labels = ["名詞", "活用語", "複合語", "文節", "文章"]
        self.chk_scopes: Dict[str, QCheckBox] = {}
        for s in self.scope_labels:
            cb = QCheckBox(s); cb.setChecked(True); self.chk_scopes[s] = cb
            row_scope.addWidget(cb)
        row_scope.addStretch(1)
        v_filters.addLayout(row_scope)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.HLine); sep2.setFrameShadow(QFrame.Sunken)
        v_filters.addWidget(sep2)

        # 3) 一致要因
        row_reason = QHBoxLayout(); row_reason.setSpacing(8)
        row_reason.addWidget(QLabel("一致要因:"))
        self.reason_labels = [
            "基本一致",
            "読み一致（英数記号除く）",
            "読み一致（表記違い）",
            "読み類似",
            "文字類似",
        ]
        self.chk_reasons: Dict[str, QCheckBox] = {}
        for r in self.reason_labels:
            cb = QCheckBox(r); cb.setChecked(True); self.chk_reasons[r] = cb
            row_reason.addWidget(cb)
        row_reason.addStretch(1)
        v_filters.addLayout(row_reason)

        sep3 = QFrame(); sep3.setFrameShape(QFrame.HLine); sep3.setFrameShadow(QFrame.Sunken)
        v_filters.addWidget(sep3)

        # 4) 端差/数字以外一致
        row_edge = QHBoxLayout(); row_edge.setSpacing(8)
        row_edge.addWidget(QLabel("特殊絞込:"))
        self.chk_edge_prefix = QCheckBox("前1文字差を隠す")
        self.chk_edge_suffix = QCheckBox("後1文字差を隠す")
        self.chk_num_only   = QCheckBox("数字以外一致を隠す")
        
        self.chk_contains    = QCheckBox("内包関係除外")
        self.chk_contains.setChecked(False)
        self.chk_contains.setToolTip("a/b のどちらかがもう一方を含む行を非表示にします。")

        self.chk_edge_prefix.setChecked(False)
        self.chk_edge_suffix.setChecked(False)
        self.chk_num_only.setChecked(False)
        self.chk_edge_prefix.setToolTip("「前1字有無」「前1字違い」の候補を表から除外します。")
        self.chk_edge_suffix.setToolTip("「後1字有無」「後1字違い」の候補を表から除外します。")
        self.chk_num_only.setToolTip("数字だけが異なる候補（例: 1時間30分 ↔ 1時間5分）を表から除外します。")
        row_edge.addWidget(self.chk_edge_prefix)
        row_edge.addWidget(self.chk_edge_suffix)
        row_edge.addWidget(self.chk_num_only)
        row_edge.addWidget(self.chk_contains)
        row_edge.addStretch(1)
        v_filters.addLayout(row_edge)

        right.addWidget(gb_filters)

        # --- タブ＆テーブル ---
        self.tabs = QTabWidget()

        self.view_unified = self._make_table()
        # 行高固定・折り返しなし・省略表記は右側（ElideRight）
        vh = self.view_unified.verticalHeader()
        vh.setSectionResizeMode(QHeaderView.Fixed)
        vh.setDefaultSectionSize(28)
        self.view_unified.setWordWrap(False)
        self.view_unified.setTextElideMode(Qt.ElideRight)  # ← ElideNoneは使わない

        # 右クリックメニュー（差分表示）
        self.view_unified.setContextMenuPolicy(Qt.CustomContextMenu)
        self.view_unified.customContextMenuRequested.connect(self.on_unified_context_menu)

        self.view_tokens = self._make_table_simple()

        self.tabs.addTab(self.view_unified, "表記ゆれ候補")
        self.tabs.addTab(self.view_tokens, "候補（トークン）")
        right.addWidget(self.tabs, 1)

        # --- 下部操作 ---
        ops = QHBoxLayout()
        self.btn_select_all = QPushButton("すべて選択")
        self.btn_clear_sel  = QPushButton("すべて解除")
        self.btn_export     = QPushButton("Excelエクスポート")
        self.btn_mark       = QPushButton("PDFにマーキング")
        # リンク風ヘルプボタン
        help_url = (
            "https://itpcojp-my.sharepoint.com/:b:/g/personal/masahiro_tanaka_itp_co_jp/"
            "Ec84iGbfkvRFl4ZHhK-XR9YBcjDRoWvi6cf3XX59l2sBzg?e=qvFOZP"
        )
        self.btn_help = QPushButton("ヘルプ")
        self.btn_help.setFlat(True)
        self.btn_help.setCursor(Qt.PointingHandCursor)
        self.btn_help.setStyleSheet("""
            QPushButton { color:#1c7ed6; background:transparent; border:none; padding:0 4px; }
            QPushButton:hover { text-decoration: underline; }
        """)
        self.btn_help.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(help_url)))

        ops.addWidget(self.btn_select_all)
        ops.addWidget(self.btn_clear_sel)
        ops.addStretch(1)
        ops.addWidget(self.btn_help)
        ops.addSpacing(8)
        ops.addWidget(self.btn_export)
        ops.addWidget(self.btn_mark)
        right.addLayout(ops)

        # ルートへ追加
        root.addWidget(left_panel)
        root.addWidget(right_panel)
        root.setStretch(0, 0)
        root.setStretch(1, 1)

        # ---- Model / Proxy ----
        self.model_unified = UnifiedModel()
        self.model_tokens  = PandasModel()
        self.proxy_unified = UnifiedFilterProxy()
        self.proxy_unified.setSourceModel(self.model_unified)
        self.proxy_tokens = QSortFilterProxyModel()
        self.proxy_tokens.setSourceModel(self.model_tokens)
        self.proxy_tokens.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.proxy_tokens.setFilterKeyColumn(1)  # tokens: 0:type, 1:token, 2:count

        self.view_unified.setModel(self.proxy_unified)
        # ===== [PATCH] MainWindow.__init__ の「モデル/ビュー接続」直後に追記 =====
        self.view_unified.setModel(self.proxy_unified)
        # チェック列（0列）の描画をセンター揃えデリゲートに
        self.view_unified.setItemDelegateForColumn(0, CheckBoxDelegate(self.view_unified))
        # ヘッダは中央寄せ（✓ ヘッダも中央に）
        self.view_unified.horizontalHeader().setDefaultAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        # ===== [/PATCH] =============================================================
        # ===== [PATCH] 選択変更で差分プレビュー更新 =====
        self.view_unified.selectionModel().selectionChanged.connect(self.on_unified_selection_changed)
        # ===== [/PATCH] =================================
        self.view_tokens.setModel(self.proxy_tokens)

        # ---- Signals ----
        self.drop.filesChanged.connect(self.on_files_changed)
        self.btn_browse.clicked.connect(self.on_browse)
        self.btn_clear.clicked.connect(self.on_clear_files)
        self.btn_run.clicked.connect(self.on_run)

        self.ed_filter.textChanged.connect(self.on_text_filter_changed)
        self.chk_shortlen.stateChanged.connect(self.on_shortlen_filter_changed)
        self.sb_shortlen.valueChanged.connect(self.on_shortlen_filter_changed)
        for cb in self.chk_reasons.values():
            cb.stateChanged.connect(self.on_reason_changed)
        for cb in self.chk_scopes.values():
            cb.stateChanged.connect(self.on_scope_changed)
        self.chk_edge_prefix.stateChanged.connect(self.on_edge_filter_changed)
        self.chk_edge_suffix.stateChanged.connect(self.on_edge_filter_changed)
        self.chk_num_only.stateChanged.connect(self.on_edge_filter_changed)
        self.chk_contains.stateChanged.connect(self.on_edge_filter_changed)

        self.btn_select_all.clicked.connect(lambda: self.model_unified.check_all(True))
        self.btn_clear_sel.clicked.connect(lambda: self.model_unified.check_all(False))
        self.view_unified.clicked.connect(self.on_unified_clicked)
        self.btn_mark.clicked.connect(self.on_mark_pdf)
        self.btn_export.clicked.connect(self.on_export)

        # ---- 状態系 ----
        self.files: List[str] = []
        self._t0: Optional[float] = None
        self._timer = QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._update_elapsed)

        self.df_groups = pd.DataFrame()
        self.surf2gid: Dict[str, int] = {}
        self.gid2members: Dict[int, List[str]] = {}

        # 初期フィルタ状態（重複初期化を排除）
        self._init_filter_state_once()
        self._refresh_candidate_count()

    # =========================================================
    # 小ユーティリティ
    # =========================================================
    # ===== [REPLACE] MainWindow._make_table（Stretch無効 + ヘッダ中央 + スクロール） =====
    def _make_table(self) -> QTableView:
        tv = QTableView()
        tv.setSortingEnabled(True)
        tv.setAlternatingRowColors(True)
        tv.setSelectionBehavior(QAbstractItemView.SelectRows)
        tv.setSelectionMode(QAbstractItemView.SingleSelection)
        tv.verticalHeader().setVisible(False)
        tv.verticalHeader().setDefaultSectionSize(28)
        # ★ ここを False に（最終列が余白で勝手に広がらない）
        tv.horizontalHeader().setStretchLastSection(False)
        tv.horizontalHeader().setHighlightSections(False)
        # ★ ヘッダの文字揃えを中央へ
        tv.horizontalHeader().setDefaultAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        # スクロールの手触り
        tv.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        tv.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        # テキストは折り返さず、長い時は右側省略
        tv.setWordWrap(False)
        tv.setTextElideMode(Qt.ElideRight)
        return tv
    # ===== [/REPLACE] ===========================================================

    def _make_table_simple(self) -> QTableView:
        return self._make_table()

    def _init_filter_state_once(self):
        # Proxy 初期状態の一括反映
        self.proxy_unified.setScopeFilter(self._selected_scopes())
        self.proxy_unified.setReasonFilter(self._selected_reasons())
        self.proxy_unified.setHidePrefixEdge(self.chk_edge_prefix.isChecked())
        self.proxy_unified.setHideSuffixEdge(self.chk_edge_suffix.isChecked())
        self.proxy_unified.setHideNumericOnly(self.chk_num_only.isChecked())
        self.proxy_unified.setHideShortEnabled(self.chk_shortlen.isChecked())
        self.proxy_unified.setHideShortLength(self.sb_shortlen.value())
        self.proxy_unified.setHideContainment(self.chk_contains.isChecked())

    def _selected_scopes(self) -> Set[str]:
        return {k for k, cb in self.chk_scopes.items() if cb.isChecked()}

    def _selected_reasons(self) -> Set[str]:
        # GUI表示ラベル → apply_reason_ja() 後の「一致要因」列の値と一致
        return {k for k, cb in self.chk_reasons.items() if cb.isChecked()}

    def _update_elapsed(self):
        if self._t0 is None:
            self.lbl_elapsed.setText("経過 00:00")
            return
        sec = max(0, int(time.monotonic() - self._t0))
        h, rem = divmod(sec, 3600)
        m, s = divmod(rem, 60)
        fmt = f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
        self.lbl_elapsed.setText(f"経過 {fmt}")

    def _refresh_candidate_count(self):
        try:
            df_u = getattr(self.model_unified, "_df", pd.DataFrame())
            total = int(df_u.shape[0]) if isinstance(df_u, pd.DataFrame) else 0
            visible = int(self.proxy_unified.rowCount())
            self.tabs.setTabText(0, f"表記ゆれ候補[{visible:,}/{total:,}]")
        except Exception:
            self.tabs.setTabText(0, "表記ゆれ候補[0/0]")

    def _cap_ab_widths(self, min_a=220, max_a=320, min_b=220, max_b=320):
        # a/b 列の幅を上限クリップ
        df = getattr(self.model_unified, "_df", pd.DataFrame())
        if df.empty:
            return
        cols = list(df.columns)
        try:
            idx_a = cols.index("a")
            idx_b = cols.index("b")
        except ValueError:
            return
        va = 1 + idx_a  # 先頭のチェック列があるので +1
        vb = 1 + idx_b
        hdr = self.view_unified.horizontalHeader()
        hdr.setSectionResizeMode(va, QHeaderView.Interactive)
        hdr.setSectionResizeMode(vb, QHeaderView.Interactive)
        w = self.view_unified.columnWidth
        self.view_unified.setColumnWidth(va, min(max_a, max(min_a, w(va))))
        self.view_unified.setColumnWidth(vb, min(max_b, max(min_b, w(vb))))

    def _hide_aux_columns(self):
        # 表示を軽くするため補助列を隠す（存在チェック付き）
        view = self.view_unified
        model = view.model()  # proxy
        if not model:
            return
        names_hide = {"端差", "数字以外一致"}  # 必要に応じて追加
        hdr = view.horizontalHeader()
        for c in range(model.columnCount()):
            name = (model.headerData(c, Qt.Horizontal, Qt.DisplayRole) or "").strip()
            if name in names_hide:
                hdr.setSectionResizeMode(c, QHeaderView.Fixed)
                view.setColumnHidden(c, True)

    # ===== [REPLACE] MainWindow._compact_columns（非 a/b 列を Fixed + 上限幅） =====
    def _compact_columns(self):
        view = self.view_unified
        model = view.model()
        if not model:
            return
        hdr = view.horizontalHeader()
        hdr.setMinimumSectionSize(40)

        NUMERIC = {"gid", "字数", "a_count", "b_count", "a数", "b数", "score", "類似度"}
        LABELS  = {"一致要因", "理由", "対象", "scope", "target"}

        # 列名→インデックスを走査
        for c in range(model.columnCount()):
            name = (model.headerData(c, Qt.Horizontal, Qt.DisplayRole) or "").strip()
            if name in {"a", "b"}:
                # a/b はユーザー操作しやすいよう Interactive（幅は別関数で上限クリップ）
                hdr.setSectionResizeMode(c, QHeaderView.Interactive)
                continue

            # a/b 以外は Fixed + 上限幅で抑制
            if name in NUMERIC:
                max_w = 88 if name not in {"gid", "字数"} else 72
            elif name in LABELS:
                max_w = 120
            else:
                max_w = 140  # その他は 140px 上限

            # 現在幅と上限を比較してセット
            cur = view.columnWidth(c)
            view.setColumnWidth(c, min(max_w, cur if cur > 0 else max_w))
            hdr.setSectionResizeMode(c, QHeaderView.Fixed)

    # =========================================================
    # DnD & 入出力
    # =========================================================
    def on_files_changed(self, files: List[str]):
        self.set_files(files)

    def set_files(self, files: List[str]):
        self.files = files
        self.list_files.clear()
        for f in files:
            self.list_files.addItem(f)
        self.lbl_count.setText(f"選択ファイル：{len(files)} 件")

    def on_browse(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "PDFを選択", "", "PDF Files (*.pdf)")
        if paths:
            self.set_files(paths)

    def on_clear_files(self):
        self.set_files([])
        self.drop.clear()

    def on_run(self):
        if not self.files:
            QMessageBox.warning(self, "注意", "PDFを指定してください。")
            return
        if not HAS_MECAB:
            QMessageBox.critical(
                self, "エラー",
                'MeCab (fugashi/unidic-lite) が必要です。\n'
                'インストール例: pip install "fugashi[unidic-lite]"'
            )
            return

        self.btn_run.setEnabled(False)
        self.progress.setValue(0)
        self._t0 = time.monotonic()
        self._update_elapsed()
        self._timer.start()
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
            df_u = df_unified.copy()
            cols = list(df_u.columns)
            pref = [c for c in ["gid", "字数", "a", "b"] if c in cols]
            rest = [c for c in cols if c not in pref]
            df_u = df_u[pref + rest]
        else:
            df_u = df_unified

        self.model_unified.setDataFrame(df_u)
        self.model_tokens.setDataFrame(results.get("tokens", pd.DataFrame()))

        # テーブル調整（再描画負荷を抑えつつ）
        self.view_unified.setUpdatesEnabled(False)
        self.view_unified.setSortingEnabled(False)
        self._cap_ab_widths(min_a=220, max_a=320, min_b=220, max_b=320)
        self._hide_aux_columns()
        self._ensure_ab_visible(min_a=220, min_b=220)
        self._compact_columns()
        self.view_unified.setSortingEnabled(True)
        self.view_unified.setUpdatesEnabled(True)

        self._refresh_candidate_count()
        QApplication.processEvents()
        self.progress.setValue(100)
        self.lbl_stage.setText("完了")
        if self._timer.isActive():
            self._timer.stop()

    def _ensure_ab_visible(self, min_a=220, min_b=220):
        df = getattr(self.model_unified, "_df", pd.DataFrame())
        if df.empty:
            return
        cols = list(df.columns)
        try:
            idx_a = cols.index("a")
            idx_b = cols.index("b")
        except ValueError:
            return
        va = 1 + idx_a  # 先頭チェック列があるため +1
        vb = 1 + idx_b
        self.view_unified.setColumnWidth(va, max(self.view_unified.columnWidth(va), min_a))
        self.view_unified.setColumnWidth(vb, max(self.view_unified.columnWidth(vb), min_b))

    # =========================================================
    # テーブル操作・メニュー
    # =========================================================
    def on_unified_clicked(self, proxy_index: QModelIndex):
        if not proxy_index.isValid() or proxy_index.column() != 0:
            return
        src_index = self.proxy_unified.mapToSource(proxy_index)
        current = self.model_unified.data(src_index, Qt.CheckStateRole)
        new_state = Qt.Unchecked if current == Qt.Checked else Qt.Checked
        self.model_unified.setData(src_index, new_state, Qt.CheckStateRole)

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
        act_diff = QAction("差分を表示…", self)

        def _show():
            dlg = DiffDialog(str(a) if a is not None else "", str(b) if b is not None else "", self)
            dlg.exec()

        act_diff.triggered.connect(_show)
        menu.addAction(act_diff)
        menu.exec(self.view_unified.viewport().mapToGlobal(pos))

    # =========================================================
    # フィルタ変更ハンドラ
    # =========================================================
    def on_text_filter_changed(self, text: str):
        self.proxy_unified.setTextFilter(text)
        self.proxy_tokens.setFilterFixedString(text)
        self._refresh_candidate_count()

    def on_shortlen_filter_changed(self, *_):
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
        self.proxy_unified.setHidePrefixEdge(self.chk_edge_prefix.isChecked())
        self.proxy_unified.setHideSuffixEdge(self.chk_edge_suffix.isChecked())
        self.proxy_unified.setHideNumericOnly(self.chk_num_only.isChecked())
        self.proxy_unified.setHideContainment(self.chk_contains.isChecked())
        self._refresh_candidate_count()

    # =========================================================
    # エクスポート / PDFマーキング
    # =========================================================
    def _build_unified_df_with_selection_all(self) -> pd.DataFrame:
        """GUIの選択状態を含めて、全件の DataFrame を返す（Excel用）"""
        src_df = getattr(self.model_unified, "_df", pd.DataFrame()).copy()
        checks = list(getattr(self.model_unified, "_checks", []))
        if src_df.empty:
            return src_df
        if len(checks) != len(src_df):
            checks = [False] * len(src_df)
        df_all = src_df.reset_index(drop=True)
        df_all.insert(0, "選択", checks)
        if "字数" not in df_all.columns and "a" in df_all.columns:
            df_all["字数"] = df_all["a"].map(lambda x: len(str(x)) if x is not None else 0).astype(int)
        return df_all

    def on_export(self):
        """Excel（全件データ＋初期フィルタ見た目）を出力"""
        import json, math as _math

        try:
            df_all = self._build_unified_df_with_selection_all()
        except Exception:
            src_df = getattr(self.model_unified, "_df", pd.DataFrame()).copy()
            checks = list(getattr(self.model_unified, "_checks", []))
            if src_df.empty:
                QMessageBox.information(self, "情報", "エクスポートするデータがありません。")
                return
            if len(checks) != len(src_df):
                checks = [False] * len(src_df)
            df_all = src_df.reset_index(drop=True)
            df_all.insert(0, "選択", checks)
            if "字数" not in df_all.columns and "a" in df_all.columns:
                df_all["字数"] = df_all["a"].map(lambda x: len(str(x)) if x is not None else 0).astype(int)

        # Excel安全化
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

        # 保存ダイアログ
        path, _ = QFileDialog.getSaveFileName(self, "Excelの保存先", "variants_unified.xlsx", "Excel ファイル (*.xlsx)")
        if not path:
            return

        # GUIフィルタ状態
        scopes_on = sorted(list(self._selected_scopes()))
        reasons_on = sorted(list(self._selected_reasons()))
        hide_prefix = self.chk_edge_prefix.isChecked()
        hide_suffix = self.chk_edge_suffix.isChecked()
        short_on = self.chk_shortlen.isChecked()
        short_n = int(self.sb_shortlen.value()) if short_on else None
        fulltext = (self.ed_filter.text() or "").strip()
        need_full = bool(fulltext)

        # 許可リスト（Blanks対応）
        BLANK_TOKEN = "Blanks"
        edge_allowed_list, edge_allow_blank = None, False
        if "端差" in df_all.columns and (hide_prefix or hide_suffix):
            series = df_all["端差"].astype(str)
            uniq = set(s.strip() for s in series.fillna(""))
            deny = set()
            if hide_prefix: deny |= {"前1字有無", "前1字違い"}
            if hide_suffix: deny |= {"後1字有無", "後1字違い"}
            edge_allow_blank = ("" in uniq)
            allowed = [v for v in sorted(uniq) if v and v not in deny]
            if edge_allow_blank:
                allowed = [BLANK_TOKEN] + allowed
            edge_allowed_list = allowed

        num_allowed_list, num_allow_blank = None, False
        if "数字以外一致" in df_all.columns and self.chk_num_only.isChecked():
            series = df_all["数字以外一致"].astype(str)
            uniq = set(s.strip() for s in series.fillna(""))
            num_allow_blank = ("" in uniq)
            allowed = [v for v in sorted(uniq) if v and v != "数字以外一致"]
            if num_allow_blank:
                allowed = [BLANK_TOKEN] + allowed
            num_allowed_list = allowed

        if need_full:
            cols_for_full = [c for c in ["a", "b", "一致要因", "対象"] if c in df_all.columns]
            df_all["全文__concat"] = (
                df_all[cols_for_full].astype(str).fillna("").agg(" / ".join, axis=1)
            ) if cols_for_full else ""

        # --- 追加: 内包関係除外（Excelの初期表示マスクにも反映） ---
        if self.chk_contains.isChecked() and "a" in df_all.columns and "b" in df_all.columns:
            def _contains_row(a, b):
                sa = nfkc(str(a or "")).lower()
                sb = nfkc(str(b or "")).lower()
                return (sa and sb and sa != sb and (sa.find(sb) != -1 or sb.find(sa) != -1))
            mask &= ~df_all.apply(lambda r: _contains_row(r.get("a", ""), r.get("b", "")), axis=1)


        # 初期表示マスク（True=表示）
        mask = pd.Series(True, index=df_all.index)
        if scopes_on and "対象" in df_all.columns:
            mask &= df_all["対象"].astype(str).isin(scopes_on)
        if reasons_on and "一致要因" in df_all.columns:
            mask &= df_all["一致要因"].astype(str).isin(reasons_on)
        if edge_allowed_list is not None and "端差" in df_all.columns:
            vals = df_all["端差"].astype(str).fillna("")
            nonblank_set = set(v for v in edge_allowed_list if v != BLANK_TOKEN)
            cond_nonblank = vals.str.strip().isin(nonblank_set) if nonblank_set else pd.Series(False, index=vals.index)
            cond_blank = vals.str.strip().eq("") if edge_allow_blank else pd.Series(False, index=vals.index)
            mask &= (cond_nonblank | cond_blank)
        if num_allowed_list is not None and "数字以外一致" in df_all.columns:
            vals = df_all["数字以外一致"].astype(str).fillna("")
            nonblank_set = set(v for v in num_allowed_list if v != BLANK_TOKEN)
            cond_nonblank = vals.str.strip().isin(nonblank_set) if nonblank_set else pd.Series(False, index=vals.index)
            cond_blank = vals.str.strip().eq("") if num_allow_blank else pd.Series(False, index=vals.index)
            mask &= (cond_nonblank | cond_blank)
        if short_on and "字数" in df_all.columns:
            mask &= pd.to_numeric(df_all["字数"], errors="coerce").fillna(0) > short_n
        if need_full and "全文__concat" in df_all.columns:
            mask &= df_all["全文__concat"].astype(str).str.contains(fulltext, case=False, na=False)

        # エクスポート（xlsxwriter 優先）
        try:
            import xlsxwriter  # noqa
            engine = "xlsxwriter"
        except Exception:
            engine = "openpyxl"

        try:
            with pd.ExcelWriter(path, engine=engine) as writer:
                sheet_main = "表記ゆれ候補"
                df_all.to_excel(writer, index=False, sheet_name=sheet_main)
                nrows, ncols = df_all.shape

                def col_idx(col_name):
                    try:
                        return df_all.columns.get_loc(col_name)
                    except Exception:
                        return None

                ci_len   = col_idx("字数")
                ci_reason= col_idx("一致要因")
                ci_scope = col_idx("対象")
                ci_edge  = col_idx("端差")
                ci_num   = col_idx("数字以外一致")
                ci_full  = col_idx("全文__concat") if need_full else None

                if engine == "xlsxwriter":
                    wb = writer.book
                    ws = writer.sheets[sheet_main]
                    ws.freeze_panes(1, 1)
                    ws.autofilter(0, 0, nrows, max(0, ncols - 1))

                    if ci_scope is not None and scopes_on:
                        uniq_scopes = set(str(x) for x in df_all["対象"].dropna().astype(str))
                        if len(scopes_on) < len(uniq_scopes):
                            ws.filter_column_list(ci_scope, [str(v) for v in scopes_on])

                    if ci_reason is not None and reasons_on:
                        uniq_reasons = set(str(x) for x in df_all["一致要因"].dropna().astype(str))
                        if len(reasons_on) < len(uniq_reasons):
                            ws.filter_column_list(ci_reason, [str(v) for v in reasons_on])

                    if ci_edge is not None and edge_allowed_list is not None:
                        uniq_edge = set(str(x).strip() for x in df_all["端差"].fillna(""))
                        if 0 < len(edge_allowed_list) < len(uniq_edge) + (1 if edge_allow_blank else 0):
                            ws.filter_column_list(ci_edge, edge_allowed_list)

                    if ci_num is not None and num_allowed_list is not None:
                        uniq_num = set(str(x).strip() for x in df_all["数字以外一致"].fillna(""))
                        if 0 < len(num_allowed_list) < len(uniq_num) + (1 if num_allow_blank else 0):
                            ws.filter_column_list(ci_num, num_allowed_list)

                    if ci_len is not None and short_on:
                        ws.filter_column(ci_len, f'x > {short_n}')

                    if ci_full is not None and fulltext:
                        def _xf_escape(s: str) -> str:
                            return s.replace('~', '~~').replace('*', '~*').replace('?', '~?')
                        ws.filter_column(ci_full, f"x == *{_xf_escape(fulltext)}*")
                        ws.set_column(ci_full, ci_full, None, None, {"hidden": True})

                    # 非該当行を非表示（ヘッダを除く）
                    for i, ok in enumerate(mask.tolist()):
                        if not ok:
                            ws.set_row(i + 1, None, None, {'hidden': True})

                    # 列幅
                    num_fmt_int = wb.add_format({"align": "right"})
                    num_fmt_3   = wb.add_format({"num_format": "0.000", "align": "right"})
                    def set_w(name, width, fmt=None):
                        ci = col_idx(name)
                        if ci is not None:
                            ws.set_column(ci, ci, width, fmt)
                    set_w("選択", 7)
                    set_w("gid", 7, num_fmt_int)
                    set_w("字数", 6, num_fmt_int)
                    set_w("a", 32); set_w("b", 32)
                    set_w("一致要因", 12); set_w("対象", 10)
                    set_w("端差", 10); set_w("数字以外一致", 12)
                    set_w("score", 8, num_fmt_3)

                else:
                    ws = writer.sheets[sheet_main]
                    try:
                        from openpyxl.utils import get_column_letter
                        last_row = max(1, nrows + 1)
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
        """表示中テーブルのチェック済み a/b を PDF にマーキング（既存ロジック準拠）"""
        if not self.files:
            QMessageBox.warning(self, "注意", "PDFを指定してください。")
            return
        df = getattr(self.model_unified, "_df", pd.DataFrame())
        if df.empty:
            QMessageBox.information(self, "情報", "マーキング対象がありません。")
            return

        # 選択抽出
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
            QMessageBox.information(self, "情報", "マーキング対象が選択されていません。")
            return

        # 重複整理（a優先）
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

        # 検索の堅牢化（元実装を簡約移植）
        CONN_CHARS = "ー-－–—・/／_＿"
        SPACE_CHARS = {" ", "\u00A0", "\u3000"}

        def nfkc(s: str) -> str:
            return unicodedata.normalize("NFKC", s)

        def hira_to_kata(s: str) -> str:
            out = []
            for ch in s:
                o = ord(ch)
                out.append(chr(o + 0x60) if 0x3041 <= o <= 0x3096 else ch)
            return "".join(out)

        def kana_norm_variants(s: str) -> Set[str]:
            base = nfkc(s or ""); k = hira_to_kata(base)
            return {k, k.replace("ー", "")}

        def base_variants(s: str) -> Set[str]:
            vs = set(); s0 = s or ""; s1 = nfkc(s0)
            for cand in (s0, s1):
                if not cand: continue
                vs.add(cand); vs.add(cand.replace(" ", "")); vs.add(cand.replace(" ", "\u00A0")); vs.add(cand.replace(" ", "\u3000"))
                for ch in list(CONN_CHARS):
                    if ch in cand:
                        vs.add(cand.replace(ch, ""))
                        vs.add(cand.replace(ch, " "))
                        vs.add(cand.replace(ch, "\u00A0"))
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
                c = hira_to_kata(nfkc(c))
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
            chars, (stream_keep, keep_idxmap), (stream_noch, noch_idxmap), _ = norm_stream_chars(page)
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

        def suppress_overlap_hits(hits, iou_th=0.60):
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
                        rects = suppress_overlap_hits(rects)
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

    # ===== [NEW] プレビュー更新: 選択インデックスから a/b を拾って差分を描画 =====
    def _show_inline_diff_for_index(self, proxy_index: QModelIndex):
        try:
            if not proxy_index or not proxy_index.isValid():
                return
            # Proxy -> Source にマップ
            src_index = self.proxy_unified.mapToSource(proxy_index)
            row = src_index.row()

            # UnifiedModel 側の DataFrame から a/b を取得
            df = self.model_unified._df  # 既存実装に倣い内部DFへアクセス
            if df is None or df.empty:
                return
            if "a" not in df.columns or "b" not in df.columns:
                return

            a_txt = "" if pd.isna(df.at[row, "a"]) else str(df.at[row, "a"])
            b_txt = "" if pd.isna(df.at[row, "b"]) else str(df.at[row, "b"])

            html = build_vertical_diff_html_embed(a_txt, b_txt)
            self.diff_view.setHtml(html)
        except Exception:
            pass

    # ===== 差し替え: 左プレビューを統合インライン差分に切り替え =====
    def on_unified_selection_changed(self, selected, deselected):
        try:
            sel = self.view_unified.selectionModel().selectedRows()
        except Exception:
            sel = []
        if not sel:
            # 何も選択されていないときのプレースホルダ
            self.diff_view.setHtml(
                "<html><body style='font-family:Yu Gothic UI, Noto Sans JP, sans-serif;"
                "color:#666; font-size:12px; margin:4px 6px;'>"
                "表のセルを選択すると、<b>統合インライン差分</b>をここに表示します。"
                "</body></html>"
            )
            return

        proxy_index = sel[0]
        src_index = self.proxy_unified.mapToSource(proxy_index)
        row = src_index.row()

        df = getattr(self.model_unified, "_df", pd.DataFrame())
        if df.empty or row < 0 or row >= len(df):
            return

        a = df.at[row, "a"] if "a" in df.columns else ""
        b = df.at[row, "b"] if "b" in df.columns else ""

        html = build_unified_inline_diff_embed(
            "" if a is None else str(a),
            "" if b is None else str(b)
        )
        self.diff_view.setHtml(html)
    # ===== 差し替えここまで =====


def main():
    app = QApplication(sys.argv)
    apply_light_theme(app, base_font_size=10)
    win = MainWindow(); win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()  # ★ 追加：凍結exeでの子プロセス起動対策
    main()

