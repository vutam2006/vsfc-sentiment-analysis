"""Tiền xử lý UIT-VSFC: dùng lại được cho notebook, huấn luyện và demo."""
import re
import unicodedata

ABBR = {
    "ko": "không", "đc": "được", "trc": "trước",
    "k": "không", "khum": "không", "hông": "không", "hok": "không", "og": "không",
    "ổng": "ông ấy", "bả": "bà ấy",
   
    "mik": "mình", "mềnh": "mình", "mk": "mình",
    "mn": "mọi người", "mng": "mọi người",
    "g": "gì", "j": "gì", "zì": "gì",
    "b": "bạn", "bn": "bạn",
    "t": "tao", "m": "mày",
    
    "bit": "biết", "bít": "biết",
    "đg": "đang",
    "dc": "được", "đc": "được",
    "rùi": "rồi", "ròi": "rồi", "ùi": "rồi", "r": "rồi",
    "đượt": "được",
    "pt": "phát triển", "bt": "biết",
    
    # Tính từ & trạng thái cảm xúc, đánh giá
    "bth": "bình thường",
    "wa": "quá", "qá": "quá",
    "nhìu": "nhiều", "nhều": "nhiều",
    "vs": "với",
    "thik": "thích", "thít": "thích",
    "lm": "lắm",
    
    # Ký hiệu viết tắt tốc độ cao
    "h": "giờ",
    "đ": "đồng", "vnd": "đồng",
    "ib": "nhắn tin", "inbox": "nhắn tin",
    "add": "thêm", "rep": "phản hồi",
    "kcj": "không có gì", "kcg": "không có gì",
}

# 2. Emoticon bị README thay bằng chữ -> gom theo ý nghĩa.
#    Token mới là chữ cái liền (không dùng '_'), vì VnCoreNLP tách 'emo_smile' thành 'emo _ smile'.
EMO = {
    "colonsmile": "emosmile", "colonsmilesmile": "emosmile",
    "colonbigsmile": "emosmile", "colonsmallsmile": "emosmile",
    "colonhihi": "emosmile",
    "colonlove": "emolove", "colonlovelove": "emolove",
    "colonsad": "emosad", "colonsadcolon": "emosad",
    "colonsurprise": "emosurprise", "colondoublesurprise": "emosurprise",
    "coloncontemn": "emocontemn",
    "coloncc": "emoother",
}

# 3. Ký hiệu bị README thay bằng chữ -> trả lại ký hiệu gốc (chỉ những ký hiệu
#    VnCoreNLP KHÔNG làm vỡ). Giữ nguyên dạng chữ cái:
#      'csharp'     (README ghi 'cshrap' là sai; 'c#' bị tách thành 'c #')
#      'coloncolon' ('>>' bị tách thành '> >', chỉ 1 lần trong dữ liệu)
SYMBOLS = {"dotdotdot": "...", "fraction": "/", "vdotv": "v.v",
           "doubledot": ":"}

_MAP = {**ABBR, **EMO, **SYMBOLS}
# Chuỗi dài xếp trước để không ăn nhầm tiền tố (colonsmile vs colonsmilesmile)
_PATTERN = re.compile(
    r"(?<!\w)(" + "|".join(sorted(map(re.escape, _MAP), key=len, reverse=True)) + r")(?!\w)"
)
# Giờ giấc: '11doubledot55' -> '11:55' (dính liền số nên regex trên không bắt được)
_TIME = re.compile(r"(?<=\d)doubledot(?=\d)")

# 4. Dấu thanh: kiểu cũ -> kiểu mới (hòa -> hoà, thủy -> thuỷ).
#    Chiều này an toàn: âm tiết đóng (toàn, khoản, hoạt) không bao giờ viết kiểu cũ.
TONE_MAP = {
    "òa": "oà", "óa": "oá", "ỏa": "oả", "õa": "oã", "ọa": "oạ",
    "òe": "oè", "óe": "oé", "ỏe": "oẻ", "õe": "oẽ", "ọe": "oẹ",
    "ùy": "uỳ", "úy": "uý", "ủy": "uỷ", "ũy": "uỹ", "ụy": "uỵ",
}

def unify_tone(s):
    for old, new in TONE_MAP.items():
        s = s.replace(old, new)
    return s

def clean_text(s):
    s = unicodedata.normalize("NFC", str(s)).lower()
    s = _TIME.sub(":", s)
    s = unify_tone(s)
    s = _PATTERN.sub(lambda m: _MAP[m.group(1)], s)
    return re.sub(r"\s+", " ", s).strip()

_seg = None
def segment(s, save_dir="/content/vncorenlp"):
    """Tách từ bằng VnCoreNLP (tải model một lần duy nhất)."""
    global _seg
    if _seg is None:
        import os, py_vncorenlp
        os.makedirs(save_dir, exist_ok=True)
        if not os.path.exists(f"{save_dir}/models"):
            py_vncorenlp.download_model(save_dir=save_dir)
        _seg = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=save_dir)
    return " ".join(_seg.word_segment(s)) if s else ""
