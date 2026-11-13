# Nhật ký tiền xử lý: Project 1 (UIT-VSFC), Tuần 6

## 1. Đầu ra
- `src/preprocess.py`: module dùng lại được (`clean_text`, `segment`).
- `data/data_clean.csv`: 16175 câu (train 11426 / dev 1583 / test 3166), có 3 cột văn bản `text_raw`, `text_clean`, `text_seg` và các cột nhãn.
- `data/data_all.csv` giữ nguyên, không bị ghi đè. Nhãn và cách chia tập không đổi (đã kiểm tra bằng assert).
- Số câu bị `clean_text` thay đổi: 579 (3.6%).

## 2. Bảng quyết định
| Việc | Quyết định | Căn cứ (số liệu thật) |
|---|---|---|
| Unicode | NFC, giữ trong `clean_text` để demo nhận chữ gõ mới | Đã chuẩn hóa NFC ở notebook gộp dữ liệu, 0 câu thay đổi |
| Viết tắt | Chuẩn hóa `ko`→không, `đc`→được, `trc`→trước | Quét 16 dạng phổ biến, chỉ có {'ko': 1, 'đc': 3, 'trc': 1, 'k': 0, 'khum': 0, 'hông': 0, 'hok': 0, 'og': 0, 'ổng': 0, 'bả': 0, 'mik': 0, 'mềnh': 0, 'mk': 0, 'mn': 0, 'mng': 0, 'g': 0, 'j': 0, 'zì': 0, 'b': 0, 'bn': 0, 't': 0, 'm': 0, 'bit': 0, 'bít': 0, 'đg': 0, 'dc': 0, 'rùi': 0, 'ròi': 0, 'ùi': 0, 'r': 0, 'đượt': 0, 'pt': 0, 'bt': 0, 'bth': 0, 'wa': 0, 'qá': 0, 'nhìu': 2, 'nhều': 0, 'vs': 0, 'thik': 0, 'thít': 0, 'lm': 0, 'h': 0, 'đ': 0, 'vnd': 0, 'ib': 0, 'inbox': 0, 'add': 1, 'rep': 3, 'kcj': 0, 'kcg': 0} (tổng 11 lần) |
| Từ mượn | Giữ nguyên `ok`, `nope`, `good` | Không phải viết tắt (`ok`: 20 lần) |
| Dấu thanh | Đổi kiểu cũ sang kiểu mới (hòa→hoà, thủy→thuỷ) | 287 lượt từ viết kiểu cũ; chiều này không làm hỏng từ hợp lệ như toàn, quý, hoạt |
| Emoticon | Gom thành token chữ cái liền: {'emocontemn': 11, 'emolove': 14, 'emoother': 1, 'emosad': 16, 'emosmile': 77, 'emosurprise': 6} | README thay emoticon bằng chữ; số lượng nhỏ, gom theo ý nghĩa |
| Ký hiệu | Trả về `...`, `/`, `v.v`, `:`; giờ giấc `11doubledot55`→`11:55` | Số lần đứng riêng: {'dotdotdot': 6, 'fraction': 33, 'vdotv': 12, 'doubledot': 123}; 2 từ giờ giấc dính liền số |
| Khoảng trắng | Gộp nhiều khoảng trắng thành một | Không câu rỗng sau xử lý |
| Tách từ | VnCoreNLP, ghi vào cột `text_seg` | PhoBERT huấn luyện trên văn bản tách bằng VnCoreNLP |
| Dữ liệu gốc | Giữ nguyên cột `text_raw` | Để làm ablation (có/không chuẩn hóa) ở tuần sau |

## 3. Phát hiện khi thử nghiệm với VnCoreNLP
- `emo_smile` bị tách thành `emo _ smile`, nên đổi sang `emosmile` (chữ cái liền).
- `c#` bị tách thành `c #`, nên giữ nguyên `csharp`. README ghi `cshrap` nhưng dữ liệu thật dùng `csharp` (5 lần, `cshrap` 0 lần).
- `>>` bị tách thành `> >`, nên giữ nguyên `coloncolon` (1 lần).
- Sau khi đổi, các token tự đặt giữ nguyên số lần xuất hiện trước và sau khi tách từ; `text_seg` chỉ khác `text_clean` ở dấu `_`.

## 4. Kết quả tách từ
- Số tiếng trung bình 14.2 giảm còn 11.1 token; p99 từ 52 còn 42; tối đa từ 161 còn 134.
- Từ ghép chiếm 27.7% số token.

## 5. Hạn chế đã biết
- Tách từ có sai sót cục bộ, ví dụ `nói_khó nghe` (bộ tách từ ghép nhầm `nói_khó`). Chấp nhận vì dùng cùng công cụ với PhoBERT; không sửa tay.
- Có 1 câu trùng nội dung nhưng khác nhãn (negative và positive); giữ nguyên vì chỉ 1 câu.
- Chỉ quét 16 dạng viết tắt phổ biến, nên chỉ kết luận "không thấy viết tắt phổ biến khác trong danh sách đã quét".

## 6. Việc để lại
- Đo lại `max_length` bằng tokenizer PhoBERT trên cột `text_seg` (số token ở trên là số từ, chưa phải subword).
- Ablation: có/không chuẩn hóa, có/không tách từ (dùng `text_raw`, `text_clean`, `text_seg`).
