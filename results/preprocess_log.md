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
| Viết tắt | Chuẩn hóa các dạng không mơ hồ (`ko`→không, `đc`→được, `trc`→trước, ...) | Quét 51 dạng, chỉ gặp {'ko': 1, 'đc': 3, 'trc': 1, 'nhìu': 2, 'add': 1, 'rep': 3} (tổng 11 lần) |
| Từ mượn | Giữ nguyên `ok`, `nope`, `good` | Không phải viết tắt (`ok`: 20 lần) |
| Dấu thanh | Đổi kiểu cũ sang kiểu mới (hòa→hoà, thủy→thuỷ) | 287 lượt từ viết kiểu cũ; chiều này không làm hỏng từ hợp lệ như toàn, quý, hoạt |
| Emoticon | Gom thành token chữ cái liền: {'emocontemn': 11, 'emolove': 14, 'emoother': 1, 'emosad': 16, 'emosmile': 77, 'emosurprise': 6} | README thay emoticon bằng chữ; số lượng nhỏ, gom theo ý nghĩa |
| Ký hiệu | Trả về `...`, `/`, `v.v`, `:`; giờ giấc `11doubledot55`→`11:55` | Số lần đứng riêng: {'dotdotdot': 6, 'fraction': 33, 'vdotv': 12, 'doubledot': 123}; 2 từ giờ giấc dính liền số |
| Khoảng trắng | Gộp nhiều khoảng trắng thành một | Không câu rỗng sau xử lý |
| Tách từ | VnCoreNLP, ghi vào cột `text_seg` | PhoBERT huấn luyện trên văn bản tách bằng VnCoreNLP |
| Độ dài đầu vào | PhoBERT `max_length=128`; BiLSTM `64` | Đo trên `text_seg`, 16175 câu: p99 = 45 subword; 128 cắt 0.01% câu, 64 cắt 0.21% câu |
| Dữ liệu gốc | Giữ nguyên cột `text_raw` | Để làm ablation (có/không chuẩn hóa) ở tuần sau |

## 3. Phát hiện khi thử nghiệm với VnCoreNLP
- `emo_smile` bị tách thành `emo _ smile`, nên đổi sang `emosmile` (chữ cái liền).
- `c#` bị tách thành `c #`, nên giữ nguyên `csharp`. README ghi `cshrap` nhưng dữ liệu thật dùng `csharp` (5 lần, `cshrap` 0 lần).
- `>>` bị tách thành `> >`, nên giữ nguyên `coloncolon` (1 lần).
- Sau khi đổi, các token tự đặt giữ nguyên số lần xuất hiện trước và sau khi tách từ; `text_seg` chỉ khác `text_clean` ở dấu `_`.

## 4. Kết quả tách từ
- Số tiếng trung bình 14.2 giảm còn 11.1 token; p99 từ 52 còn 42; tối đa từ 161 còn 134.
- Từ ghép chiếm 27.7% số token.
- Đây là số từ theo VnCoreNLP; số subword của PhoBERT xem dòng "Độ dài đầu vào" ở bảng quyết định.

## 5. Hạn chế đã biết
- Tách từ có sai sót cục bộ, ví dụ `nói_khó nghe` (bộ tách từ ghép nhầm `nói_khó`). Chấp nhận vì dùng cùng công cụ với PhoBERT; không sửa tay.
- Có 1 câu trùng nội dung nhưng khác nhãn (negative và positive); giữ nguyên vì chỉ 1 câu.
- Chỉ quét 51 dạng viết tắt đã liệt kê, nên chỉ kết luận "không thấy viết tắt khác trong danh sách đã quét".
- Dấu thanh được thống nhất về kiểu mới (hoà, thuỷ) theo kiểu chiếm đa số trong dữ liệu. PhoBERT được huấn luyện chủ yếu trên văn bản kiểu cũ nên một số từ có thể bị cắt thành nhiều subword hơn; chưa đo ảnh hưởng, kiểm tra bằng ablation `text_raw` và `text_clean`.

## 6. Việc để lại
- Ablation: có/không chuẩn hóa, có/không tách từ (dùng `text_raw`, `text_clean`, `text_seg`).
