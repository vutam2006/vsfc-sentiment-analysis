# Nhật ký baseline ML truyền thống: Project 1 (UIT-VSFC)

## 1. Thiết lập
- Dữ liệu: `data_clean.csv`, train 11426 / dev 1583 / test 3166; neutral: 458 / 73 / 167 câu.
- Quy tắc: chọn mô hình chỉ bằng dev và cross-validation 5 fold trên train; test chấm một lần (ô 8).
- Metric chính Macro-F1, kèm Precision / Recall / F1 từng lớp và ma trận nhầm lẫn. Seed = 42.

## 2. Các quyết định (dev, Macro-F1)
| Bước | Kết quả | Quyết định |
|---|---|---|
| Class weight (ô 3-4) | LR 0.7023 → 0.7182; SVM 0.7072 → 0.7604 | Dùng trọng số lớp: không dùng thì Recall neutral chỉ ~0,15-0,18 |
| Bộ phân loại, unigram (ô 4) | SVM-bal 0.7604; LR-bal 0.7182; NB 0.6705 | SVM dẫn đầu; NB làm mốc thấp |
| Đặc trưng, SVM-bal (ô 5) | word(1,1) 0.7604; word(1,2) 0.7723; word(1,3) 0.7627; char_wb(2,5) 0.7648; word(1,2)+char_wb(2,5) 0.7834 | Chọn `word(1,2)+char_wb(2,5)`: tốt nhất ở cả SVM và LR; trigram không giúp |
| text_clean và text_seg (ô 6) | Chênh (seg − clean) từ -0.0099 đến +0.0148, hướng không nhất quán; KTC bootstrap của SVM đều chứa 0 | Giữ `text_clean` (không cần VnCoreNLP khi triển khai) |
| Tinh chỉnh (ô 7) | Lưới SVM trên dev 0.7556-0.7890; CV của mặc định = 0.7640 | Giữ mặc định (`C`=1.0, `balanced`): xếp đầu theo CV, các ô lưới trên dev chỉ là nhiễu |

## 3. Kết quả trên test
| model | macro_f1 | accuracy | F1_negative | F1_neutral | F1_positive | P_neutral | R_neutral |
|---|---|---|---|---|---|---|---|
| SVM (chính) | 0.751 | 0.895 | 0.913 | 0.418 | 0.921 | 0.500 | 0.359 |
| LR | 0.733 | 0.887 | 0.906 | 0.373 | 0.918 | 0.421 | 0.335 |
| NB | 0.652 | 0.815 | 0.834 | 0.258 | 0.865 | 0.234 | 0.287 |

- Mô hình chính SVM (chính): Macro-F1 = 0.7509, KTC 95% (bootstrap) = [0.7248, 0.7748]; dev = 0.7834, CV = 0.7640.
- SVM − LR trên test: +0.0183, KTC 95% [+0.0095, +0.0291], P(SVM > LR) = 1.00.
- Ma trận nhầm lẫn: `fig_cm_ml_best_test.png`. Các cặp nhầm nhiều nhất: positive→negative: 109; negative→positive: 57; neutral→negative: 57; neutral→positive: 50.

## 4. Phân tích lỗi (test, mô hình chính)
- Câu có từ nối đối lập: 139 câu, accuracy 0.806; còn lại: 3027 câu, accuracy 0.899.
- Theo độ dài câu:

| do_dai | n | acc | R_neutral |
|---|---|---|---|
| <=5 | 310 | 0.845 | 0.589 |
| 6-10 | 1133 | 0.909 | 0.352 |
| 11-20 | 1155 | 0.893 | 0.156 |
| >20 | 568 | 0.898 | 0.083 |

- Lớp neutral: Recall 0.359, Precision 0.500; 107 trên 167 câu neutral bị đoán sai (file `ml_neutral_errors_to_label.csv` để phân loại nguyên nhân).

## 5. Hạn chế
- Neutral dev chỉ 73 câu: điểm dev lạc quan (đã dùng để chọn đặc trưng) và dao động mạnh; điểm CV và test là ước lượng gần thực tế hơn.
- Test cũng chỉ có 167 câu neutral; KTC bootstrap chỉ phản ánh sai số lấy mẫu của test, chưa tính biến thiên do huấn luyện.
- Chỉ thử các bộ phân loại tuyến tính và Naive Bayes, đặc trưng TF-IDF; chưa dùng embedding.
- Nhãn neutral có phần mơ hồ (ví dụ câu cảm ơn bị gán nhãn không nhất quán), nên một phần trần của mô hình đến từ nhãn.

## 6. Việc để lại
- Huấn luyện BiLSTM và PhoBERT trên cùng split, dùng cùng metric và cùng cách báo KTC để so sánh công bằng với bảng ở mục 3.
- Phân loại tay 50 câu neutral bị sai để mô tả nguyên nhân lỗi trong báo cáo.
