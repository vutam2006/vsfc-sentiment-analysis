# vsfc-sentiment-analysis

## Phân tích cảm xúc phản hồi sinh viên tiếng Việt (UIT-VSFC)

Project 1 - Vũ Huy Tâm - 202416347
Giảng viên hướng dẫn: Đỗ Tuấn Anh

## Mục tiêu

Phân loại câu phản hồi của sinh viên thành 3 lớp: tiêu cực / trung tính / tích cực, và so sánh ba hướng tiếp cận:

- Mô hình ML truyền thống: TF-IDF + Logistic Regression / Linear SVM / Naive Bayes
- BiLSTM
- PhoBERT (`vinai/phobert-base`), chưa thực hiện

Metric chính: **Macro-F1**, kèm Precision / Recall / F1 từng lớp và ma trận nhầm lẫn. Dùng Macro-F1 vì lớp trung tính chỉ chiếm khoảng 4%, nên Accuracy dễ cho điểm cao dù mô hình gần như bỏ qua lớp này.

## Kết quả trên tập test

Mô hình được chọn và cấu hình được cố định bằng tập dev. Tập test chỉ chấm một lần.

| Mô hình | Macro-F1 | Accuracy | F1 tiêu cực | F1 trung tính | F1 tích cực |
|---|---|---|---|---|---|
| TF-IDF + SVM (class_weight balanced) | 0,7399 | 0,8756 | 0,8971 | 0,4119 | 0,9108 |
| BiLSTM (trung bình ± độ lệch chuẩn, 5 seed) | 0,7637 ± 0,0068 | 0,8923 ± 0,0080 | 0,9127 ± 0,0063 | 0,4545 ± 0,0119 | 0,9239 ± 0,0030 |
| PhoBERT | chưa chạy | | | | |

Nhận xét:

- BiLSTM cao hơn SVM 0,0238 Macro-F1, và cả 5 seed (thấp nhất 0,7544) đều cao hơn SVM. Mức chênh này nhỏ và sát ngưỡng 0,02 đặt trước khi xem kết quả; SVM là baseline chưa tinh chỉnh (C = 1, chỉ unigram) nên chưa nên kết luận mạnh hơn "BiLSTM nhỉnh hơn nhẹ".
- Hai lớp tiêu cực và tích cực đạt F1 khoảng 0,90 đến 0,92 ở cả hai mô hình. Điểm yếu nằm ở lớp trung tính: F1 chỉ khoảng 0,41 đến 0,45; SVM đoán sai 91/167 câu trung tính, BiLSTM sai trung bình 85,6/167 câu.
- Lớp trung tính chỉ có 167 câu ở test, nên F1 lớp này dao động mạnh. Chênh lệch Macro-F1 dưới 0,02 giữa hai mô hình nên được đọc là "tương đương".

Ma trận nhầm lẫn: `results/fig_cm_ml_best_test.png` (SVM) và `results/fig_cm_bilstm_test.png` (BiLSTM, trung bình 5 seed). Bảng chi tiết: `results/ml_test_metrics.csv`, `results/bilstm_test_metrics.csv`, `results/compare_ml_bilstm_test.csv`.

## Dữ liệu

UIT-VSFC (Vietnamese Students' Feedback Corpus): 16.175 câu phản hồi của sinh viên, đã chia sẵn train / dev / test. Project chỉ dùng **nhãn cảm xúc** (0 = tiêu cực, 1 = trung tính, 2 = tích cực), không dùng nhãn chủ đề.

| Tập | Tiêu cực | Trung tính | Tích cực | Tổng |
|---|---|---|---|---|
| train | 5.325 (46,6%) | 458 (4,0%) | 5.643 (49,4%) | 11.426 |
| dev | 705 (44,5%) | 73 (4,6%) | 805 (50,9%) | 1.583 |
| test | 1.409 (44,5%) | 167 (5,3%) | 1.590 (50,2%) | 3.166 |
| **Tổng** | 7.439 (46,0%) | 698 (4,3%) | 8.038 (49,7%) | 16.175 |

Repo này không đăng lại dữ liệu, chỉ có notebook để tải về từ Google Drive. Dữ liệu thuộc về nhóm tác giả gốc.

## Quy trình và số liệu chính

### 02 - Khảo sát dữ liệu (sơ bộ)

- Không có giá trị thiếu hay câu rỗng. Chỉ có 1 câu trùng lặp mang hai nhãn khác nhau, và 0 câu trùng giữa train, dev, test (không rò rỉ dữ liệu).
- Độ dài câu (đếm theo token tách bằng khoảng trắng, gồm cả dấu câu): trung bình 14,2; trung vị 11; p95 = 33; p99 = 52; tối đa 161.
- Độ dài theo lớp: tiêu cực dài nhất (trung vị 13), tích cực 10, trung tính ngắn nhất (trung vị 8).
- Có 217 câu (1,3%) không quá 3 token, một số có nhãn khó đoán (ví dụ câu chỉ gồm "đầy đủ ."), cho thấy dữ liệu có nhiễu nhãn tự nhiên.
- Khoảng 4,3% câu có từ nối đối lập (nhưng, tuy nhiên, ...); trong nhóm này 76,0% là tiêu cực, so với 44,6% ở các câu còn lại.
- Gần như không có viết tắt kiểu teencode (ví dụ `ko` 1 lần, `đc` 3 lần, `trc` 1 lần). Emoticon đã được tác giả gốc thay bằng chữ (nhóm `colon...`), tổng cộng 125 lần xuất hiện.

### 03 - Tiền xử lý

Tạo `data_clean.csv` với các cột `split, text_raw, text_clean, text_seg, sentiment, sentiment_name`. Không bỏ câu nào (vẫn 16.175 câu).

- Chuẩn hóa Unicode NFC, viết thường, khôi phục giờ giấc dính liền (`11doubledot55` thành `11:55`).
- Thống nhất cách đặt dấu thanh (kiểu `hòa` sang kiểu `hoà`), có kiểm tra không làm hỏng các từ hợp lệ như `toàn`, `quý`.
- Gộp 13 loại emoticon thành 6 token (`emosmile`, `emosad`, `emolove`, `emosurprise`, `emocontemn`, `emoother`); tổng 125 lần xuất hiện được giữ nguyên, không mất token nào.
- Khôi phục ký hiệu bị thay chữ: `doubledot` thành `:`, `fraction` thành `/`, `vdotv` thành `v.v`, `dotdotdot` thành `...`. Dữ liệu thực dùng `csharp` (5 lần) chứ không phải `cshrap` như README của bộ dữ liệu.
- Không bỏ stopword, để giữ các từ phủ định ("không", "chưa") và từ chỉ mức độ.
- `text_clean`: văn bản đã làm sạch, mức tiếng (chưa tách từ), dùng cho TF-IDF và BiLSTM. `text_seg`: tách từ bằng VnCoreNLP (nối bằng dấu `_`), dành cho PhoBERT. Tách từ chỉ thêm dấu `_`, không đổi chữ nào (đã kiểm tra bằng `assert`).
- Sau tách từ, độ dài trung bình còn 11,1 từ (p99 = 42, tối đa 134); 27,7% token là từ ghép.
- Độ dài theo tokenizer PhoBERT (đã gồm `<s>`, `</s>`): trung bình 13,4; p99 = 45; tối đa 136. Với `max_length = 64` cắt 0,21% câu, với 128 cắt 0,01%.

### 04 - Baseline ML

TF-IDF unigram trên `text_clean` (`min_df = 2`, `sublinear_tf`, từ vựng 1.484 từ), bọc trong `Pipeline` và chỉ `fit` trên train. Chọn mô hình theo Macro-F1 trên dev:

| Cấu hình | Macro-F1 (dev) | Accuracy | F1 trung tính |
|---|---|---|---|
| SVM (balanced) | **0,7604** | 0,8958 | 0,4358 |
| LR (balanced) | 0,7182 | 0,8490 | 0,3704 |
| SVM | 0,7072 | 0,9059 | 0,2708 |
| LR | 0,7023 | 0,9078 | 0,2558 |
| MultinomialNB (fit_prior=False) | 0,6705 | 0,8351 | 0,2822 |
| ComplementNB | 0,6083 | 0,8553 | 0,0732 |
| MultinomialNB | 0,5863 | 0,8591 | 0,0000 |

Nhận xét: không dùng trọng số lớp thì mô hình có Accuracy cao (0,908) nhưng gần như bỏ sót lớp trung tính (LR chỉ bắt được 11/73 câu trung tính trên dev). `class_weight = 'balanced'` cải thiện rõ Macro-F1. Mô hình được chọn là SVM balanced; kết quả test ở bảng đầu README.

### 05 - BiLSTM

- Từ vựng chỉ lấy từ train (`text_clean`, tần suất ít nhất 2): 1.526 mục, gồm `<PAD>` và `<UNK>`. Độ dài tối đa 128 token, đệm động theo batch (64 câu).
- Kiến trúc: Embedding 128 chiều (khởi tạo ngẫu nhiên) → BiLSTM 1 lớp (hidden 128 mỗi chiều) → max-pooling theo thời gian (bỏ qua vị trí đệm) → Dropout 0,5 → Linear 3 lớp. Tổng 460.291 tham số.
- Huấn luyện: cross-entropy có trọng số lớp `n / (3 × số câu của lớp)` = 0,715 / 8,316 / 0,675 (tương đương `balanced`); Adam, learning rate 1e-3, cắt gradient ở 1,0; tối đa 40 epoch, dừng sớm theo Macro-F1 trên dev với patience 6, nạp lại trọng số epoch tốt nhất.
- Chạy 5 seed (42 đến 46). Kết quả tái lập được (seed 42 chạy lại cho đúng dev Macro-F1 = 0,7986, epoch tốt nhất = 16).

| Seed | Epoch tốt nhất | Macro-F1 (dev) | Macro-F1 (test) |
|---|---|---|---|
| 42 | 16 | 0,7986 | 0,7544 |
| 43 | 15 | 0,8049 | 0,7621 |
| 44 | 25 | 0,8043 | 0,7718 |
| 45 | 23 | 0,8102 | 0,7687 |
| 46 | 18 | 0,8166 | 0,7616 |
| **TB ± std** | | 0,8069 ± 0,0068 | 0,7637 ± 0,0068 |

Điểm dev cao hơn test vì epoch được chọn theo chính dev và lớp trung tính ở dev chỉ có 73 câu; vì vậy so sánh giữa các mô hình dùng điểm test.

Precision / Recall / F1 từng lớp trên test (trung bình 5 seed):

| Lớp | Precision | Recall | F1 |
|---|---|---|---|
| Tiêu cực | 0,9129 | 0,9127 | 0,9127 |
| Trung tính | 0,4363 | 0,4874 | 0,4545 |
| Tích cực | 0,9314 | 0,9167 | 0,9239 |

## Hạn chế

- Lớp trung tính quá ít (698 câu, 4,3%): F1 lớp này thấp và dao động mạnh, nhất là ở dev (73 câu) và test (167 câu).
- Độ lệch chuẩn giữa các seed (khoảng 0,007) chỉ phản ánh dao động do khởi tạo, chưa phản ánh sai số do kích thước tập test. Mỗi cấu hình chỉ chấm trên một lần chia train / dev / test cố định.
- Baseline ML chưa được tinh chỉnh (chỉ unigram, tham số mặc định); BiLSTM chọn epoch bằng dev. So sánh giữa hai mô hình vì thế mang tính tham khảo.
- Dữ liệu có nhiễu nhãn (câu rất ngắn, câu chứa từ nối đối lập). Đây là giới hạn của bộ dữ liệu, không phải của mô hình.
- Bảng viết tắt trong `src/preprocess.py` rộng hơn nhu cầu của dữ liệu này (dữ liệu gần như không có viết tắt). Nếu dùng hàm này cho văn bản mới, cần rà lại bảng viết tắt trước.

## Cấu trúc repo

```
vsfc-sentiment-analysis/
├── README.md
├── notebooks/
│   ├── 01_check_env.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocess.ipynb
│   ├── 04_ml_baseline.ipynb
│   └── 05_bilstm.ipynb
├── src/
│   └── preprocess.py
└── results/
    ├── fig_sentiment_dist.png
    ├── fig_length_hist.png
    ├── fig_length_by_sentiment.png
    ├── preprocess_log.md
    ├── ml_baseline_log.md
    ├── ml_test_metrics.csv
    ├── fig_cm_ml_best_test.png
    ├── bilstm_test_metrics.csv
    ├── compare_ml_bilstm_test.csv
    └── fig_cm_bilstm_test.png
```

## Danh sách notebook

| Notebook | Nội dung | Trạng thái |
|---|---|---|
| `01_check_env.ipynb` | Cài môi trường, tải dữ liệu từ Google Drive | Xong |
| `02_eda.ipynb` | Khảo sát dữ liệu: phân bố nhãn, độ dài câu, từ đặc trưng theo lớp, câu có từ nối đối lập | Xong (khảo sát sơ bộ) |
| `03_preprocess.ipynb` | Tiền xử lý, tách từ VnCoreNLP, tạo `data_clean.csv` | Xong |
| `04_ml_baseline.ipynb` | Baseline ML: TF-IDF + LR / SVM / NB, chọn mô hình bằng dev, chấm test một lần | Xong |
| `05_bilstm.ipynb` | BiLSTM, 5 seed, so sánh với SVM trên test | Xong |
| `06_phobert.ipynb` | Fine-tune PhoBERT | Chưa |

## Cách chạy

Các notebook viết cho Google Colab, lưu dữ liệu và kết quả trong Google Drive tại `MyDrive/project1_vsfc/` (thư mục `data`, `notebooks`, `src`, `results`).

1. Chạy lần lượt `01` đến `05`; mỗi notebook đọc đầu ra của notebook trước (`data_all.csv`, `data_clean.csv`, `ml_test_metrics.csv`).
2. `01` cài thư viện và tải dữ liệu. `03` cần Java và `py_vncorenlp` để tách từ (Colab có sẵn Java; mô hình VnCoreNLP được tải tự động mỗi phiên).
3. `05` nên chạy với GPU (Runtime → Change runtime type → GPU).
4. Thư viện chính: `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `torch`, `transformers`, `py_vncorenlp`, `underthesea`, `gdown`.

## Tài liệu tham khảo

- Bộ dữ liệu UIT-VSFC: https://ieeexplore.ieee.org/document/8573337
- PhoBERT: https://github.com/VinAIResearch/PhoBERT
- VnCoreNLP: https://github.com/vncorenlp/VnCoreNLP
