# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603711
- **Họ và tên:** Vũ Nhật Minh
- **SEED** (in ở ô đầu notebook): 2211

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | |

## 2. Kết quả chính

| Phương án | Precision | Recall | F1 (test) | F1 (SUB 40) | p95 ms/tin | USD / 1k tin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg (tự code) | 0.987 | 0.812 | 0.891 | 0.824 | 0.065 | 0 |
| 2. BoW + MLP | 0.977 | 0.896 | 0.935 | 0.857 | 2.3 | 0 |
| 3a. Encoder (MiniLM) + kNN | 0.837 | 0.906 | 0.870 | 0.889 | 62.5 | 0 |
| 3b. Encoder (MiniLM) + LogReg | 0.967 | 0.917 | 0.941 | 0.919 | 7.2 | 0 |
| 3c. Encoder (MiniLM) + MLP head | 0.947 | 0.938 | 0.942 | 0.919 | 11.5 | 0 |
| 4a. LLM nhỏ zero-shot (gpt-6-luna) | 1.000 | 0.900 | — | 0.947 | 1673 | 0.006 |
| 4b. LLM lớn zero-shot (gpt-6.1-terra) | 1.000 | 0.900 | — | 0.947 | 1418 | 0.061 |
| 4c. LLM nhỏ few-shot | 1.000 | 0.900 | — | 0.947 | 1155 | 0.019 |

Hàng LLM chỉ đo trên SUB 40 tin. Chênh dưới ~0.05 F1 trên SUB coi như hoà (1 tin sai ≈ 3 điểm F1).

- **Đường Pareto (1 − F1 ↔ p95):** 1. BoW + LogReg → 2. BoW + MLP → 3b. Encoder + LogReg → 4c. LLM few-shot.
- **Đạt SLA p95 < 50 ms:** 1, 2, 3b, 3c. 3a (62.5 ms) và mọi LLM (> 1 giây) không đạt.

## 3. Dự đoán và thực tế

**(1) `ml_f1`: F1 của BoW + LogReg.**
- *Tôi đoán:* F1 khoảng 0.75, recall cao hơn precision vì spam có từ khoá dễ nhận biết nhưng dễ chặn nhầm tin thường.
- *Thực tế:* F1 = 0.891, precision 0.987 > recall 0.812, ngược với dự đoán.
- *Vì sao lệch:* dữ liệu chỉ có ~13% spam nên mô hình học được bias b âm, mặc định nghiêng về ham. Tin chỉ bị chặn khi có nhiều từ spam cộng dồn, nên gần như không chặn nhầm (precision cao). Ngược lại, spam viết giống tin bạn bè ("Oh my god! I've found your number again…") chứa nhiều từ kiểu ham (i, my, me) nên lọt qua (recall thấp hơn).

**(2) `llm_models`: model lớn và model nhỏ.**
- *Tôi đoán:* model lớn F1 cao hơn khoảng 0.05, đắt gấp ~10 lần, LLM không thắng rõ 3b.
- *Thực tế:* hai model bằng nhau (F1 SUB 0.947), cùng sai 2 tin; model lớn đắt gấp ~10 lần ($0.061 so với $0.006 / 1k tin). Few-shot cũng không tăng F1 mà tốn gấp ~3 lần.
- *Vì sao lệch:* bài phân loại spam/ham đủ dễ để model nhỏ làm tốt; model lớn có thêm năng lực nhưng bài này không cần dùng tới. Model to hơn, đắt hơn chưa chắc tốt hơn.

**(3) `hard_eval`: 50 case khó.**
- *Tôi đoán:* accuracy các model tụt còn khoảng 60–70%; LLM khó nhất ở nhóm spam giọng tự nhiên.
- *Thực tế:* BoW tụt sâu hơn tôi nghĩ (acc 52–54%, F1 0.33–0.38), còn LLM gần như không tụt (acc 98–100%), đúng 100% cả nhóm spam giọng tự nhiên.
- *Vì sao lệch:* tôi đánh giá thấp khả năng đọc hiểu ý đồ của LLM và đánh giá cao BoW. BoW chỉ nhìn từ khoá nên khi spam không dùng từ khoá thì gần như "mù".

## 4. Phân tích 50 case khó

| Model | F1 SUB | F1 50 case khó | Acc 50 case khó |
| --- | ---: | ---: | ---: |
| 1. BoW + LogReg | 0.824 | 0.333 | 0.52 |
| 2. BoW + MLP | 0.857 | 0.378 | 0.54 |
| 3a. Emb + kNN | 0.889 | 0.737 | 0.70 |
| 3b. Emb + LogReg | 0.919 | 0.640 | 0.64 |
| 3c. Emb + MLP head | 0.919 | 0.640 | 0.64 |
| 4a. LLM nhỏ zero-shot | 0.974 | 0.983 | 0.98 |
| 4d. LLM nhỏ + prompt phòng thủ | 0.974 | 1.000 | 1.00 |

- **Tụt nhiều nhất: BoW** (F1 giảm ~0.49). Khó nhất ở các nhóm "Spam giọng tự nhiên" (LogReg 10%, MLP 0%), "Né từ khoá" (17%, 0%) và "Thao túng / lừa đảo" (12%, 38%). BoW đạt 100% ở nhóm "Tưởng spam nhưng không phải" không phải vì hiểu, mà vì nó gần như đoán ham cho mọi tin.
- **Embedding** khó nhất ở "Spam giọng tự nhiên" (20–40%) và "Nói về spam" (kNN 25%, 3b/3c 50%): tin cảnh báo về lừa đảo có nghĩa rất gần tin lừa đảo thật.
- **LLM** gần như đúng hết; prompt phòng thủ (4d) không làm hỏng nhóm nào và nâng nhóm "Tưởng spam nhưng không phải" từ 90% lên 100%.

**Case 1, né từ khoá (#29):** *"C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd. Cl4im n0w at w1n-gift.c0m"*. BoW tách thành các token như `c0ngr4tulat10ns`, `h4ve`, `w0n` không có trong từ điển 2.000 từ, nên vector gần như toàn 0, điểm z ≈ b (âm) → đoán ham. Encoder chia thành sub-word và vẫn nhận ra nghĩa "chúc mừng trúng thưởng, nhận ngay", nên làm tốt hơn hẳn (3b đúng 100% nhóm này).

**Case 2, spam giọng tự nhiên (#11):** *"Hey it's Jess from the gym 😊 we're doing a members-only thing this week, 3 months for the price of 1, just reply JOIN and I'll sort it"*. Không có free/win/prize, lại có nhiều từ thân mật (hey, it's, I'll) mà BoW gán trọng số ham, nên bị đoán ham. Embedding cũng dễ sai vì về nghĩa nó gần tin nhắn bạn bè. Chỉ LLM, vốn đọc cả câu và hiểu ý đồ "khuyến mãi giả làm người quen", mới nhận ra là spam.

## 5. Thử thách cá nhân

**Đề #4 (theo SEED 2211):** chạy `knn_predict` với k = 1, 5, 25, báo F1, giải thích vì sao k quá nhỏ hoặc quá lớn đều có hại.

| k | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| 1 | 0.835 | 0.948 | **0.888** |
| 5 | 0.837 | 0.906 | 0.870 |
| 25 | 0.795 | 0.927 | 0.856 |

- **k = 1 cao nhất**, nhưng một phần là nhờ rò rỉ dữ liệu: còn nhiều spam mẫu chỉ khác số điện thoại nằm ở cả train và test, nên 1-NN chỉ cần tìm "bản sao gần giống". Về nguyên tắc k = 1 dễ bị nhiễu (1 tin gán nhãn sai là đủ làm sai), nên dễ bị lừa với case mới.
- **k = 25 thấp nhất**, chủ yếu do precision tụt (0.835 → 0.795). Spam tạo thành cụm dày trong không gian embedding, nên tin ham nằm gần cụm đó gom phải nhiều phiếu spam và bị chặn nhầm. k quá lớn làm ranh giới bị mờ (underfit).
- Chênh lệch chỉ ~0.03 F1 (vài tin trên 775). Muốn chọn k đúng cách phải thử trên tập validation, không phải test.

## 6. Phương án cho sản phẩm thật

**Yêu cầu:** 10 triệu SMS/ngày (~116 tin/giây), p95 < 50 ms/tin.

- **Loại LLM:** p95 1.2–1.7 giây, vượt SLA hơn 20 lần. Ngay cả LLM nhỏ cũng tốn ≈ $0.006/1k tin × 10.000k tin = **$60/ngày ≈ $1.800/tháng**.
- **Loại 3a (kNN):** p95 62.5 ms, vượt SLA; mỗi tin phải so với toàn bộ tập train nên càng nhiều dữ liệu càng chậm.
- **Chọn 3b. Encoder (MiniLM) + LogReg:** F1 test 0.941 (gần cao nhất), p95 7.2 ms (dư nhiều so với 50 ms), thông lượng ~1.200 tin/s trên một máy (gấp ~10 lần nhu cầu trung bình), chi phí API $0. Hơn BoW + LogReg (F1 0.891) chủ yếu ở recall (0.917 so với 0.812) và chịu được tin "ngụy trang" tốt hơn (50 case khó: 64% so với 52%).
- **Đánh đổi chấp nhận:** cần máy chạy encoder (CPU mạnh hoặc GPU) thay vì BoW gần như miễn phí; vẫn yếu ở spam giọng tự nhiên (20–30%).
- **Cải tiến (router):** 3b xử lý mọi tin; tin chắc chắn (p < 0.3 hoặc p > 0.7) quyết định ngay, chỉ vài % tin "lưỡng lự" mới gửi cho LLM nhỏ với prompt phòng thủ. Đa số tin vẫn nhanh và rẻ, chi phí LLM giảm hàng chục lần, mà tin khó vẫn được LLM xử lý.

## 7. AI log và điều đã học

### 🤖 AI log
| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| | | | |
| Chuẩn bị | Nhờ Claude (Claude Code) đọc repo và giải thích bài lab cần làm gì; hướng dẫn tạo repo, mở Colab, bật GPU | Tóm tắt các phần, luật, rubric, cách nộp; hướng dẫn từng bước thao tác | Tự làm theo, kiểm tra SEED=2211 và số liệu chia dữ liệu |
| 0 | Hỏi ý chính Câu hỏi 0.1 (data leakage) và cách làm Tính tay #1 | Giải thích tin trùng ở cả train/test làm điểm test lạc quan; công thức accuracy = ham/tổng | Tự viết lại câu trả lời, tự bấm máy 4516/5169 |
| 4 – TODO 4.1–4.4 | Nhờ Claude viết code call_openai, prompt, parse_label, llm_classify_batch (vùng được dùng AI) | Code đầy đủ kèm giải thích | Tôi đọc lại code trước khi gọi API, kiểm tra không có key trong code, chạy kiểm tra ✅ |
| 4 – debug key | Hỏi vì sao lỗi 401 và key OpenAI mới không được nhận | Giải thích ô cấu hình ưu tiên biến môi trường cũ; đưa code in key dạng che bớt và gọi thử | Phát hiện key cũ kẹt trong bộ nhớ; xoá biến môi trường, chạy lại cấu hình → kết nối OK |
| 4 – debug F1 = 0 | Hỏi vì sao 4.A/4.B ra F1 = 0 | Đưa ô chẩn đoán in câu trả lời thô và finish_reason | Phát hiện max_completion_tokens=3 cho chuỗi rỗng (finish_reason=length); tôi sửa PROVIDERS openai max_tokens 3 → 20, chạy lại được F1 ≈ 0.95–0.97 |
| 4 – Câu hỏi 4, Bắt lỗi 3 | Nhờ gợi ý câu trả lời | Câu trả lời kèm số liệu tôi gửi (F1, chi phí, phát hiện max_tokens) | Tôi kiểm tra lại số trên output của mình |
| 5 – Pareto, stress test | Gửi biểu đồ và bảng 50 case khó, nhờ gợi ý trả lời Câu hỏi 5, 6 | Phân tích dựa trên số của tôi (mỗi bậc Pareto ≈ 1 tin; BoW 100% nhóm ham vì đoán ham gần hết) | Tôi đối chiếu với output của mình trước khi dán |
| 5 – Tính tay | Nhờ kiểm tra các bước 5 phép tính | Các bước tính | Tôi tự viết tay trên giấy, tự bấm máy, kiểm tra hash ✅ |
| 5 – Phản ánh, AI log, báo cáo | Nhờ gợi ý phản ánh cuối bài và soạn AI log | Khung câu trả lời và bảng AI log này | Tôi điền số liệu, đọc lại cho đúng với những gì đã làm |
| Lỗi Colab | Hỏi vì sao NameError (pd, os, predict_bow_lr) | Giải thích phiên Colab bị reset, hướng dẫn Run before và ô kiểm tra biến | Chạy lại toàn bộ notebook |
