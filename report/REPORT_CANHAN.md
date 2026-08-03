# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Hà Nhật Khánh Duy
**MSSV:** 2A202602031
**Ngày:** 03/08/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Hai đoạn văn bản có độ tương tự cosine cao nghĩa là vector embedding của chúng chỉ cùng hướng trong không gian chiều cao — tức là hai đoạn văn bản đó mang ý nghĩa, chủ đề, hoặc từ ngữ tương đồng nhau. Điểm cosine gần 1.0 cho thấy nội dung ngữ nghĩa gần như giống nhau.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Sinh viên đăng ký học phần qua cổng thông tin sinh viên."
- Câu B: "Thời gian đăng ký môn học bắt đầu vào tháng 7."
- Tại sao tương đồng: Cả hai câu đều nói về quy trình/thời điểm đăng ký học phần — cùng chủ đề, từ khóa liên quan (đăng ký, học phần, sinh viên).

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Sinh viên được mượn 5 quyển sách tại thư viện."
- Câu B: "Ký túc xá áp dụng giờ giới nghiêm lúc 23:00."
- Tại sao khác: Hai câu thuộc hai chủ đề hoàn toàn khác nhau (thư viện vs ký túc xá), không chia sẻ từ ngữ hay ngữ nghĩa chung.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Vì khoảng cách Euclid bị ảnh hưởng bởi độ lớn (magnitude) của vector — một văn bản dài sẽ có vector lớn hơn văn bản ngắn dù cùng chủ đề. Cosine similarity chỉ đo **góc** giữa hai vector (bỏ qua độ lớn), nên phản ánh chính xác hơn sự tương đồng về ngữ nghĩa bất kể độ dài văn bản.

---

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**

> **Phép tính:**
> - step = chunk_size - overlap = 500 - 50 = 450
> - số_chunk = ceil((10000 - 50) / 450) = ceil(9950 / 450) = ceil(22.11) = **23 chunks**

> **Đáp án: 23 chunks**

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Với overlap=100: step = 500 - 100 = 400; số_chunk = ceil((10000 - 100) / 400) = ceil(9900/400) = ceil(24.75) = **25 chunks** — tăng thêm 2 chunks. Ta muốn overlap cao hơn để đảm bảo thông tin nằm ở biên giữa hai chunk không bị mất ngữ cảnh — mỗi chunk sẽ "nhìn lại" một phần nội dung của chunk trước, giúp truy xuất chính xác hơn khi thông tin quan trọng nằm ở chỗ tiếp giáp.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Tôi dùng `re.split(r'(?<=[.!?])\s+|(?<=\.)\n', text)` để tách câu dựa trên dấu chấm câu (`.`, `!`, `?`) theo sau bởi khoảng trắng hoặc xuống dòng. Sau khi tách, mỗi câu được `strip()` bỏ khoảng trắng thừa, rồi nhóm thành chunks theo `max_sentences_per_chunk`. Edge case được xử lý: nếu text rỗng trả về `[]`, nếu không có câu nào sau tách thì trả về toàn bộ text như 1 chunk.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thuật toán thử tách text bằng từng separator theo thứ tự ưu tiên (`\n\n`, `\n`, `. `, ` `, `""`). Base case: nếu `len(text) <= chunk_size` thì trả về `[text]`; nếu không còn separator nào thì tách theo ký tự (character-level). Với mỗi separator, tôi gom các phần (parts) lại cho đến khi tổng vượt `chunk_size`, rồi flush và tiếp tục đệ quy với `remaining_separators` cho phần còn quá dài.

---

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> `add_documents`: với mỗi `Document`, gọi `_embedding_fn(doc.content)` để lấy vector, rồi lưu dict `{id, content, embedding, metadata}` vào `self._store` (list in-memory). `search`: embed query → tính `_dot(query_vec, stored_vec)` cho mỗi record → sort giảm dần theo score → trả về top_k dict với key `content`, `score`, `metadata`, `id`.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> `search_with_filter`: **filter trước** — lọc `self._store` giữ lại các records thỏa `metadata[k] == v` cho mọi cặp trong `metadata_filter`, rồi gọi `_search_records` trên tập đã lọc. `delete_document`: scan toàn bộ `self._store`, giữ lại records có `metadata["doc_id"] != doc_id` VÀ `record["id"] != doc_id`, trả về `True` nếu list bị thu nhỏ.

---

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Bước 1: `store.search(question, top_k=top_k)` để lấy các chunks liên quan nhất. Bước 2: build prompt theo cấu trúc: "You are a helpful assistant. Use the following context... Context: [1] chunk1 [2] chunk2... Question: ... Answer:". Bước 3: truyền prompt vào `llm_fn(prompt)` và trả về kết quả. Thiết kế này cho phép inject bất kỳ LLM nào qua dependency injection.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

### Kết Quả Kiểm Thử (Test Results)

```
============================= test session starts =============================
platform win32 -- Python 3.14.6, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\AI\LABS\DAY07_2A202602031_HaNhatKhanhDuy
plugins: anyio-4.14.2, langsmith-0.10.10
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.31s ==============================
```

**Số lượng bài test vượt qua (pass):** **42 / 42** ✅

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

Chạy `compute_similarity()` với `_mock_embed` trên 5 cặp câu (thực tế đã chạy qua `benchmark_runner.py`):

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Sinh viên đăng ký học phần qua cổng thông tin. | Thời gian đăng ký môn học bắt đầu vào tháng 7. | cao | **0.4809** (cao) | ✓ |
| 2 | Học bổng khuyến khích học tập yêu cầu GPA từ 3.2. | Điều kiện nhận học bổng là điểm trung bình cao. | cao | **-0.1264** (thấp) | ✗ |
| 3 | Sinh viên được mượn 5 quyển sách tại thư viện. | Ký túc xá áp dụng giờ giới nghiêm lúc 23:00. | thấp | **0.0626** (thấp) | ✓ |
| 4 | Học phí được tính theo số tín chỉ đăng ký. | Phí tín chỉ kỹ thuật là 550.000 VNĐ/tín chỉ. | cao | **0.2903** (thấp*) | ✗ |
| 5 | Sinh viên vắng quá 20% số tiết bị cấm thi. | Sách trả trễ bị phạt 2.000 VNĐ/quyển/ngày. | thấp | **0.3396** (cao) | ✗ |

> *Ngưỡng phân loại cao/thấp = 0.3

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Bất ngờ nhất là Cặp 2: hai câu đều nói về học bổng và GPA nhưng mock embedder lại cho điểm âm (-0.1264) — tức là chúng bị coi là "đối lập". Điều này cho thấy **mock embedder** (dựa trên MD5 hash) **không phản ánh ngữ nghĩa** — nó tạo vector giả ngẫu nhiên dựa trên chuỗi ký tự, không hiểu nội dung. Bài học: khi đánh giá chiến lược retrieval, phải dùng embedder thật (`EMBEDDING_PROVIDER=local`) thay vì mock.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** với chiến lược **RecursiveChunker (chunk_size=300)** — 36 chunks tổng.

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Score | Có liên quan không? | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Sinh viên đăng ký học phần ở đâu và trong thời gian nào? | "Sinh viên đăng ký học phần trong thời gian quy định... HK1: tháng 7-8, HK2: tháng 12-1" (doc: dang-ky-hoc-phan) | 0.3263 | ✅ Có | Dựa trên chunk về thời gian đăng ký từ tài liệu đăng ký học phần |
| 2 | Điều kiện để được nhận học bổng khuyến khích học tập là gì? | "Học phí được tính theo đơn vị tín chỉ..." (doc: hoc-phi-dong-tien) | 0.2561 | ❌ Không | Agent lấy thông tin sai từ học phí thay vì học bổng |
| 3 | Học phí phải đóng theo hình thức nào và trong thời hạn bao lâu? | "Thư viện cung cấp mượn tài liệu..." (doc: k3-library-services) | 0.3855 | ❌ Không | Agent lấy chunk không liên quan (vấn đề mock embedder) |
| 4 | Sinh viên được mượn tối đa bao nhiêu quyển sách thư viện? | "Nội quy ký túc xá: cổng đóng lúc 23:00..." (doc: ky-tuc-xa-noi-tru) | 0.2227 | ❌ Không (top-2 đúng) | Top-2 chứa bảng số sách mượn đúng nhưng không ở top-1 |
| 5 | Điều kiện để được ở ký túc xá là gì? | "Ưu tiên xét duyệt theo thứ tự: 1. SV năm nhất từ tỉnh khác..." (doc: ky-tuc-xa-noi-tru) | 0.3392 | ✅ Có | Đúng tài liệu, đúng thông tin điều kiện ở KTX |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** **3 / 5**

> Câu 1, 5 có top-1 đúng. Câu 4 có top-2 đúng (bảng số sách mượn). Câu 2, 3 không tìm được chunk liên quan — nguyên nhân chính: **mock embedder không phản ánh ngữ nghĩa tiếng Việt**, dẫn đến tìm sai tài liệu.

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Khi dùng local embedder (sentence-transformers), kết quả retrieval cải thiện rõ rệt — câu 2 và 3 tìm đúng tài liệu. Điều này xác nhận: với dữ liệu tiếng Việt, cần embedder đa ngữ thật sự. Ngoài ra, chiến lược chia theo tiêu đề (heading chunker) giữ ngữ cảnh tốt hơn cho văn bản quy định có cấu trúc rõ ràng.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 9 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 8 / 10 |
| **Tổng phần cá nhân** | **57 / 60** |
