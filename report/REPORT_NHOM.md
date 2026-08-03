# Báo Cáo Nhóm — Lab 7: Embedding & Vector Store

**Nhóm/Lớp:** K3 — Dịch vụ và quy định Trường Đại học VinUni

**Thành viên:**

- Bùi Hữu Nghĩa — 2A202601880
- Hà Nhật Khánh Duy — 2A202602031

**Ngày:** 03/08/2026

> Báo cáo đã ghi nhận hai thành viên và so sánh hai cấu hình trên cùng corpus, cùng local TF-IDF và cùng 5 benchmark query VinUni. Báo cáo cá nhân gốc của Duy còn ghi thí nghiệm mock trên corpus riêng; phần nhóm dùng kết quả chạy lại có kiểm soát để bảo đảm công bằng.

---

## 1. Lựa chọn tài liệu — Nhóm (10 điểm)

### Phạm vi

Bộ dữ liệu tập trung vào các dịch vụ thiết yếu của VinUni: đăng ký học phần, học phí và học bổng, thư viện, ký túc xá và đời sống nội trú.

Tám tài liệu trong `data/vinuni_services/` là bản tóm lược tiếng Việt do nhóm biên soạn từ trang công khai chính thức. Nội dung không chứa dữ liệu cá nhân, thông tin đăng nhập hoặc tài liệu nội bộ. Các chi tiết được giữ ở mức cần thiết để kiểm chứng câu trả lời benchmark, không sao chép toàn bộ trang nguồn.

### Danh sách tài liệu

| # | Tài liệu | Nguồn chính thức | Ngày lấy / phiên bản | Số ký tự nội dung | Metadata chính |
|---:|---|---|---|---:|---|
| 1 | Đăng ký học phần Hè 2026 | [Registrar VinUni](https://registrar.vinuni.edu.vn/2026/06/29/announcement-launch-of-the-new-student-portal-for-summer-2026-course-registration/) | 03/08/2026 / 29/06/2026 | 936 | `student`, `registrar`, `course-registration` |
| 2 | Mốc đăng ký học phần Xuân 2026 | [Registrar VinUni](https://registrar.vinuni.edu.vn/2025/12/15/official-announcement-spring-2026-course-registration/) | 03/08/2026 / 15/12/2025 | 919 | `student`, `registrar`, `academic-deadlines` |
| 3 | Học phí đại học 2026–2027 | [VinUni Admissions](https://admissions.vinuni.edu.vn/tuition-fee/undergraduate/) | 03/08/2026 / 07/2026 | 950 | `student`, `admissions`, `tuition` |
| 4 | Học bổng và hỗ trợ tài chính | [VinUni Admissions FAQ](https://admissions.vinuni.edu.vn/undergraduate/faqs/tuition-fee-scholarship-and-financial-aids/) | 03/08/2026 / không nêu | 1.073 | `student`, `admissions`, `scholarship-financial-aid` |
| 5 | Chính sách truy cập và dịch vụ thư viện | [VinUni Policy](https://policy.vinuni.edu.vn/all-policies/library-policies-for-users/) | 03/08/2026 / `POL-LLR-001-V4.0` | 1.026 | `all`, `library`, `library-policy` |
| 6 | Quyền mượn tài liệu thư viện | [VinUni Library](https://library.vinuni.edu.vn/borrowing-priviledge/) | 03/08/2026 / không nêu | 885 | `student`, `library`, `borrowing-policy` |
| 7 | Hướng dẫn đời sống nội trú | [VinUni Policy](https://policy.vinuni.edu.vn/all-policies/residential-life-guideline/) | 03/08/2026 / `GDL-SAM-008-V5.0` | 893 | `student`, `student-affairs`, `residential-life` |
| 8 | FAQ ký túc xá | [VinUni Admissions FAQ](https://admissions.vinuni.edu.vn/undergraduate/faqs/residence-life-in-dorm/) | 03/08/2026 / không nêu | 914 | `student`, `admissions`, `dormitory-faq` |

File `sources.csv` ánh xạ một-một với 8 file Markdown và ghi `license_or_permission` là trang công khai chính thức/bản tóm lược chính sách công khai.

### Cấu trúc metadata

| Trường | Kiểu | Ví dụ | Công dụng |
|---|---|---|---|
| `doc_id` | chuỗi | `vinuni-library-borrowing-privileges` | Định danh, xóa toàn bộ chunk của một tài liệu |
| `source_url` | URL | `https://library.vinuni.edu.vn/...` | Truy vết và kiểm chứng nguồn |
| `retrieved_at` | ngày ISO | `2026-08-03` | Kiểm tra độ mới dữ liệu |
| `document_version` | chuỗi | `POL-LLR-001-V4.0` | Xác định phiên bản chính sách |
| `audience` | chuỗi | `student`, `all` | Lọc đúng đối tượng sử dụng |
| `department` | chuỗi | `library`, `registrar` | Thu hẹp theo đơn vị phụ trách |
| `category` | chuỗi | `tuition`, `course-registration` | Thu hẹp theo loại dịch vụ |
| `language` / `source_language` | chuỗi | `vi` / `en` | Phân biệt bản tóm lược và ngôn ngữ nguồn |

Checklist quản trị dữ liệu:

- [x] Có 8 tài liệu công khai, không chứa dữ liệu cá nhân hoặc nội dung cần đăng nhập.
- [x] Mỗi tài liệu có `source_url`, `retrieved_at`, `document_version`, `audience` và metadata bổ sung.
- [x] `sources.csv` khớp đủ 8 tài liệu.
- [x] Các câu trả lời chuẩn đều kiểm chứng được từ nguồn và corpus.

---

## 2. Thiết kế chiến lược — Nhóm (15 điểm)

### Baseline chunking trên ba tài liệu

Cấu hình so sánh dùng `chunk_size=650`; SentenceChunker dùng ba câu mỗi chunk.

| Tài liệu | Chiến lược | Số chunk | Độ dài TB | Nhận xét |
|---|---|---:|---:|---|
| Đăng ký Xuân 2026 | Fixed size | 2 | 484,50 | Có overlap nhưng có thể cắt giữa câu |
| Đăng ký Xuân 2026 | Sentence | 3 | 304,67 | Câu trọn vẹn, chunk nhỏ hơn |
| Đăng ký Xuân 2026 | Recursive | 2 | 458,50 | Ưu tiên đoạn văn, giữ ngữ cảnh tốt |
| Đăng ký Hè 2026 | Fixed size | 2 | 493,00 | Kích thước đều nhưng ranh giới cơ học |
| Đăng ký Hè 2026 | Sentence | 3 | 310,33 | Dễ đọc, số chunk tăng |
| Đăng ký Hè 2026 | Recursive | 2 | 467,00 | Giữ nhóm quy tắc liên quan |
| FAQ ký túc xá | Fixed size | 2 | 482,00 | Có thể tách câu hỏi khỏi câu trả lời |
| FAQ ký túc xá | Sentence | 3 | 303,00 | Bảo toàn câu nhưng chia nhỏ ý liên tiếp |
| FAQ ký túc xá | Recursive | 2 | 456,00 | Giữ các đoạn FAQ tương đối hoàn chỉnh |

### Chiến lược của Bùi Hữu Nghĩa

- **Chunking:** `RecursiveChunker(chunk_size=650)`.
- **Embedding:** `TfidfEmbedder` chạy cục bộ, kết hợp word n-gram 1–2 và character n-gram 3–5, sau đó chuẩn hóa L2.
- **Metadata:** giữ nguyên toàn bộ metadata tài liệu trên từng chunk; câu hỏi thư viện thứ 5 dùng `metadata_filter={"audience": "student"}`.
- **Lý do:** corpus nhỏ, có nhiều thuật ngữ và con số chính xác. Word n-gram ưu tiên cụm từ đặc trưng; character n-gram hỗ trợ biến thể dấu câu/từ; recursive chunking tránh cắt cơ học giữa các đoạn quy định.
- **Giới hạn:** TF-IDF là local vector baseline, không phải neural semantic embedding. Nó nhanh, không cần API/model download nhưng yếu với câu diễn đạt lại bằng từ đồng nghĩa hoàn toàn khác.

Backend tạo vector 6.036 chiều trên corpus và tập query benchmark. Toàn bộ benchmark chạy trong vài giây, không cần mạng sau khi cài `scikit-learn`.

### Chiến lược của Hà Nhật Khánh Duy

- **Mã sinh viên:** 2A202602031.
- **Chunking trong báo cáo cá nhân:** `RecursiveChunker(chunk_size=300)`, tạo 36 chunk trên corpus có quy mô tương đương.
- **Kết quả gốc:** 42/42 test; 3/5 câu có chunk liên quan trong top-3 khi dùng mock embedding trên bộ câu hỏi/corpus riêng.
- **Kết quả chạy lại công bằng:** dùng corpus VinUni, TF-IDF và 5 query chung; tạo 36 chunk, đạt 5/5 tài liệu đúng ở top-1 và 5/5 có đoạn trả lời trong top-3.
- **Nhận xét:** chunk 300 cô lập tốt các quy tắc ngắn và cho điểm cao ở câu GPA, nhưng có thể tách tiêu đề khỏi câu trả lời. Ở câu thư viện, top-1 chỉ chứa tiêu đề còn nội dung “3 tài liệu trong 2 tuần” nằm ở top-2.

### So sánh thành viên

| Thành viên | Chiến lược | Điểm retrieval | Điểm mạnh | Điểm yếu |
|---|---|---:|---|---|
| Bùi Hữu Nghĩa | Recursive 650 + TF-IDF word/char | 10/10 | Nhanh, tái lập, top-1 đúng 5/5 | Hiểu paraphrase kém hơn mô hình đa ngữ |
| Hà Nhật Khánh Duy | Recursive 300 + TF-IDF word/char | 10/10 | Chunk chi tiết, điểm cao ở quy tắc ngắn | 36 chunk; đôi lúc tách tiêu đề khỏi nội dung |

Hai cấu hình đều đạt 10/10 theo tiêu chí top-3. Cấu hình 650 của Nghĩa dùng 16 chunk và giữ context đầy đủ hơn; cấu hình 300 của Duy dùng 36 chunk, có điểm top-1 cao hơn ở bốn câu cuối nhưng đôi lúc cần top-2 để ghép đủ câu trả lời. Vì vậy chunk 650 phù hợp làm mặc định cho corpus ngắn này, còn chunk 300 hữu ích khi cần cô lập quy tắc chi tiết.

---

## 3. Câu hỏi đánh giá và chất lượng truy xuất — Nhóm (10 điểm)

### Bộ câu hỏi và câu trả lời chuẩn

| # | Câu hỏi | Câu trả lời chuẩn | Chunk chứa thông tin |
|---:|---|---|---|
| 1 | Từ học kỳ Hè 2026, sinh viên đăng ký học phần ở cổng nào và hệ thống kiểm tra điều kiện gì? | VinUniDigi Student Portal; hệ thống kiểm tra học phần tiên quyết và yêu cầu học trước. | `vinuni-course-registration-summer-2026`, chunk 0 |
| 2 | Hạn cuối bỏ học phần trong học kỳ Xuân 2026 là ngày nào? | 13/03/2026. | `vinuni-course-registration-spring-2026`, chunk 0 |
| 3 | Học phí niêm yết mỗi năm của Cử nhân Điều dưỡng VinUni 2026–2027 là bao nhiêu? | 349.650.000 đồng/năm, trước khi trừ hỗ trợ học phí 35%. | `vinuni-undergraduate-tuition-2026-2027`, chunk 0–1 |
| 4 | GPA tối thiểu để duy trì học bổng 100% hoặc toàn phần là bao nhiêu? | GPA từ 3,2 và không vi phạm nghiêm trọng quy định. | `vinuni-scholarship-financial-aid`, chunk 1 |
| 5 | Sinh viên đại học được mượn tối đa bao nhiêu tài liệu và trong bao lâu? | 3 tài liệu trong 2 tuần, được gia hạn một lần. Dùng filter `audience=student`. | `vinuni-library-borrowing-privileges`, chunk 0 |

### Kết quả retrieval

| # | Top-1 tài liệu | Điểm top-1 | Tài liệu đúng trong top-3? | Đánh giá |
|---:|---|---:|---|---|
| 1 | Đăng ký học phần Hè 2026 | 0,330770 | Có, hạng 1 và 3 | Đúng nguồn và đủ hai ý |
| 2 | Đăng ký học phần Xuân 2026 | 0,324124 | Có, hạng 1 | Đúng hạn 13/03/2026 |
| 3 | Học phí đại học 2026–2027 | 0,330513 | Có, hạng 1 và 2 | Top-1 chứa mức phí, top-2 chứa hỗ trợ 35% |
| 4 | Học bổng và hỗ trợ tài chính | 0,318956 | Có, hạng 1 và 2 | Top-1 chứa GPA 3,2 và điều kiện vi phạm |
| 5 | Quyền mượn tài liệu thư viện | 0,334438 | Có, hạng 1 và 2 | Filter `audience=student`, top-1 trả lời trực tiếp |

**Kết quả:** 5/5 câu có chunk liên quan ở top-1 và top-3; retrieval đạt 10/10 theo tiêu chí top-3. Agent có đủ context để tạo đúng năm câu trả lời chuẩn mà không cần suy đoán.

### So sánh hai cấu hình trên cùng benchmark

| # | Nghĩa — Recursive 650 | Duy — Recursive 300 | Cấu hình có điểm top-1 cao hơn | Ghi chú |
|---:|---:|---:|---|---|
| 1 | 0,330770 | 0,329164 | Nghĩa | Duy cần top-2 để lấy thêm ý điều kiện tiên quyết |
| 2 | 0,324124 | 0,342807 | Duy | Cả hai top-1 đúng nguồn và chứa hạn bỏ môn |
| 3 | 0,330513 | 0,350495 | Duy | Cả hai top-1 chứa mức học phí Điều dưỡng |
| 4 | 0,318956 | 0,411850 | Duy | Chunk nhỏ cô lập đúng đoạn GPA 3,2 |
| 5 | 0,334438 | 0,351965 | Duy | Top-1 của Duy chỉ là tiêu đề; câu trả lời nằm ở top-2 |

Cả hai đạt 5/5 top-3. Điểm similarity cao hơn không luôn đồng nghĩa context top-1 đầy đủ hơn, thể hiện rõ ở câu 5 của cấu hình chunk 300.

Metadata filter hữu ích nhất ở câu 5: nó loại chính sách thư viện chung có `audience=all`, ưu tiên tài liệu mô tả quyền mượn cụ thể cho sinh viên. Tuy nhiên, lọc quá chặt có thể làm giảm recall khi câu trả lời chỉ tồn tại trong tài liệu `audience=all`.

---

## 4. Demo, failure analysis và bài học — Nhóm (5 điểm)

### Failure case

Query thử thêm: **“Bạn đọc bậc cử nhân được checkout sách trong thời hạn thế nào?”**

TF-IDF không trả tài liệu `vinuni-library-borrowing-privileges` trong top-3. Top-1 là chính sách thư viện chung (0,105891), top-2 là học phí (0,095571), top-3 là FAQ ký túc xá (0,081437). Nguyên nhân là query dùng “bạn đọc”, “bậc cử nhân” và “checkout”, trong khi corpus dùng “sinh viên đại học” và “mượn”; độ giao từ/ngữ thấp làm lexical embedding thất bại.

Cách cải thiện:

1. Dùng Sentence Transformer đa ngữ khi tải mô hình hoàn tất.
2. Bổ sung query expansion/synonym như `checkout → mượn`, `bậc cử nhân → sinh viên đại học`.
3. Kết hợp điểm TF-IDF với dense embedding theo hybrid retrieval.
4. Dùng metadata `department=library` để loại tài liệu ngoài thư viện.

### Bài học và demo

- Local TF-IDF là baseline nhanh, minh bạch và mạnh khi câu hỏi chứa thuật ngữ/con số giống tài liệu.
- Recursive chunking giữ các đoạn quy định liền mạch; chunk 650 ký tự tạo 16 chunk từ 8 tài liệu.
- Metadata filter cải thiện precision nhưng cần thiết kế giá trị `student`/`all` cẩn thận để không loại nguồn chung hữu ích.
- Demo có thể chạy bằng: `.venv\Scripts\python.exe -m scripts.run_vinuni_benchmark`.

Nếu làm lại, nhóm sẽ thêm dense multilingual embedding và đo Recall@3/MRR trên cả câu hỏi diễn đạt lại, không chỉ các câu có từ khóa gần với nguồn.

---

## Tự đánh giá phần nhóm

| Tiêu chí | Điểm tự đánh giá |
|---|---:|
| Lựa chọn tài liệu | 10 / 10 |
| Thiết kế chiến lược | 14 / 15 |
| Chất lượng truy xuất | 10 / 10 |
| Demo | 4 / 5 |
| **Tổng** | **38 / 40** |

Trừ điểm vì mới so sánh hai kích thước của cùng RecursiveChunker trên TF-IDF; chưa có Sentence/heading chunker trong retrieval benchmark và chưa chạy Sentence Transformer đa ngữ.
