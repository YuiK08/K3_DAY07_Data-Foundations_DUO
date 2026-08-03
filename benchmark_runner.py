"""
benchmark_runner.py — chạy 5 câu hỏi đánh giá trên 2 chiến lược chunking
và tính similarity cho 5 cặp câu. Kết quả dùng để điền vào báo cáo.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingest import build_knowledge_base
from src.embeddings import _mock_embed
from src.chunking import (
    FixedSizeChunker, RecursiveChunker, SentenceChunker,
    ChunkingStrategyComparator, compute_similarity
)
from src.agent import KnowledgeBaseAgent

DATA_DIR = "data/k3_university"

QUERIES = [
    "Sinh viên đăng ký học phần ở đâu và trong thời gian nào?",
    "Điều kiện để được nhận học bổng khuyến khích học tập là gì?",
    "Học phí phải đóng theo hình thức nào và trong thời hạn bao lâu?",
    "Sinh viên được mượn tối đa bao nhiêu quyển sách thư viện?",
    "Điều kiện để được ở ký túc xá là gì?",
]

GOLD_ANSWERS = [
    "Sinh viên đăng ký học phần qua Cổng thông tin sinh viên (portal) trong thời gian quy định của từng học kỳ. Học kỳ 1: tháng 7-8, Học kỳ 2: tháng 12-1, Hè: tháng 5-6.",
    "GPA đạt từ 3.2/4.0 trở lên, đăng ký tối thiểu 14 tín chỉ, không bị điểm F, không bị kỷ luật, đóng học phí đúng hạn.",
    "Đóng học phí qua chuyển khoản ngân hàng, cổng thanh toán trực tuyến, hoặc nộp trực tiếp. Thời hạn: trong 3 tuần đầu học kỳ.",
    "Sinh viên đại học được mượn tối đa 5 quyển sách.",
    "Ưu tiên: sinh viên năm nhất từ tỉnh khác, hộ nghèo/cận nghèo, dân tộc thiểu số, GPA ≥ 3.6.",
]

def demo_llm(prompt: str) -> str:
    lines = prompt.split("\n")
    context_lines = [l for l in lines if l.startswith("[")]
    context_preview = " | ".join(context_lines[:3])[:200]
    return f"[Agent] Dựa trên ngữ cảnh: {context_preview}..."

def run_benchmark(chunker, strategy_name: str):
    print(f"\n{'='*60}")
    print(f"CHIẾN LƯỢC: {strategy_name}")
    print(f"{'='*60}")

    store = build_knowledge_base(DATA_DIR, embedding_fn=_mock_embed, chunker=chunker)
    print(f"Số chunks nạp vào store: {store.get_collection_size()}")

    agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)

    results = []
    for i, (query, gold) in enumerate(zip(QUERIES, GOLD_ANSWERS), 1):
        print(f"\nQ{i}: {query}")
        top3 = store.search(query, top_k=3)
        answer = agent.answer(query, top_k=3)

        print(f"  Gold: {gold[:80]}...")
        for j, r in enumerate(top3, 1):
            src = r['metadata'].get('doc_id', r['metadata'].get('source', '?'))
            print(f"  [{j}] score={r['score']:.4f} doc={src}")
            print(f"       {r['content'][:100].replace(chr(10),' ')}...")

        results.append({
            "query": query,
            "top3": top3,
            "answer": answer,
        })

    # Q2 với metadata filter (audience=student)
    print(f"\n--- Test search_with_filter (audience=student) trên Q2 ---")
    filtered = store.search_with_filter(QUERIES[1], top_k=3, metadata_filter={"audience": "student"})
    for j, r in enumerate(filtered, 1):
        print(f"  [{j}] score={r['score']:.4f} audience={r['metadata'].get('audience','?')} doc={r['metadata'].get('doc_id','?')}")

    return results


def run_comparator():
    print(f"\n{'='*60}")
    print("BASELINE: ChunkingStrategyComparator trên 2 tài liệu mẫu")
    print(f"{'='*60}")
    sample_texts = []
    import pathlib
    for fn in ["dang-ky-hoc-phan.md", "thu-vien-muon-tra-sach.md"]:
        p = pathlib.Path(DATA_DIR) / fn
        if p.exists():
            txt = p.read_text(encoding="utf-8")
            # Remove front matter
            if txt.startswith("---"):
                parts = txt.split("---", 2)
                body = parts[2].strip() if len(parts) >= 3 else txt
            else:
                body = txt
            sample_texts.append((fn, body))

    comp = ChunkingStrategyComparator()
    for fname, text in sample_texts:
        print(f"\nFile: {fname} ({len(text)} ký tự)")
        result = comp.compare(text, chunk_size=300)
        for strat, stats in result.items():
            print(f"  {strat:20s}: {stats['count']:3d} chunks, avg_len={stats['avg_length']:.0f}")


def run_similarity_pairs():
    print(f"\n{'='*60}")
    print("SIMILARITY: 5 cặp câu dự đoán")
    print(f"{'='*60}")
    pairs = [
        ("Sinh viên đăng ký học phần qua cổng thông tin.", "Thời gian đăng ký môn học bắt đầu vào tháng 7."),
        ("Học bổng khuyến khích học tập yêu cầu GPA từ 3.2.", "Điều kiện nhận học bổng là điểm trung bình cao."),
        ("Sinh viên được mượn 5 quyển sách tại thư viện.", "Ký túc xá áp dụng giờ giới nghiêm lúc 23:00."),
        ("Học phí được tính theo số tín chỉ đăng ký.", "Phí tín chỉ kỹ thuật là 550.000 VNĐ/tín chỉ."),
        ("Sinh viên vắng quá 20% số tiết bị cấm thi.", "Sách trả trễ bị phạt 2.000 VNĐ/quyển/ngày."),
    ]
    predictions = ["cao", "cao", "thấp", "cao", "thấp"]

    for i, ((a, b), pred) in enumerate(zip(pairs, predictions), 1):
        vec_a = _mock_embed(a)
        vec_b = _mock_embed(b)
        score = compute_similarity(vec_a, vec_b)
        actual = "cao" if score > 0.3 else "thấp"
        correct = "✓" if pred == actual else "✗"
        print(f"\nCặp {i}: {correct} (Dự đoán: {pred}, Thực tế: {actual}, Score: {score:.4f})")
        print(f"  A: {a}")
        print(f"  B: {b}")


if __name__ == "__main__":
    # Baseline comparator
    run_comparator()

    # Chiến lược 1: RecursiveChunker (chiến lược của tôi)
    run_benchmark(RecursiveChunker(chunk_size=300), "RecursiveChunker (chunk_size=300)")

    # Chiến lược 2: FixedSizeChunker (baseline)
    run_benchmark(FixedSizeChunker(chunk_size=300, overlap=50), "FixedSizeChunker (chunk_size=300, overlap=50)")

    # Similarity pairs
    run_similarity_pairs()
