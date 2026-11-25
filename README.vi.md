# ReTrace: Đánh giá độ bền đối kháng của hệ thống quy kết mối đe dọa mạng

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Conference](https://img.shields.io/badge/NSS--SocialSec-2026-blue.svg)](https://nsclab.org/nss2026/)
[![Paper](https://img.shields.io/badge/paper-s%E1%BA%AFp%20c%C3%B3-lightgrey.svg)](#trích-dẫn)

Mã nguồn chính thức của bài báo

> **Rewriting the Traces: Adversarial Robustness Evaluation of Temporal Graph-Based Cyber Threat Attribution**
> Duong Chi Vy, Hien Do Hoang, Van-Hau Pham, Phan The Duy.
> Được chấp nhận tại [NSS-SocialSec 2026](https://nsclab.org/nss2026/) (Springer LNCS).
> Đường dẫn bài báo: *sẽ bổ sung khi kỷ yếu được xuất bản.*

> 🇬🇧 English version: [README.md](README.md)

Repo này **chỉ chứa mã nguồn ReTrace**. ReTrace đánh giá các mô hình quy kết được huấn luyện bằng
bản tái hiện ThreatMAMBA, nên chạy trên nền repo
**[Deja-Vyx/ThreatMamba-reproduction](https://github.com/Deja-Vyx/ThreatMamba-reproduction)**
(chuẩn bị dữ liệu, bộ trích TTP, bộ dựng CSKG, các mô hình ThreatMAMBA). Phần tái hiện không được
chép lại ở đây.

---

## Mục lục

- [ReTrace làm gì](#retrace-làm-gì)
- [Kết quả chính](#kết-quả-chính)
- [Cấu trúc repo](#cấu-trúc-repo)
- [Cài đặt](#cài-đặt)
- [Chạy ReTrace](#chạy-retrace)
- [Các phép nhiễu](#các-phép-nhiễu)
- [Độ đo và giao thức](#độ-đo-và-giao-thức)
- [Đối chiếu bài báo và mã nguồn](#đối-chiếu-bài-báo-và-mã-nguồn)
- [Thay đổi trong bản tái hiện (patch)](#thay-đổi-trong-bản-tái-hiện-patch)
- [Kiểm thử](#kiểm-thử)
- [Trích dẫn](#trích-dẫn)
- [Lời cảm ơn và giấy phép](#lời-cảm-ơn-và-giấy-phép)

---

## ReTrace làm gì

Hệ thống quy kết mối đe dọa mạng (cyber threat attribution, CTA) ánh xạ báo cáo tình báo mối đe dọa
mạng (CTI) tới nhóm tấn công đứng sau. Các hệ thống này thường được đánh giá trên báo cáo sạch hoặc
bị cắt ngắn, dù quy kết vốn mang tính đối kháng: nhóm tấn công thay đổi dấu vết và cài cờ giả, và
báo cáo thay đổi theo. ReTrace là một phương pháp đánh giá độ bền đối kháng của các hệ thống CTA:

- **Mô hình mối đe dọa.** CTA đối kháng được hình thức hóa thành bài toán hộp đen có ràng buộc trên
  các biến thể của báo cáo, tách ba nguồn thay đổi: khác biệt cách viết báo cáo (R), thao túng hoạt
  động bởi nhóm tấn công (A) và thao túng trực tiếp báo cáo (M).
- **19 phép nhiễu** ở ba mức (văn bản, bằng chứng, đồ thị), mỗi phép có kỳ vọng kiểm chứng được
  (INV, DIR, TGT) và các bước kiểm tra hợp lệ tự động (giữ nguyên IOC, không thêm tên nhóm mới, tỷ lệ
  độ dài, NLI hai chiều cho bản viết lại bằng LLM).
- **Tìm kiếm hộp đen** (ngẫu nhiên, greedy-k, best-of-N, có mục tiêu) ở chế độ **chuyển giao**
  (transfer) và **thích ứng** (adaptive).
- **Chấm điểm trên tiền tố báo cáo**: x% đơn vị bằng chứng đầu tiên theo thứ tự kể.
- **Giao thức ghép cặp** so sánh các hệ thống có độ chính xác khác nhau trên những báo cáo mà cả hai
  cùng quy kết đúng, kèm khoảng tin cậy bootstrap theo tài liệu.
- **Chẩn đoán theo tầng**: phân rã theo tầng (trích xuất / biểu diễn / phân loại), chẩn đoán đồ thị
  và sáu giả thuyết shortcut (S1–S6).

ReTrace được áp dụng cho ba hệ thống: **ThreatMAMBA** (đồ thị tri thức theo thời gian),
**TTP-Match** (khớp kỹ thuật bằng embedding, Guru và cộng sự 2025) và **TTP-Vec** (phân loại véc-tơ
kỹ thuật, Hussain và Petrova 2026).

## Kết quả chính

Theo bài báo (12.538 biến thể hợp lệ của 144 báo cáo CTI thực từ 12 nhóm):

- **Câu mô tả kỹ thuật là bề mặt tấn công chính.** Loại bỏ câu của một đến ba kỹ thuật, được chọn
  bằng tìm kiếm hộp đen, làm đổi nhãn 22–72% số quy kết đúng (hệ thống dựa trên đồ thị: 22% bị đổi
  nhãn, chỉ 6% bị quy kết sai một cách tự tin). Cờ giả kỹ thuật có mục tiêu thành công 14–30%.
- **Phần lớn các lần đổi nhãn đi kèm thay đổi trong bằng chứng được trích xuất**: có chủ đích khi
  câu kỹ thuật bị loại bỏ, nhưng là lỗi trích xuất khi câu vô hại hoặc câu chứa IOC bị gán kỹ thuật
  (bộ trích TTP gán nhãn cho 82% câu vô hại). Sửa trực tiếp đồ thị chỉ làm đổi nhãn tối đa 4%.
- **Cách biểu diễn bằng chứng quyết định thay đổi lan truyền tới đâu.** Trên các báo cáo mà cả hai
  hệ thống quy kết đúng, thao túng thích ứng làm đổi nhãn nhiều hơn 12.4–15.0 điểm phần trăm số quy
  kết của các hệ thống dựa trên kỹ thuật so với hệ thống dựa trên đồ thị, một khoảng cách mà tấn
  công chuyển giao đánh giá thấp (+3.1 / +4.5 điểm phần trăm).

## Cấu trúc repo

```
ReTrace-CTA-robustness/
├── retrace/                    # phương pháp ReTrace (đánh giá các checkpoint ThreatMAMBA)
│   ├── operators/              #   T1–T6 (văn bản), E1–E8 (bằng chứng), G1–G5 (đồ thị) + phép đối chứng ID
│   ├── pipeline.py             #   văn bản -> CSKG bằng GraphBuilder của bản tái hiện; kiểm tra parity
│   ├── validate.py             #   kiểm tra hợp lệ, NLI, xuất dữ liệu đánh giá bằng người, Cohen's kappa
│   ├── search.py  runner.py    #   chiến lược tìm kiếm hộp đen; sinh và chấm biến thể (chạy tiếp được)
│   ├── models.py               #   nạp/chấm checkpoint trên các tiền tố báo cáo
│   ├── eligible.py  pools.py   #   tập đánh giá; pool câu vô hại / câu mang IOC / cờ giả
│   ├── lexicons.py  llm.py     #   bí danh, viết tắt, ngày tháng; client Ollama có cache SQLite
│   ├── metrics.py  analysis.py #   flip, Δp, ASR, Δacc + CI bootstrap; S1–S6, phân rã theo tầng
│   ├── report.py               #   outputs/retrace/REPORT.md
│   └── preflight.py  sentcache.py  baselines.py  mitigate.py  config.py  common.py
├── cta_systems/                # các hệ thống được đánh giá bằng ReTrace
│   ├── ttp_match/              #   TTP-Match (Guru và cộng sự 2025)
│   ├── ttp_vec/                #   TTP-Vec (Hussain và Petrova 2026)
│   ├── evaluation/             #   chấm lại (chuyển giao), tấn công thích ứng, báo cáo so sánh
│   └── prepare.py  common.py
├── configs/
│   ├── retrace.yaml            # lưới tham số phép nhiễu, LLM, ngưỡng hợp lệ, tiền tố
│   └── cta_systems.yaml        # cấu hình TTP-Match / TTP-Vec, tấn công thích ứng
├── scripts/
│   ├── run_retrace.sh          # chạy toàn bộ ReTrace trên ThreatMAMBA (một lệnh)
│   ├── run_cta_systems.sh      # chạy TTP-Match và TTP-Vec trên benchmark ReTrace (một lệnh)
│   └── train_seeds.sh          # tùy chọn: thêm seed, mẫu âm cùng kích thước
├── tests/
│   ├── test_retrace.py         # 17 test chạy trên CPU
│   └── test_cta_systems.py     # 14 test chạy trên CPU
├── patches/
│   └── threatmamba-reproduction.patch   # thay đổi nhỏ, có cờ, trên 4 file của bản tái hiện
└── docs/
    ├── PIPELINE.md             # từng bước, đầu ra, xử lý sự cố (tiếng Anh)
    └── CASE_STUDIES.md         # phần giữ nguyên / thay đổi so với mã TTP-Match và TTP-Vec (tiếng Anh)
```

## Cài đặt

**1. Cài bản tái hiện.** Làm theo README của
[ThreatMamba-reproduction](https://github.com/Deja-Vyx/ThreatMamba-reproduction) (GPU,
`mamba-ssm`, tải dữ liệu) và chạy pipeline đến bước huấn luyện, để có đủ: `data/processed/`,
`data/attck/`, `data/cskg/*.pt`, `outputs/ttp_extractor.pt`, `outputs/model_main.pt`,
`outputs/model_no_cl.pt`, `outputs/model_no_mamba.pt`.

**2. Thêm ReTrace và áp dụng patch.**

```bash
git clone https://github.com/Deja-Vyx/ThreatMamba-reproduction.git
git clone https://github.com/Deja-Vyx/ReTrace-CTA-robustness.git

cd ThreatMamba-reproduction
git apply ../ReTrace-CTA-robustness/patches/threatmamba-reproduction.patch
cp -r ../ReTrace-CTA-robustness/retrace ../ReTrace-CTA-robustness/cta_systems .
cp ../ReTrace-CTA-robustness/configs/*.yaml configs/
cp ../ReTrace-CTA-robustness/scripts/*.sh scripts/
cp ../ReTrace-CTA-robustness/tests/*.py tests/
```

Patch được tạo trên commit `61cc452` của bản tái hiện. Mọi thay đổi đều có cờ và giữ nguyên hành vi
mặc định, nên checkpoint và kết quả đã có vẫn dùng được (xem
[bên dưới](#thay-đổi-trong-bản-tái-hiện-patch)).

**3. Thư viện.** Không cần cài thêm gói nào ngoài `requirements.txt` của bản tái hiện. Thành phần
tùy chọn:

| Thành phần | Dùng cho | Cài đặt |
| --- | --- | --- |
| [Ollama](https://ollama.com) + `qwen3.5:9b` | phép nhiễu dùng LLM: T1, T2, E2-neutral, E6 | `ollama pull qwen3.5:9b`; đặt `OLLAMA_HOST` nếu Ollama chạy trên máy khác |
| `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | kiểm tra NLI hai chiều cho bản viết lại bằng LLM | tự tải từ Hugging Face lần đầu |
| `BAAI/bge-large-en-v1.5` | embedding của TTP-Match | tự tải từ Hugging Face lần đầu |

## Chạy ReTrace

```bash
export PYTHONHASHSEED=0

# ReTrace trên các checkpoint ThreatMAMBA (main, no_cl, no_mamba)
DOCS=pilot LLM=0 bash scripts/run_retrace.sh   # chạy thử: 50 tài liệu, không LLM
bash scripts/run_retrace.sh                     # chạy đầy đủ -> outputs/retrace/REPORT.md

# các hệ thống khác trên cùng benchmark (sau khi đã có các run ReTrace)
bash scripts/run_cta_systems.sh                 # -> outputs/cta_systems/REPORT_CTA_SYSTEMS.md
```

Cả hai script đều chạy tiếp được: nếu bị ngắt, chạy lại đúng lệnh cũ, bước nào xong rồi sẽ được bỏ
qua. Biến môi trường hay dùng: `DOCS`, `LLM`, `LLM_DOCS`, `FORCE`, `MODELS` cho `run_retrace.sh`;
`LIMIT`, `ADAPT`, `FORCE` cho `run_cta_systems.sh`. Từng bước cũng có thể chạy tay, xem
[docs/PIPELINE.md](docs/PIPELINE.md).

## Các phép nhiễu

Kịch bản: R = khác biệt cách viết báo cáo, A = thao túng hoạt động, D = chỉ dùng để chẩn đoán.
Kỳ vọng: **INV** không được đổi nhãn · **DIR** được giảm p(nhóm đúng) nhưng không được đổi nhãn tự
tin · **TGT** không được bị lái sang nhóm mồi B.

| Mức | Mã | Phép nhiễu | Kịch bản | Kỳ vọng |
| --- | --- | --- | --- | --- |
| 0 | ID | Dựng lại báo cáo không đổi (đối chứng tính tất định) | – | INV |
| 1 | T1 | LLM diễn đạt lại k câu (giữ IOC, tên riêng) | R | INV |
| 1 | T2 | LLM tách/gộp k câu, đổi thể chủ động/bị động | R | INV |
| 1 | T3 | Viết tắt ↔ dạng đầy đủ (ví dụ RDP) | R | INV |
| 1 | T4 | Chuyển phụ lục IOC lên đầu báo cáo | R | INV |
| 1 | T5 | Chèn k câu vô hại (không IOC, công cụ, nhóm) | R | INV |
| 1 | T6 | Đổi định dạng ngày (đối chứng âm) | R | INV |
| 2 | E1 | Che k IOC bằng cụm từ chung chung | A | DIR |
| 2 | E2 | Bỏ hoặc trung tính hóa (LLM) câu của k kỹ thuật | A | DIR |
| 2 | E3 | Thay tên công cụ bằng bí danh ATT&CK khác | A | INV |
| 2 | E4 | Che tên nhóm thật hoặc mọi tên nhóm | R | DIR |
| 2 | E5 | Bỏ liên từ thời gian hoặc hoán đổi k câu | A | DIR |
| 2 | E6 | LLM khái quát hóa tên công cụ/lệnh | A | DIR |
| 2 | E7 | Chèn k IOC hợp lệ nhưng không liên quan | A | INV |
| 2 | E8 | Cờ giả: chèn IOC, câu TTP hoặc tên của B | A | TGT |
| 3 | G1 | Xóa k (hoặc mọi) nút IOC, kỹ thuật hoặc chiến thuật | D | DIR |
| 3 | G2 | Xóa hoặc nối lại ngẫu nhiên cạnh IOC–kỹ thuật | D | DIR |
| 3 | G3 | Sắp xếp lại nút bằng chứng (đảo, xáo, hoán đổi) | D | DIR |
| 3 | G4 | Chèn k nút từ đồ thị của nhóm khác | D | INV |
| 3 | G5 | Xáo trộn loại quan hệ của cạnh | D | DIR |

Lưới tham số nằm trong `configs/retrace.yaml`.

## Độ đo và giao thức

| Độ đo | Định nghĩa |
| --- | --- |
| flip / flip_c | nhãn top-1 của biến thể khác bản gốc; flip_c chỉ tính trên báo cáo mà hệ thống quy kết đúng |
| Δp | p(nhóm đúng \| bản gốc) − p(nhóm đúng \| biến thể) |
| vi phạm | sai kỳ vọng của phép (INV: đổi nhãn, DIR: đổi nhãn tự tin với p > 0.5, TGT: bị quy cho B) |
| ASR | (E8) tỷ lệ biến thể bị quy cho nhóm mồi B |
| Δacc | độ chính xác top-1 của bản gốc trừ của biến thể |

Mọi tỷ lệ được lấy trung bình theo tài liệu và kèm khoảng tin cậy bootstrap 95% theo tài liệu (1.000
lần lặp). Một hiệu ứng **được ủng hộ** (supported) khi CI không chứa 0 và hiệu ứng tối thiểu 1 điểm
phần trăm, **yếu** (weak) khi chỉ đạt một điều kiện, và **không được ủng hộ** trong các trường hợp
còn lại. **Giao thức ghép cặp** so sánh hai hệ thống bằng chênh lệch flip_c trên các báo cáo mà cả
hai cùng quy kết đúng.

## Đối chiếu bài báo và mã nguồn

| Bài báo | Mã nguồn |
| --- | --- |
| Mô hình mối đe dọa, phép nhiễu (Bảng 1) | `retrace/operators/`, `configs/retrace.yaml` |
| Sinh biến thể, kiểm tra hợp lệ | `retrace/llm.py`, `retrace/pools.py`, `retrace/lexicons.py`, `retrace/validate.py` |
| Tìm kiếm hộp đen (chuyển giao / thích ứng) | `retrace/search.py`, `retrace/runner.py`, `cta_systems/evaluation/adaptive.py` |
| Tiền tố báo cáo | `retrace/models.py` (`text_cut`), `cta_systems/prepare.py` |
| Độ đo, bootstrap, kết luận | `retrace/metrics.py`, `retrace/analysis.py` (`verdict`) |
| Giao thức ghép cặp | `cta_systems/evaluation/rows.py`, `cta_systems/evaluation/report.py` (mục 7) |
| Phân rã theo tầng, chẩn đoán đồ thị, S1–S6 | `retrace/analysis.py` |
| Tập đánh giá (bộ lọc CTI, 144 báo cáo) | `retrace/eligible.py` |
| Các hệ thống được đánh giá | `cta_systems/ttp_match/`, `cta_systems/ttp_vec/` |

## Thay đổi trong bản tái hiện (patch)

`patches/threatmamba-reproduction.patch` thêm các điểm móc mà ReTrace cần. Mọi thay đổi đều có cờ,
mặc định giữ nguyên hành vi cũ.

| File | Thay đổi |
| --- | --- |
| `src/encoder.py` | `encode_doc_cached` bỏ qua cache đĩa theo tài liệu khi encoder có `skip_disk_cache=True` (ReTrace cache theo câu) |
| `src/contrastive.py` | thêm `neg_splice_matched()` (mẫu âm cùng kích thước, Eq. 8) và `truncate_tail()`; `neg_splice` giữ nguyên |
| `src/train.py` | thêm cờ `--cl-neg {splice,matched}`, `--cskg-dir`, `--aug-dir` |
| `src/evaluate.py` | thêm cờ `--cskg-dir` |
| `configs/default.yaml` | `train.cl_neg_mode: splice` |

## Kiểm thử

```bash
THREATMAMBA_FAKE_ENCODER=1 THREATMAMBA_SSM=simple python tests/test_retrace.py   # 17/17 PASS
CTA_SYSTEMS_TEST=1 python tests/test_cta_systems.py                              # 14 test PASS, ~1 phút
```

Các test chạy trên CPU với encoder, bộ trích TTP, embedding và LLM giả, chỉ ghi vào thư mục tạm.
Test cần `data/processed/` và `data/attck/` của bản tái hiện, và không thay thế việc chạy thật trên
máy có GPU.

## Trích dẫn

Bài báo đã được chấp nhận tại NSS-SocialSec 2026; DOI và số trang sẽ được bổ sung khi kỷ yếu xuất
bản.

```bibtex
@inproceedings{duong2026retrace,
  title     = {Rewriting the Traces: Adversarial Robustness Evaluation of Temporal Graph-Based Cyber Threat Attribution},
  author    = {{Duong Chi Vy} and {Hien Do Hoang} and {Van-Hau Pham} and {Phan The Duy}},
  booktitle = {Proceedings of NSS-SocialSec 2026},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  year      = {2026},
  note      = {To appear}
}
```

Vui lòng trích dẫn cả các hệ thống được đánh giá:
ThreatMAMBA (Ge và cộng sự, *IEEE TIFS* 21:4386–4401, 2026, [10.1109/TIFS.2026.3685967](https://doi.org/10.1109/TIFS.2026.3685967)),
TTP-Match (Guru và cộng sự, Cyber-AI 2025, [10.1109/Cyber-AI66431.2025.11233715](https://doi.org/10.1109/Cyber-AI66431.2025.11233715))
và TTP-Vec (Hussain và Petrova, *Future Internet* 18(8):433, 2026, [10.3390/fi18080433](https://doi.org/10.3390/fi18080433)).

## Lời cảm ơn và giấy phép

Nghiên cứu được tài trợ bởi Đại học Quốc gia Thành phố Hồ Chí Minh (ĐHQG-HCM) trong khuôn khổ đề tài
mã số NCM2025-26-01. Dữ liệu CTI2Attacker và thiết kế ThreatMAMBA thuộc về Ge và cộng sự; phần cài
lại TTP-Match và TTP-Vec theo mã nguồn do tác giả công bố (xem
[docs/CASE_STUDIES.md](docs/CASE_STUDIES.md)). Repo không phân phối lại dữ liệu; dữ liệu được tải bằng
`scripts/download_data.sh` của bản tái hiện.

ReTrace chỉ làm việc trên báo cáo CTI công khai và không thực thi tấn công nào; các biến thể cờ giả
chỉ được sinh ngoại tuyến để đánh giá.

Phát hành theo [giấy phép MIT](LICENSE).
