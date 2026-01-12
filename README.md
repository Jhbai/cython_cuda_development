# Cython + CUDA 開發範本 (繁體中文) 🔧

## 概要 ✅
本專案為一個簡單且實用的開發範本，目的在於示範如何：
- 使用 CUDA (`.cu`) 撰寫 GPU kernel，
- 以 C (`.c` / `.h`) 包裝（expose C ABI），
- 再由 Cython (`.pyx`) 將功能暴露給 Python 使用。

此結構讓開發者可以在 Python 中直接呼叫高效能的 C/CUDA 實作。

---

## 目錄與主要檔案 
- `method.pyx` — Cython 層 (Python 可呼叫的 API)。
- `method.c` — Cython 產生的 C 檔（編譯後會出現）。
- `alg.c`, `alg.h` — C wrapper：提供穩定的 C API 並呼叫 CUDA wrapper。
- `cuda/simd.cu`, `cuda/simd.cuh` — CUDA kernel 及其宣告。
- `setup.py` — 自訂的 build_ext，會用 `nvcc` 編譯 `.cu` 檔。
- `test.py` — 範例程式：初始化資料、呼叫介面並比對結果。
- `requirements.txt` — Python 相依套件。

---

## 快速上手（建置 & 執行）🚀
1. 建議建立並啟用虛擬環境（venv / conda / pipenv 等）。
2. 安裝需求：

```powershell
pip install -r requirements.txt
```

3. 編譯 extension（在專案目錄下）：

```powershell
python setup.py build_ext --inplace
# 或： python -m pip install -e .
```

4. 執行範例：

```powershell
python test.py
```

> 注意：`setup.py` 會嘗試自動偵測 `CUDA_HOME` 或系統常見路徑，若找不到請設定環境變數 `CUDA_HOME` 指向 CUDA 工具組安裝目錄（含 `bin/nvcc` 與 `lib64`）。

---

## 程式架構說明 

### CUDA 層 (`cuda/*.cu`) 
- 實作 GPU kernel（例如 `vector_add_kernel`）。
- 提供 `extern "C"` 的 wrapper（例如 `launch_cuda_add`），負責：
  - 分配 device 記憶體（`cudaMalloc`）、
  - 傳輸資料（`cudaMemcpy` host↔device）、
  - 啟動 kernel、
  - 同步並複製結果回 host、
  - 釋放記憶體。

### C wrapper (`alg.c`, `alg.h`) 
- 提供簡單且穩定的 C API（例如 `void c_process_add(...)`），方便 Cython 或其他 C/C++ 程式呼叫。

### Cython 層 (`method.pyx`) 🐍
- 在 `cdef extern from "alg.h"` 宣告 C 函數。
- 提供 Python 可呼叫的 function（例：`cuda_add`）並檢查輸入（dtype、長度、C-contiguous）。
- 使用 memoryview (`float[:]`) 並以 `&arr[0]` 傳遞原始指標，避免不必要的資料複製。

---

## API 與資料規則 ✅
- 輸入陣列必須為 `np.float32`（單精度）且為 C-contiguous（連續記憶體）。
- 三個陣列大小需相同（a、b、out）。
- 若要允許非連續或不同 dtype，請在 Python 端先用 `np.ascontiguousarray(..., dtype=np.float32)` 轉換。

**範例使用**：

```python
import numpy as np
import cuda_module

N = 1024
a = np.random.rand(N).astype(np.float32)
b = np.random.rand(N).astype(np.float32)
out = np.empty_like(a)

cuda_module.cuda_add(a, b, out)
```

---

## 開發流程（新增 kernel 或功能）
1. 在 `cuda/` 新增或修改 `.cu` 與 `.cuh`，實作 kernel 並提供 `extern "C"` 的 wrapper。 
2. 若需要，修改或新增 `alg.c`/`alg.h` 作為 C level wrapper（保持 API 簡潔）。
3. 在 `method.pyx` 宣告新的 C API，並提供安全的 Python wrapper（檢查 shape/dtype/contiguity）。
4. 在 `setup.py` 的 Extension `sources` 中加入新的 `.cu` / `.c` 檔案（若需要）。
5. 重新編譯：`python setup.py build_ext --inplace`。
6. 編寫或更新 `test.py` 做驗證與效能測試。

---

## 常見問題與排錯 
- nvcc 找不到：
  - 確認 CUDA 已安裝，`nvcc` 在 PATH，或設 `CUDA_HOME` 指向安裝目錄。
- 找不到 `libcudart`：
  - 檢查 `setup.py` 中的 `library_dirs` / `runtime_library_dirs` 是否指向正確的 CUDA `lib64` 路徑。
- Cython 無法編譯 `.pyx`：
  - 確認 `cython` 已安裝（`requirements.txt` 已列出）。
- Windows 特殊性：
  - `setup.py` 針對 Windows 與 Linux 有不同編譯旗標，必要時需依照 MSVC 與 NVCC 的版本調整 `nvcc_flags`。

---

## 進階優化建議 
- 使用 pinned host memory（`cudaHostAlloc` / page-locked memory）以改善資料傳輸效率。
- 利用 streams 與非同步複製重疊傳輸與計算（`cudaMemcpyAsync`）。
- 調整 `threadsPerBlock` 與 block 數量以取得最佳吞吐。
- 若需要支援多個功能，考慮把公共 C API 與錯誤處理集中管理。

---

## 範例：新增一個 `scale` kernel（快速步驟）
1. 新增 `cuda/scale.cu` 與 `cuda/scale.cuh`，實作 `launch_cuda_scale`。
2. 在 `alg.c` 新增簡單 wrapper（或直接在 Cython 中宣告 `extern`）。
3. 在 `setup.py` 的 Extension `sources` 加入 `cuda/scale.cu`。
4. 在 `method.pyx` 加入 Python wrapper 並重新編譯。
