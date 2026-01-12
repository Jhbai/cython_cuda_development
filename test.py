import numpy as np
import time
import cuda_module # 這是我們 setup.py 編譯出來的名稱

def main():
    N = 10_000_000
    print(f"Initializing arrays with {N} elements...")

    # ----- 準備數據 (必須是 float32 且記憶體連續) ----- #
    a = np.random.rand(N).astype(np.float32)
    b = np.random.rand(N).astype(np.float32)
    out = np.zeros(N, dtype=np.float32)

    print("Executing CUDA implementation...")
    start_time = time.time()
    
    # ----- 呼叫 Cython 介面 ----- #
    cuda_module.cuda_add(a, b, out)
    
    end_time = time.time()
    print(f"CUDA Time: {end_time - start_time:.4f} seconds")

    # ----- 驗證結果 ----- #
    print("Verifying results...")
    expected = a + b
    
    # ----- 使用 np.allclose 檢查浮點數誤差 ----- #
    if np.allclose(out, expected, atol=1e-5):
        print("SUCCESS: Results match CPU implementation.")
    else:
        print("FAILURE: Results do not match.")
        print("First 5 GPU:", out[:5])
        print("First 5 CPU:", expected[:5])

if __name__ == "__main__":
    main()
