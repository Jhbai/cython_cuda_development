# cython: language_level=3
import numpy as np
cimport numpy as np

cdef extern from "alg.h":
    void c_process_add(float* a, float* b, float* out, int n)

def cuda_add(float[:] arr_a, float[:] arr_b, float[:] arr_out):
    """
    接收 Numpy arrays (具備連續記憶體)，並傳遞指標給 C。
    """

    # ----- 檢查維度一致性 ----- #
    if arr_a.shape[0] != arr_b.shape[0] or arr_a.shape[0] != arr_out.shape[0]:
        raise ValueError("All arrays must have the same size")
    
    cdef int n = arr_a.shape[0]

    # ----- 確保記憶體是連續的 (C-contiguous) ----- #
    # 這裡我們傳遞記憶體視圖 (MemoryView) 的第一個元素的地址
    if not (arr_a.is_c_contig() and arr_b.is_c_contig() and arr_out.is_c_contig()):
        raise ValueError("Arrays must be C-contiguous")

    c_process_add(&arr_a[0], &arr_b[0], &arr_out[0], n)
