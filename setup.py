import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import numpy

# ----- 自動偵測 CUDA 路徑 ----- #
def find_cuda_home():
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        try_paths = ['/usr/local/cuda', '/opt/cuda', '/usr/lib/cuda']
        for p in try_paths:
            if os.path.exists(p):
                cuda_home = p
                break
    if cuda_home is None:
        pass 
    return cuda_home

CUDA_HOME = find_cuda_home()

# ----- 自定義 Build Extension ----- #
# 這是為了讓 setuptools 知道如何用 nvcc 編譯 .cu 檔案
class CUDA_build_ext(build_ext):
    def build_extensions(self):

        # ----- 針對 .cu 檔案，替換編譯器命令 ----- #
        default_compiler = self.compiler.compiler_so
        
        # ----- Windows and Linux ----- #
        if sys.platform == 'win32':
             # Windows 設定 (假設使用 MSVC + NVCC)
            cflags = ['/O2'] 
            nvcc_flags = ['-O3', '--ptxas-options=-v', '-Xcompiler', '/MD']
        else:
            # Linux 設定
            cflags = ['-O3', '-fPIC', '-std=c99']
            nvcc_flags = ['-O3', '--ptxas-options=-v', '--compiler-options', '-fPIC']

        # ----- 攔截原本的 compile 方法 ----- #
        original_compile = self.compiler.compile

        def unix_cuda_compile(sources, output_dir=None, macros=None, include_dirs=None, debug=0, extra_preargs=None, extra_postargs=None, depends=None):
            # ----- 分離 .cu 和其他檔案 ----- #
            cu_sources = [s for s in sources if os.path.splitext(s)[1] == '.cu']
            other_sources = [s for s in sources if os.path.splitext(s)[1] != '.cu']
            
            # ----- 編譯一般 C/C++ 檔案 ----- #
            objects = original_compile(other_sources, output_dir, macros, include_dirs, debug, extra_preargs, extra_postargs, depends)

            # ------ 編譯 CUDA 檔案 ----- #
            for source in cu_sources:
                target = os.path.join(output_dir, os.path.splitext(os.path.basename(source))[0] + self.compiler.obj_extension)
                
                # ----- 組合 nvcc 命令 ----- #
                cmd = ['nvcc', source, '-c', '-o', target] + nvcc_flags
                for inc in include_dirs:
                    cmd.extend(['-I', inc])
                
                print(f"Executing: {' '.join(cmd)}")
                self.spawn(cmd)
                objects.append(target)
            
            return objects

        self.compiler.compile = unix_cuda_compile
        
        # ----- 設定一般 C 檔案的參數 ----- #
        if sys.platform != 'win32':
            self.compiler.compiler_so = default_compiler
            for ext in self.extensions:
                ext.extra_compile_args = cflags

        build_ext.build_extensions(self)

# ----- 定義 Extension ----- #
ext_modules = [
    Extension(
        name="cuda_module",  # 編譯出來的 .so 或 .pyd 名稱
        sources=[
            "method.pyx",   # Cython
            "alg.c",        # C Algorithm
            "cuda/simd.cu"  # CUDA Kernel
        ],
        include_dirs=[
            numpy.get_include(),
            ".",
            "cuda",
            os.path.join(CUDA_HOME, 'include') if CUDA_HOME else '/usr/local/cuda/include'
        ],
        library_dirs=[
            os.path.join(CUDA_HOME, 'lib64') if CUDA_HOME else '/usr/local/cuda/lib64'
        ],
        libraries=['cudart'], # 必須連結 CUDA Runtime
        runtime_library_dirs=[
            os.path.join(CUDA_HOME, 'lib64') if CUDA_HOME else '/usr/local/cuda/lib64'
        ] if sys.platform != 'win32' else None
    )
]

setup(
    name="cuda_heterogeneous_demo",
    ext_modules=ext_modules,
    cmdclass={'build_ext': CUDA_build_ext},
    install_requires=['numpy', 'cython'],
)
