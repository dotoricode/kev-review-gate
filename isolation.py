"""macOS per-process Seatbelt isolation; failure is fatal, no unisolated fallback."""
import os,ctypes
from pathlib import Path
def isolate():
    for k,v in {'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1','DO_NOT_TRACK':'1','TOKENIZERS_PARALLELISM':'false','PYTORCH_ENABLE_MPS_FALLBACK':'0'}.items():os.environ[k]=v
    for k in list(os.environ):
        if any(x in k for x in ['API_KEY','HF_TOKEN','GATEWAY','ANTHROPIC']):os.environ.pop(k,None)
    lib=ctypes.CDLL('/usr/lib/libsandbox.dylib');lib.sandbox_init.argtypes=[ctypes.c_char_p,ctypes.c_uint64,ctypes.POINTER(ctypes.c_char_p)];err=ctypes.c_char_p()
    if lib.sandbox_init(Path(__file__).with_name('local-only.sb').read_bytes(),0,ctypes.byref(err))!=0:raise RuntimeError('process_isolation_unavailable')
