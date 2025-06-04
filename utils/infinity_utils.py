import json

import math
import argparse
import random
from math import log2, ceil
from functools import partial
try:
    # Python 3.9+
    from functools import cache
except ImportError:
    # Python 3.8 and earlier
    from functools import lru_cache
    def cache(func):
        return lru_cache(maxsize=None)(func)

from collections import namedtuple
from contextlib import nullcontext
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed import nn as dist_nn
from torch import Tensor
from torch.amp import autocast
from torch.nn.functional import scaled_dot_product_attention as slow_attn    # q, k, v: BHLc
from einops import rearrange, reduce, pack, unpack
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    flex_attention_available = True
except ImportError:
    print(f"[Warning] flex attention need pytorch 2.5.0+ but your version is {torch.__version__}")
    flex_attention_available = False

from timm.models.layers import DropPath
import numpy as np

try:
    from flash_attn import flash_attn_varlen_kvpacked_func
    from flash_attn.ops.rms_norm import rms_norm as rms_norm_impl
    from flash_attn.ops.fused_dense import fused_mlp_func
except ImportError:
    def rms_norm_impl(x, weight, epsilon):
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True).add_(epsilon))) * weight


"""
=============================================================================================================
= This utility grocery contains the following things:
= - Some constants used in this file
= - The bitwise self-correction (BSC) code for the video generation head
= - The BSQ quantization code
= - The vae model definition and loading code for the video generation head
= - The attention building blocks and other components (position embd, contants, norm, adaptation, etc.)
=============================================================================================================
"""


################################
# Here is some constants
################################

dynamic_resolution_h_w = {
    1.0: {
        '0.06M': {
            'pixel': (256, 256),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 4, 4), (4, 6, 6), (5, 8, 8), (6, 12, 12), (7, 16, 16)]
        },
        '0.25M': {
            'pixel': (512, 512),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 4, 4), (4, 6, 6), (5, 8, 8), (6, 12, 12), (7, 16, 16),
                      (9, 20, 20), (11, 24, 24), (13, 32, 32)]
        },
        '0.60M': {
            'pixel': (768, 768),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 4, 4), (4, 6, 6), (5, 8, 8), (6, 12, 12), (7, 16, 16),
                      (9, 20, 20), (11, 24, 24), (13, 32, 32), (15, 40, 40), (17, 48, 48)]
        },
        '1M': {
            'pixel': (1024, 1024),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 4, 4), (4, 6, 6), (5, 8, 8), (6, 12, 12), (7, 16, 16),
                      (9, 20, 20), (11, 24, 24), (13, 32, 32), (15, 40, 40), (17, 48, 48), (21, 64, 64)]
        }
    },
    1.25: {
        '0.06M': {
            'pixel': (320, 256),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 5, 4), (5, 10, 8), (6, 15, 12), (7, 20, 16)]
        },
        '0.25M': {
            'pixel': (560, 448),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 5, 4), (5, 10, 8), (6, 15, 12), (7, 20, 16),
                      (9, 25, 20), (11, 30, 24), (13, 35, 28)]
        },
        '0.60M': {
            'pixel': (880, 704),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 5, 4), (5, 10, 8), (6, 15, 12), (7, 20, 16),
                      (9, 25, 20), (11, 30, 24), (13, 35, 28), (15, 45, 36), (17, 55, 44)]
        },
        '1M': {
            'pixel': (1120, 896),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 5, 4), (5, 10, 8), (6, 15, 12), (7, 20, 16),
                      (9, 25, 20), (11, 30, 24), (13, 35, 28), (15, 45, 36), (17, 55, 44), (21, 70, 56)]
        }
    },
    0.8: {
        '0.06M': {
            'pixel': (256, 320),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 5), (5, 8, 10), (6, 12, 15), (7, 16, 20)]
        },
        '0.25M': {
            'pixel': (448, 560),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 5), (5, 8, 10), (6, 12, 15), (7, 16, 20),
                      (9, 20, 25), (11, 24, 30), (13, 28, 35)]
        },
        '0.60M': {
            'pixel': (704, 880),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 5), (5, 8, 10), (6, 12, 15), (7, 16, 20),
                      (9, 20, 25), (11, 24, 30), (13, 28, 35), (15, 36, 45), (17, 44, 55)]
        },
        '1M': {
            'pixel': (896, 1120),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 5), (5, 8, 10), (6, 12, 15), (7, 16, 20),
                      (9, 20, 25), (11, 24, 30), (13, 28, 35), (15, 36, 45), (17, 44, 55), (21, 56, 70)]
        }
    },
    1.333: {
        '0.06M': {
            'pixel': (320, 240),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 4, 3), (4, 8, 6), (5, 12, 9), (6, 16, 12), (7, 20, 15)]
        },
        '0.25M': {
            'pixel': (576, 432),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 4, 3), (4, 8, 6), (5, 12, 9), (6, 16, 12), (7, 20, 15),
                      (9, 24, 18), (11, 28, 21), (13, 36, 27)]
        },
        '0.60M': {
            'pixel': (960, 720),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 4, 3), (4, 8, 6), (5, 12, 9), (6, 16, 12), (7, 20, 15),
                      (9, 24, 18), (11, 28, 21), (13, 36, 27), (15, 48, 36), (17, 60, 45)]
        },
        '1M': {
            'pixel': (1152, 864),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 4, 3), (4, 8, 6), (5, 12, 9), (6, 16, 12), (7, 20, 15),
                      (9, 24, 18), (11, 28, 21), (13, 36, 27), (15, 48, 36), (17, 60, 45), (21, 72, 54)]
        }
    },
    0.75: {
        '0.06M': {
            'pixel': (240, 320),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 3, 4), (4, 6, 8), (5, 9, 12), (6, 12, 16), (7, 15, 20)]
        },
        '0.25M': {
            'pixel': (432, 576),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 3, 4), (4, 6, 8), (5, 9, 12), (6, 12, 16), (7, 15, 20),
                      (9, 18, 24), (11, 21, 28), (13, 27, 36)]
        },
        '0.60M': {
            'pixel': (720, 960),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 3, 4), (4, 6, 8), (5, 9, 12), (6, 12, 16), (7, 15, 20),
                      (9, 18, 24), (11, 21, 28), (13, 27, 36), (15, 36, 48), (17, 45, 60)]
        },
        '1M': {
            'pixel': (864, 1152),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 3, 4), (4, 6, 8), (5, 9, 12), (6, 12, 16), (7, 15, 20),
                      (9, 18, 24), (11, 21, 28), (13, 27, 36), (15, 36, 48), (17, 45, 60), (21, 54, 72)]
        }
    },
    1.5: {
        '0.06M': {
            'pixel': (336, 224),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 3, 2), (4, 6, 4), (5, 9, 6), (6, 15, 10), (7, 21, 14)]
        },
        '0.25M': {
            'pixel': (624, 416),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 3, 2), (4, 6, 4), (5, 9, 6), (6, 15, 10), (7, 21, 14),
                      (9, 27, 18), (11, 33, 22), (13, 39, 26)]
        },
        '0.60M': {
            'pixel': (1008, 672),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 3, 2), (4, 6, 4), (5, 9, 6), (6, 15, 10), (7, 21, 14),
                      (9, 27, 18), (11, 33, 22), (13, 39, 26), (15, 48, 32), (17, 63, 42)]
        },
        '1M': {
            'pixel': (1248, 832),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 3, 2), (4, 6, 4), (5, 9, 6), (6, 15, 10), (7, 21, 14),
                      (9, 27, 18), (11, 33, 22), (13, 39, 26), (15, 48, 32), (17, 63, 42), (21, 78, 52)]
        }
    },
    0.666: {
        '0.06M': {
            'pixel': (224, 336),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 2, 3), (4, 4, 6), (5, 6, 9), (6, 10, 15), (7, 14, 21)]
        },
        '0.25M': {
            'pixel': (416, 624),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 2, 3), (4, 4, 6), (5, 6, 9), (6, 10, 15), (7, 14, 21),
                      (9, 18, 27), (11, 22, 33), (13, 26, 39)]
        },
        '0.60M': {
            'pixel': (672, 1008),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 2, 3), (4, 4, 6), (5, 6, 9), (6, 10, 15), (7, 14, 21),
                      (9, 18, 27), (11, 22, 33), (13, 26, 39), (15, 32, 48), (17, 42, 63)]
        },
        '1M': {
            'pixel': (832, 1248),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 2, 3), (4, 4, 6), (5, 6, 9), (6, 10, 15), (7, 14, 21),
                      (9, 18, 27), (11, 22, 33), (13, 26, 39), (15, 32, 48), (17, 42, 63), (21, 52, 78)]
        }
    },
    1.75: {
        '0.06M': {
            'pixel': (336, 192),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 7, 4), (5, 11, 6), (6, 14, 8), (7, 21, 12)]
        },
        '0.25M': {
            'pixel': (672, 384),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 7, 4), (5, 11, 6), (6, 14, 8), (7, 21, 12),
                      (9, 28, 16), (11, 35, 20), (13, 42, 24)]
        },
        '0.60M': {
            'pixel': (1120, 640),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 7, 4), (5, 11, 6), (6, 14, 8), (7, 21, 12),
                      (9, 28, 16), (11, 35, 20), (13, 42, 24), (15, 56, 32), (17, 70, 40)]
        },
        '1M': {
            'pixel': (1344, 768),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 7, 4), (5, 11, 6), (6, 14, 8), (7, 21, 12),
                      (9, 28, 16), (11, 35, 20), (13, 42, 24), (15, 56, 32), (17, 70, 40), (21, 84, 48)]
        }
    },
    0.571: {
        '0.06M': {
            'pixel': (192, 336),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 7), (5, 6, 11), (6, 8, 14), (7, 12, 21)]
        },
        '0.25M': {
            'pixel': (384, 672),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 7), (5, 6, 11), (6, 8, 14), (7, 12, 21),
                      (9, 16, 28), (11, 20, 35), (13, 24, 42)]
        },
        '0.60M': {
            'pixel': (640, 1120),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 7), (5, 6, 11), (6, 8, 14), (7, 12, 21),
                      (9, 16, 28), (11, 20, 35), (13, 24, 42), (15, 32, 56), (17, 40, 70)]
        },
        '1M': {
            'pixel': (768, 1344),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 7), (5, 6, 11), (6, 8, 14), (7, 12, 21),
                      (9, 16, 28), (11, 20, 35), (13, 24, 42), (15, 32, 56), (17, 40, 70), (21, 48, 84)]
        }
    },
    2.0: {
        '0.06M': {
            'pixel': (352, 176),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 4, 2), (4, 6, 3), (5, 10, 5), (6, 16, 8), (7, 22, 11)]
        },
        '0.25M': {
            'pixel': (736, 368),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 4, 2), (4, 6, 3), (5, 10, 5), (6, 16, 8), (7, 22, 11),
                      (9, 30, 15), (11, 38, 19), (13, 46, 23)]
        },
        '0.60M': {
            'pixel': (1184, 592),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 4, 2), (4, 6, 3), (5, 10, 5), (6, 16, 8), (7, 22, 11),
                      (9, 30, 15), (11, 38, 19), (13, 46, 23), (15, 60, 30), (17, 74, 37)]
        },
        '1M': {
            'pixel': (1440, 720),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 4, 2), (4, 6, 3), (5, 10, 5), (6, 16, 8), (7, 22, 11),
                      (9, 30, 15), (11, 38, 19), (13, 46, 23), (15, 60, 30), (17, 74, 37), (21, 90, 45)]
        }
    },
    0.5: {
        '0.06M': {
            'pixel': (176, 352),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 2, 4), (4, 3, 6), (5, 5, 10), (6, 8, 16), (7, 11, 22)]
        },
        '0.25M': {
            'pixel': (368, 736),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 2, 4), (4, 3, 6), (5, 5, 10), (6, 8, 16), (7, 11, 22),
                      (9, 15, 30), (11, 19, 38), (13, 23, 46)]
        },
        '0.60M': {
            'pixel': (592, 1184),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 2, 4), (4, 3, 6), (5, 5, 10), (6, 8, 16), (7, 11, 22),
                      (9, 15, 30), (11, 19, 38), (13, 23, 46), (15, 30, 60), (17, 37, 74)]
        },
        '1M': {
            'pixel': (720, 1440),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 2, 4), (4, 3, 6), (5, 5, 10), (6, 8, 16), (7, 11, 22),
                      (9, 15, 30), (11, 19, 38), (13, 23, 46), (15, 30, 60), (17, 37, 74), (21, 45, 90)]
        }
    },
    2.5: {
        '0.06M': {
            'pixel': (400, 160),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 5, 2), (4, 10, 4), (5, 15, 6), (6, 20, 8), (7, 25, 10)]
        },
        '0.25M': {
            'pixel': (800, 320),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 5, 2), (4, 10, 4), (5, 15, 6), (6, 20, 8), (7, 25, 10),
                      (9, 30, 12), (11, 40, 16), (13, 50, 20)]
        },
        '0.60M': {
            'pixel': (1280, 512),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 5, 2), (4, 10, 4), (5, 15, 6), (6, 20, 8), (7, 25, 10),
                      (9, 30, 12), (11, 40, 16), (13, 50, 20), (15, 65, 26), (17, 80, 32)]
        },
        '1M': {
            'pixel': (1600, 640),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 5, 2), (4, 10, 4), (5, 15, 6), (6, 20, 8), (7, 25, 10),
                      (9, 30, 12), (11, 40, 16), (13, 50, 20), (15, 65, 26), (17, 80, 32), (21, 100, 40)]
        }
    },
    0.4: {
        '0.06M': {
            'pixel': (160, 400),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 2, 5), (4, 4, 10), (5, 6, 15), (6, 8, 20), (7, 10, 25)]
        },
        '0.25M': {
            'pixel': (320, 800),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 2, 5), (4, 4, 10), (5, 6, 15), (6, 8, 20), (7, 10, 25),
                      (9, 12, 30), (11, 16, 40), (13, 20, 50)]
        },
        '0.60M': {
            'pixel': (512, 1280),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 2, 5), (4, 4, 10), (5, 6, 15), (6, 8, 20), (7, 10, 25),
                      (9, 12, 30), (11, 16, 40), (13, 20, 50), (15, 26, 65), (17, 32, 80)]
        },
        '1M': {
            'pixel': (640, 1600),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 2, 5), (4, 4, 10), (5, 6, 15), (6, 8, 20), (7, 10, 25),
                      (9, 12, 30), (11, 16, 40), (13, 20, 50), (15, 26, 65), (17, 32, 80), (21, 40, 100)]
        }
    },
    3.0: {
        '0.06M': {
            'pixel': (432, 144),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 6, 2), (4, 9, 3), (5, 15, 5), (6, 21, 7), (7, 27, 9)]
        },
        '0.25M': {
            'pixel': (864, 288),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 6, 2), (4, 9, 3), (5, 15, 5), (6, 21, 7), (7, 27, 9),
                      (9, 36, 12), (11, 45, 15), (13, 54, 18)]
        },
        '0.60M': {
            'pixel': (1440, 480),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 6, 2), (4, 9, 3), (5, 15, 5), (6, 21, 7), (7, 27, 9),
                      (9, 36, 12), (11, 45, 15), (13, 54, 18), (15, 72, 24), (17, 90, 30)]
        },
        '1M': {
            'pixel': (1776, 592),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 6, 2), (4, 9, 3), (5, 15, 5), (6, 21, 7), (7, 27, 9),
                      (9, 36, 12), (11, 45, 15), (13, 54, 18), (15, 72, 24), (17, 90, 30), (21, 111, 37)]
        }
    },
    0.333: {
        '0.06M': {
            'pixel': (144, 432),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 2, 6), (4, 3, 9), (5, 5, 15), (6, 7, 21), (7, 9, 27)]
        },
        '0.25M': {
            'pixel': (288, 864),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 2, 6), (4, 3, 9), (5, 5, 15), (6, 7, 21), (7, 9, 27),
                      (9, 12, 36), (11, 15, 45), (13, 18, 54)]
        },
        '0.60M': {
            'pixel': (480, 1440),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 2, 6), (4, 3, 9), (5, 5, 15), (6, 7, 21), (7, 9, 27),
                      (9, 12, 36), (11, 15, 45), (13, 18, 54), (15, 24, 72), (17, 30, 90)]
        },
        '1M': {
            'pixel': (592, 1776),
            'scales': [(1, 1, 1), (2, 2, 2), (3, 2, 6), (4, 3, 9), (5, 5, 15), (6, 7, 21), (7, 9, 27),
                      (9, 12, 36), (11, 15, 45), (13, 18, 54), (15, 24, 72), (17, 30, 90), (21, 37, 111)]
        }
    }
}

h_div_w_templates = np.array([
    1.0, 1.25, 0.8, 1.333, 0.75, 1.5, 0.666,
    1.75, 0.571, 2.0, 0.5, 2.5, 0.4, 3.0, 0.333
])

predefined_HW_Scales_dynamic = {
    (16, 16): [(1, 1), (2, 2), (4, 4), (6, 6), (8, 8), (12, 12), (16, 16)],
    (32, 32): [(1, 1), (2, 2), (4, 4), (6, 6), (8, 8), (12, 12), (16, 16), (20, 20), (24, 24), (32, 32)],
    (64, 64): [(1, 1), (2, 2), (4, 4), (6, 6), (8, 8), (12, 12), (16, 16), (20, 20), (24, 24), (32, 32), (40, 40), (48, 48), (64, 64)],
    
    (20, 16): [(1, 1), (2, 2), (3, 3), (5, 4), (10, 8), (15, 12), (20, 16)],
    (35, 28): [(1, 1), (2, 2), (3, 3), (5, 4), (10, 8), (15, 12), (20, 16), (25, 20), (30, 24), (35, 28)],
    (70, 56): [(1, 1), (2, 2), (3, 3), (5, 4), (10, 8), (15, 12), (20, 16), (25, 20), (30, 24), (35, 28), (45, 36), (55, 44), (70, 56)],
    
    (16, 20): [(1, 1), (2, 2), (3, 3), (4, 5), (8, 10), (12, 15), (16, 20)],
    (28, 35): [(1, 1), (2, 2), (3, 3), (4, 5), (8, 10), (12, 15), (16, 20), (20, 25), (24, 30), (28, 35)],
    (56, 70): [(1, 1), (2, 2), (3, 3), (4, 5), (8, 10), (12, 15), (16, 20), (20, 25), (24, 30), (28, 35), (36, 45), (44, 55), (56, 70)],
    
    (20, 15): [(1, 1), (2, 2), (4, 3), (8, 6), (12, 9), (16, 12), (20, 15)],
    (36, 27): [(1, 1), (2, 2), (4, 3), (8, 6), (12, 9), (16, 12), (20, 15), (24, 18), (28, 21), (36, 27)],
    (72, 54): [(1, 1), (2, 2), (4, 3), (8, 6), (12, 9), (16, 12), (20, 15), (24, 18), (28, 21), (36, 27), (48, 36), (60, 45), (72, 54)],
    
    (15, 20): [(1, 1), (2, 2), (3, 4), (6, 8), (9, 12), (12, 16), (15, 20)],
    (27, 36): [(1, 1), (2, 2), (3, 4), (6, 8), (9, 12), (12, 16), (15, 20), (18, 24), (21, 28), (27, 36)],
    (54, 72): [(1, 1), (2, 2), (3, 4), (6, 8), (9, 12), (12, 16), (15, 20), (18, 24), (21, 28), (27, 36), (36, 48), (45, 60), (54, 72)],
    
    (21, 14): [(1, 1), (2, 2), (3, 2), (6, 4), (9, 6), (15, 10), (21, 14)],
    (39, 26): [(1, 1), (2, 2), (3, 2), (6, 4), (9, 6), (15, 10), (21, 14), (27, 18), (33, 22), (39, 26)],
    (78, 52): [(1, 1), (2, 2), (3, 2), (6, 4), (9, 6), (15, 10), (21, 14), (27, 18), (33, 22), (39, 26), (48, 32), (63, 42), (78, 52)],
    
    (14, 21): [(1, 1), (2, 2), (2, 3), (4, 6), (6, 9), (10, 15), (14, 21)],
    (26, 39): [(1, 1), (2, 2), (2, 3), (4, 6), (6, 9), (10, 15), (14, 21), (18, 27), (22, 33), (26, 39)],
    (52, 78): [(1, 1), (2, 2), (2, 3), (4, 6), (6, 9), (10, 15), (14, 21), (18, 27), (22, 33), (26, 39), (32, 48), (42, 63), (52, 78)],
    
    (21, 12): [(1, 1), (2, 2), (3, 3), (7, 4), (11, 6), (14, 8), (21, 12)],
    (42, 24): [(1, 1), (2, 2), (3, 3), (7, 4), (11, 6), (14, 8), (21, 12), (28, 16), (35, 20), (42, 24)],
    (84, 48): [(1, 1), (2, 2), (3, 3), (7, 4), (11, 6), (14, 8), (21, 12), (28, 16), (35, 20), (42, 24), (56, 32), (70, 40), (84, 48)],
    
    (12, 21): [(1, 1), (2, 2), (3, 3), (4, 7), (6, 11), (8, 14), (12, 21)],
    (24, 42): [(1, 1), (2, 2), (3, 3), (4, 7), (6, 11), (8, 14), (12, 21), (16, 28), (20, 35), (24, 42)],
    (48, 84): [(1, 1), (2, 2), (3, 3), (4, 7), (6, 11), (8, 14), (12, 21), (16, 28), (20, 35), (24, 42), (32, 56), (40, 70), (48, 84)],
    
    (22, 11): [(1, 1), (2, 2), (4, 2), (6, 3), (10, 5), (16, 8), (22, 11)],
    (46, 23): [(1, 1), (2, 2), (4, 2), (6, 3), (10, 5), (16, 8), (22, 11), (30, 15), (38, 19), (46, 23)],
    (90, 45): [(1, 1), (2, 2), (4, 2), (6, 3), (10, 5), (16, 8), (22, 11), (30, 15), (38, 19), (46, 23), (60, 30), (74, 37), (90, 45)],
    
    (11, 22): [(1, 1), (2, 2), (2, 4), (3, 6), (5, 10), (8, 16), (11, 22)],
    (23, 46): [(1, 1), (2, 2), (2, 4), (3, 6), (5, 10), (8, 16), (11, 22), (15, 30), (19, 38), (23, 46)],
    (45, 90): [(1, 1), (2, 2), (2, 4), (3, 6), (5, 10), (8, 16), (11, 22), (15, 30), (19, 38), (23, 46), (30, 60), (37, 74), (45, 90)],
    
    (25, 10): [(1, 1), (2, 2), (5, 2), (10, 4), (15, 6), (20, 8), (25, 10)],
    (50, 20): [(1, 1), (2, 2), (5, 2), (10, 4), (15, 6), (20, 8), (25, 10), (30, 12), (40, 16), (50, 20)],
    (100, 40): [(1, 1), (2, 2), (5, 2), (10, 4), (15, 6), (20, 8), (25, 10), (30, 12), (40, 16), (50, 20), (65, 26), (80, 32), (100, 40)],
    
    (10, 25): [(1, 1), (2, 2), (2, 5), (4, 10), (6, 15), (8, 20), (10, 25)],
    (20, 50): [(1, 1), (2, 2), (2, 5), (4, 10), (6, 15), (8, 20), (10, 25), (12, 30), (16, 40), (20, 50)],
    (40, 100): [(1, 1), (2, 2), (2, 5), (4, 10), (6, 15), (8, 20), (10, 25), (12, 30), (16, 40), (20, 50), (26, 65), (32, 80), (40, 100)],
    
    (27, 9): [(1, 1), (2, 2), (6, 2), (9, 3), (15, 5), (21, 7), (27, 9)],
    (54, 18): [(1, 1), (2, 2), (6, 2), (9, 3), (15, 5), (21, 7), (27, 9), (36, 12), (45, 15), (54, 18)],
    (111, 37): [(1, 1), (2, 2), (6, 2), (9, 3), (15, 5), (21, 7), (27, 9), (36, 12), (45, 15), (54, 18), (72, 24), (90, 30), (111, 37)],
    
    (9, 27): [(1, 1), (2, 2), (2, 6), (3, 9), (5, 15), (7, 21), (9, 27)],
    (18, 54): [(1, 1), (2, 2), (2, 6), (3, 9), (5, 15), (7, 21), (9, 27), (12, 36), (15, 45), (18, 54)],
    (37, 111): [(1, 1), (2, 2), (2, 6), (3, 9), (5, 15), (7, 21), (9, 27), (12, 36), (15, 45), (18, 54), (24, 72), (30, 90), (37, 111)]
}


################################
# Here is the bitwise self-correction part
################################

Return = namedtuple('Return', ['quantized', 'indices', 'bit_indices', 'entropy_aux_loss'])
LossBreakdown = namedtuple('LossBreakdown', ['per_sample_entropy', 'batch_entropy', 'commitment'])

@cache
def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1

def maybe_distributed_mean(t):
    if not is_distributed():
        return t

    dist_nn.all_reduce(t)
    t = t / dist.get_world_size()
    return t

# helper functions

def exists(v):
    return v is not None

def identity(t):
    return t

def default(*args):
    for arg in args:
        if exists(arg):
            return arg() if callable(arg) else arg
    return None

def round_up_multiple(num, mult):
    return ceil(num / mult) * mult

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def l2norm(t):
    return F.normalize(t, dim = -1)

# entropy

def log(t, eps = 1e-5):
    return t.clamp(min = eps).log()

def entropy(prob):
    return (-prob * log(prob)).sum(dim=-1)

# cosine sim linear

class CosineSimLinear(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        scale = 1.
    ):
        super().__init__()
        self.scale = scale
        self.weight = nn.Parameter(torch.randn(dim_in, dim_out))

    def forward(self, x):
        x = F.normalize(x, dim = -1)
        w = F.normalize(self.weight, dim = 0)
        return (x @ w) * self.scale


class BitwiseSelfCorrection(object):
    def __init__(self, vae, args):
        self.noise_apply_layers = args.noise_apply_layers
        self.noise_apply_requant = args.noise_apply_requant
        self.noise_apply_strength = args.noise_apply_strength
        self.apply_spatial_patchify = args.apply_spatial_patchify
        self.vae = vae
        self.debug_bsc = args.debug_bsc

    def flip_requant(self, vae_scale_schedule, inp_B3HW, raw_features, device):
        with torch.amp.autocast('cuda', enabled = False):
            B = raw_features.shape[0]
            if raw_features.dim() == 4:
                codes_out = raw_features.unsqueeze(2)
            else:
                codes_out = raw_features
            cum_var_input = 0
            gt_all_bit_indices = []
            pred_all_bit_indices = []
            x_BLC_wo_prefix = []
            for si, (pt, ph, pw) in enumerate(vae_scale_schedule):
                residual = codes_out - cum_var_input
                if si != len(vae_scale_schedule)-1:
                    residual = F.interpolate(residual, size=vae_scale_schedule[si], mode=self.vae.quantizer.z_interplote_down).contiguous()
                quantized, _, bit_indices, loss = self.vae.quantizer.lfq(residual) # quantized shape: [B, d_vae, 1, h, w], bit_indices shape: [B,1,h,w,d_vae]
                gt_all_bit_indices.append(bit_indices)
                if si < self.noise_apply_layers:
                    noise_apply_strength = np.random.randint(0, 100 * self.noise_apply_strength+1) * 0.01
                    mask = torch.rand(*bit_indices.shape).to(device) < noise_apply_strength
                    pred_bit_indices = bit_indices.clone()
                    pred_bit_indices[mask] = 1 - pred_bit_indices[mask]
                    pred_all_bit_indices.append(pred_bit_indices)
                    if self.noise_apply_requant:
                        quantized = self.vae.quantizer.lfq.indices_to_codes(pred_bit_indices, label_type = 'bit_label')
                else:
                    pred_all_bit_indices.append(bit_indices)
                cum_var_input = cum_var_input + F.interpolate(quantized, size=vae_scale_schedule[-1], mode=self.vae.quantizer.z_interplote_up).contiguous()
                if si < len(vae_scale_schedule)-1:
                    this_scale_input = F.interpolate(cum_var_input, size=vae_scale_schedule[si+1], mode=self.vae.quantizer.z_interplote_up).contiguous()
                    x_BLC_wo_prefix.append(this_scale_input.reshape(*this_scale_input.shape[:2], -1).permute(0,2,1)) # (B,H/2*W/2,4C) or (B,H*W,C)

            gt_ms_idx_Bl = [item.reshape(B, -1, self.vae.codebook_dim) for item in gt_all_bit_indices]
            x_BLC_wo_prefix = torch.cat(x_BLC_wo_prefix, 1)
        
        return x_BLC_wo_prefix, gt_ms_idx_Bl






################################
# Here is the BSQ tokenizer part
################################
   
def get_latent2scale_schedule(T: int, H: int, W: int, mode="original"):
    assert mode in ["original", "dynamic", "dense", "same1", "same2", "same3"]
    predefined_HW_Scales = {
        # 256 * 256
        (32, 32): [(1, 1), (2, 2), (3, 3), (4, 4), (6, 6), (9, 9), (13, 13), (18, 18), (24, 24), (32, 32)],
        (16, 16): [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (8, 8), (10, 10), (13, 13), (16, 16)],
        # 1024x1024
        (64, 64): [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (7, 7), (9, 9), (12, 12), (16, 16), (21, 21), (27, 27), (36, 36), (48, 48), (64, 64)],

        (36, 64): [(1, 1), (2, 2), (3, 3), (4, 4), (6, 6), (9, 12), (13, 16), (18, 24), (24, 32), (32, 48), (36, 64)],
    }
    if mode == "dynamic":
        predefined_HW_Scales.update(predefined_HW_Scales_dynamic)
    elif mode == "dense":
        predefined_HW_Scales[(16, 16)] = [(x, x) for x in range(1, 16+1)]
        predefined_HW_Scales[(32, 32)] = predefined_HW_Scales[(16, 16)] + [(20, 20), (24, 24), (28, 28), (32, 32)]
        predefined_HW_Scales[(64, 64)] = predefined_HW_Scales[(32, 32)] + [(40, 40), (48, 48), (56, 56), (64, 64)]
    elif mode.startswith("same"):
        num_quant = int(mode[len("same"):])
        predefined_HW_Scales[(16, 16)] = [(16, 16) for _ in range(num_quant)]
        predefined_HW_Scales[(32, 32)] = [(32, 32) for _ in range(num_quant)]
        predefined_HW_Scales[(64, 64)] = [(64, 64) for _ in range(num_quant)]

    predefined_T_Scales = [1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15, 17, 17, 17, 17, 17]
    patch_THW_shape_per_scale = predefined_HW_Scales[(H, W)]
    if len(predefined_T_Scales) < len(patch_THW_shape_per_scale):
        # print("warning: the length of predefined_T_Scales is less than the length of patch_THW_shape_per_scale!")
        predefined_T_Scales += [predefined_T_Scales[-1]] * (len(patch_THW_shape_per_scale) - len(predefined_T_Scales))
    patch_THW_shape_per_scale = [(min(T, t), h, w ) for (h, w), t in zip(patch_THW_shape_per_scale, predefined_T_Scales[:len(patch_THW_shape_per_scale)])]
    return patch_THW_shape_per_scale

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    normalized_shape: int
    """
    def __init__(self, normalized_shape, norm_weight=False, eps=1e-6, data_format="channels_first"):
        super().__init__()
        if norm_weight:
            self.weight = nn.Parameter(torch.ones(normalized_shape)/(normalized_shape**0.5))
        else:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            if x.ndim == 4: # (b, c, h, w)
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
            elif x.ndim == 5: # (b, c, t, h, w)
                x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            else:
                raise ValueError("the number of dimensions of the input should be 4 or 5")
            return x

class BSQ(nn.Module):
    def __init__(
        self,
        *,
        dim = None,
        codebook_size = None,
        entropy_loss_weight = 0.1,
        commitment_loss_weight = 0.25,
        diversity_gamma = 1.,
        straight_through_activation = nn.Identity(),
        num_codebooks = 1,
        keep_num_codebooks_dim = None,
        codebook_scale = 1.,                        # for residual LFQ, codebook scaled down by 2x at each layer
        frac_per_sample_entropy = 1.,               # make less than 1. to only use a random fraction of the probs for per sample entropy
        has_projections = None,
        projection_has_bias = True,
        soft_clamp_input_value = None,
        cosine_sim_project_in = False,
        cosine_sim_project_in_scale = None,
        channel_first = None,
        experimental_softplus_entropy_loss = False,
        entropy_loss_offset = 5.,                   # how much to shift the loss before softplus
        spherical = True,                          # from https://arxiv.org/abs/2406.07548
        force_quantization_f32 = True,               # will force the quantization step to be full precision
        inv_temperature = 100.0,
        gamma0=1.0, gamma=1.0, zeta=1.0,
        preserve_norm = False, # whether to preserve the original norm info
        new_quant = False, # new quant function，
        mask_out = False, # mask the output as 0 in some conditions
        use_out_phi = False, # use output phi network
        use_out_phi_res = False, # residual out phi
    ):
        super().__init__()

        # some assert validations

        assert exists(dim) or exists(codebook_size), 'either dim or codebook_size must be specified for LFQ'
        assert not exists(codebook_size) or log2(codebook_size).is_integer(), f'your codebook size must be a power of 2 for lookup free quantization (suggested {2 ** ceil(log2(codebook_size))})'

        codebook_size = default(codebook_size, lambda: 2 ** dim)
        self.codebook_size = codebook_size

        codebook_dim = int(log2(codebook_size))
        codebook_dims = codebook_dim * num_codebooks
        dim = default(dim, codebook_dims)
        self.codebook_dims = codebook_dims

        has_projections = default(has_projections, dim != codebook_dims)

        if cosine_sim_project_in:
            cosine_sim_project_in = default(cosine_sim_project_in_scale, codebook_scale)
            project_in_klass = partial(CosineSimLinear, scale = cosine_sim_project_in)
        else:
            project_in_klass = partial(nn.Linear, bias = projection_has_bias)

        self.project_in = project_in_klass(dim, codebook_dims) if has_projections else nn.Identity() # nn.Identity()
        self.project_out = nn.Linear(codebook_dims, dim, bias = projection_has_bias) if has_projections else nn.Identity() # nn.Identity()
        self.has_projections = has_projections

        self.out_phi = nn.Linear(codebook_dims, codebook_dims) if use_out_phi else nn.Identity()
        self.use_out_phi_res = use_out_phi_res
        if self.use_out_phi_res:
            self.out_phi_scale = nn.Parameter(torch.zeros(codebook_dims), requires_grad=True) # init as zero

        self.dim = dim
        self.codebook_dim = codebook_dim
        self.num_codebooks = num_codebooks

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        # channel first

        self.channel_first = channel_first

        # straight through activation

        self.activation = straight_through_activation

        # For BSQ (binary spherical quantization)
        if not spherical:
            raise ValueError("For BSQ, spherical must be True.")
        self.persample_entropy_compute = 'analytical'
        self.inv_temperature = inv_temperature
        self.gamma0 = gamma0  # loss weight for entropy penalty
        self.gamma = gamma  # loss weight for entropy penalty
        self.zeta = zeta    # loss weight for entire entropy penalty
        self.preserve_norm = preserve_norm
        self.new_quant = new_quant
        self.mask_out = mask_out

        # entropy aux loss related weights

        assert 0 < frac_per_sample_entropy <= 1.
        self.frac_per_sample_entropy = frac_per_sample_entropy

        self.diversity_gamma = diversity_gamma
        self.entropy_loss_weight = entropy_loss_weight

        # codebook scale

        self.codebook_scale = codebook_scale

        # commitment loss

        self.commitment_loss_weight = commitment_loss_weight

        # whether to soft clamp the input value from -value to value

        self.soft_clamp_input_value = soft_clamp_input_value
        assert not exists(soft_clamp_input_value) or soft_clamp_input_value >= codebook_scale

        # whether to make the entropy loss positive through a softplus (experimental, please report if this worked or not in discussions)

        self.entropy_loss_offset = entropy_loss_offset
        self.experimental_softplus_entropy_loss = experimental_softplus_entropy_loss

        # for no auxiliary loss, during inference

        self.register_buffer('mask', 2 ** torch.arange(codebook_dim - 1, -1, -1))
        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # whether to force quantization step to be f32

        self.force_quantization_f32 = force_quantization_f32

        # codes

        # all_codes = torch.arange(codebook_size)
        # bits = ((all_codes[..., None].int() & self.mask) != 0).float()
        # codebook = self.bits_to_codes(bits)

        # self.register_buffer('codebook', codebook.float(), persistent = False)

    def bits_to_codes(self, bits):
        return bits * self.codebook_scale * 2 - self.codebook_scale

    # @property
    # def dtype(self):
    #     return self.codebook.dtype

    def indices_to_codes(
        self,
        indices,
        label_type = 'int_label',
        project_out = True
    ):
        assert label_type in ['int_label', 'bit_label']
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))
        should_transpose = default(self.channel_first, is_img_or_video)

        if not self.keep_num_codebooks_dim:
            if label_type == 'int_label':
                indices = rearrange(indices, '... -> ... 1')
            else:
                indices = indices.unsqueeze(-2)

        # indices to codes, which are bits of either -1 or 1

        if label_type == 'int_label':
            assert indices[..., None].int().min() > 0
            bits = ((indices[..., None].int() & self.mask) != 0).float() # .to(self.dtype)
        else:
            bits = indices

        codes = self.bits_to_codes(bits)

        codes = l2norm(codes) # must normalize when using BSQ

        codes = rearrange(codes, '... c d -> ... (c d)')

        # whether to project codes out to original dimensions
        # if the input feature dimensions were not log2(codebook size)

        if project_out:
            codes = self.project_out(codes)

        # rearrange codes back to original shape

        if should_transpose:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    def quantize(self, z):
        assert z.shape[-1] == self.codebook_dims, f"Expected {self.codebook_dims} dimensions, got {z.shape[-1]}"

        zhat = torch.where(z > 0, 
                           torch.tensor(1, dtype=z.dtype, device=z.device), 
                           torch.tensor(-1, dtype=z.dtype, device=z.device))
        return z + (zhat - z).detach()

    def quantize_new(self, z):
        assert z.shape[-1] == self.codebook_dims, f"Expected {self.codebook_dims} dimensions, got {z.shape[-1]}"

        zhat = torch.where(z > 0, 
                           torch.tensor(1, dtype=z.dtype, device=z.device), 
                           torch.tensor(-1, dtype=z.dtype, device=z.device))

        q_scale = 1. / (self.codebook_dims ** 0.5)
        zhat = q_scale * zhat # on unit sphere

        return z + (zhat - z).detach()

    def soft_entropy_loss(self, z):
        if self.persample_entropy_compute == 'analytical':
            # if self.l2_norm:
            p = torch.sigmoid(-4 * z / (self.codebook_dims ** 0.5) * self.inv_temperature)
            # else:
            #     p = torch.sigmoid(-4 * z * self.inv_temperature)
            prob = torch.stack([p, 1-p], dim=-1) # (b, h, w, 18, 2)
            per_sample_entropy = self.get_entropy(prob, dim=-1, normalize=False).sum(dim=-1).mean() # (b,h,w,18)->(b,h,w)->scalar
        else:
            per_sample_entropy = self.get_entropy(prob, dim=-1, normalize=False).sum(dim=-1).mean()

        # macro average of the probability of each subgroup
        avg_prob = reduce(prob, '... g d ->g d', 'mean') # (18, 2)
        codebook_entropy = self.get_entropy(avg_prob, dim=-1, normalize=False)

        # the approximation of the entropy is the sum of the entropy of each subgroup
        return per_sample_entropy, codebook_entropy.sum(), avg_prob

    def get_entropy(self, count, dim=-1, eps=1e-4, normalize=True):
        if normalize: # False
            probs = (count + eps) / (count + eps).sum(dim=dim, keepdim =True)
        else: # True
            probs = count
        H = -(probs * torch.log(probs + 1e-8)).sum(dim=dim)
        return H

    def forward(
        self,
        x,
        return_loss_breakdown = False,
        mask = None,
        entropy_weight=0.1
    ):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """

        is_img_or_video = x.ndim >= 4
        should_transpose = default(self.channel_first, is_img_or_video)

        # standardize image or video into (batch, seq, dimension)

        if should_transpose:
            x = rearrange(x, 'b d ... -> b ... d')
            x, ps = pack_one(x, 'b * d') # x.shape [b, hwt, c]

        assert x.shape[-1] == self.dim, f'expected dimension of {self.dim} but received {x.shape[-1]}'

        x = self.project_in(x)

        # split out number of codebooks

        x = rearrange(x, 'b n (c d) -> b n c d', c = self.num_codebooks)

        x = l2norm(x)

        # whether to force quantization step to be full precision or not

        force_f32 = self.force_quantization_f32

        quantization_context = partial(autocast, 'cuda', enabled = False) if force_f32 else nullcontext

        indices = None
        with quantization_context():

            if force_f32:
                orig_dtype = x.dtype
                x = x.float()
            
            # use straight-through gradients (optionally with custom activation fn) if training
            if self.new_quant:
                quantized = self.quantize_new(x)

            # calculate indices
            bit_indices = (quantized > 0).int()
            entropy_penalty = persample_entropy = cb_entropy = self.zero
            commit_loss = self.zero

            # input back to original dtype if needed

            if force_f32:
                x = x.type(orig_dtype)

        # merge back codebook dim
        x = quantized # rename quantized to x for output
        x = rearrange(x, 'b n c d -> b n (c d)')

        # project out to feature dimension if needed

        x = self.project_out(x)

        # reconstitute image or video dimensions

        if should_transpose:
            x = unpack_one(x, ps, 'b * d')
            x = rearrange(x, 'b ... d -> b d ...')

            bit_indices = unpack_one(bit_indices, ps, 'b * c d')

        # whether to remove single codebook dim

        if not self.keep_num_codebooks_dim:
            bit_indices = rearrange(bit_indices, '... 1 d -> ... d')

        # complete aux loss

        aux_loss = commit_loss * self.commitment_loss_weight + (self.zeta * entropy_penalty / self.inv_temperature)*entropy_weight
        # returns

        ret = Return(x, indices, bit_indices, aux_loss)

        if not return_loss_breakdown:
            return ret

        return ret, LossBreakdown(persample_entropy, cb_entropy, commit_loss)

class MultiScaleBSQ(nn.Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """

    def __init__(
        self,
        *,
        dim,
        codebook_size,
        soft_clamp_input_value = None,
        aux_loss = False, # intermediate auxiliary loss
        ln_before_quant=False, # add a LN before multi-scale RQ
        ln_init_by_sqrt=False, # weight init by 1/sqrt(d)
        use_decay_factor=False,
        use_stochastic_depth=False,
        drop_rate=0.,
        schedule_mode="original", # ["original", "dynamic", "dense"]
        keep_first_quant=False,
        keep_last_quant=False,
        remove_residual_detach=False,
        random_flip = False,
        flip_prob = 0.5,
        flip_mode = "stochastic", # "stochastic", "deterministic"
        max_flip_lvl = 1,
        random_flip_1lvl = False, # random flip one level each time
        flip_lvl_idx = None,
        drop_when_test=False,
        drop_lvl_idx=None,
        drop_lvl_num=0,
        **kwargs
    ):
        super().__init__()
        codebook_dim = int(log2(codebook_size))

        requires_projection = codebook_dim != dim
        self.project_in = nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        self.has_projections = requires_projection
        self.layernorm = LayerNorm(codebook_dim, norm_weight=ln_init_by_sqrt) if ln_before_quant else nn.Identity()
        self.use_stochastic_depth = use_stochastic_depth
        self.drop_rate = drop_rate
        self.remove_residual_detach = remove_residual_detach
        self.random_flip = random_flip
        self.flip_prob = flip_prob
        self.flip_mode = flip_mode
        self.max_flip_lvl = max_flip_lvl
        self.random_flip_1lvl = random_flip_1lvl
        self.flip_lvl_idx = flip_lvl_idx
        assert (random_flip and random_flip_1lvl) == False
        self.drop_when_test = drop_when_test
        self.drop_lvl_idx = drop_lvl_idx
        self.drop_lvl_num = drop_lvl_num
        if self.drop_when_test:
            assert drop_lvl_idx is not None
            assert drop_lvl_num > 0

        self.lfq = BSQ(
            dim = codebook_dim,
            codebook_scale = 1/np.sqrt(codebook_dim),
            soft_clamp_input_value = soft_clamp_input_value,
            # experimental_softplus_entropy_loss=True,
            # entropy_loss_offset=2,
            **kwargs
        )

        self.z_interplote_up = 'trilinear'
        self.z_interplote_down = 'area'
        
        self.use_decay_factor = use_decay_factor
        self.schedule_mode = schedule_mode
        self.keep_first_quant = keep_first_quant
        self.keep_last_quant = keep_last_quant
        if self.use_stochastic_depth and self.drop_rate > 0:
            assert self.keep_first_quant or self.keep_last_quant

    @property
    def codebooks(self):
        return self.lfq.codebook

    def get_codes_from_indices(self, indices_list):
        all_codes = []
        for indices in indices_list:
            codes = self.lfq.indices_to_codes(indices)
            all_codes.append(codes)
        _, _, T, H, W = all_codes[-1].size()
        summed_codes = 0
        for code in all_codes:
            summed_codes += F.interpolate(code, size=(T, H, W), mode=self.z_interplote_up)
        return summed_codes

    def get_output_from_indices(self, indices):
        codes = self.get_codes_from_indices(indices)
        codes_summed = reduce(codes, 'q ... -> ...', 'sum')
        return self.project_out(codes_summed)

    def flip_quant(self, x):
        assert self.flip_mode == 'stochastic'
        flip_mask = torch.rand_like(x) < self.flip_prob
        x = x.clone()
        x[flip_mask] = -x[flip_mask]
        return x

    def forward(
        self,
        x,
        scale_schedule=None,
        mask = None,
        return_all_codes = False,
        return_residual_norm_per_scale = False
    ):
        if x.ndim == 4:
            x = x.unsqueeze(2)
        B, C, T, H, W = x.size()    

        if scale_schedule is None:
            if self.schedule_mode.startswith("same"):
                scale_num = int(self.schedule_mode[len("same"):])
                assert T == 1
                scale_schedule = [(1, H, W)] * scale_num
            else:
                scale_schedule = get_latent2scale_schedule(T, H, W, mode=self.schedule_mode)
                scale_num = len(scale_schedule)

        # x = self.project_in(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous() # (b, c, t, h, w) => (b, t, h, w, c)
        x = self.project_in(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous() # (b, t, h, w, c) => (b, c, t, h, w) 
        x = self.layernorm(x)

        quantized_out = 0.
        residual = x

        all_losses = []
        all_indices = []
        all_bit_indices = []
        var_inputs = []
        residual_norm_per_scale = []
        
        # go through the layers
        out_fact = init_out_fact = 1.0
        # residual_list = []
        # interpolate_residual_list = []
        # quantized_list = []
        if self.drop_when_test:
            drop_lvl_start = self.drop_lvl_idx
            drop_lvl_end = self.drop_lvl_idx + self.drop_lvl_num
        scale_num = len(scale_schedule)
        with autocast('cuda', enabled = False):
            for si, (pt, ph, pw) in enumerate(scale_schedule):
                out_fact = max(0.1, out_fact) if self.use_decay_factor else init_out_fact
                if (pt, ph, pw) != (T, H, W):
                    interpolate_residual = F.interpolate(residual, size=(pt, ph, pw), mode=self.z_interplote_down)
                else:
                    interpolate_residual = residual
                if return_residual_norm_per_scale:
                    residual_norm_per_scale.append((torch.abs(interpolate_residual) < 0.05 * self.lfq.codebook_scale).sum() / interpolate_residual.numel())

                if self.training and self.use_stochastic_depth and random.random() < self.drop_rate:
                    if (si == 0 and self.keep_first_quant) or (si == scale_num - 1 and self.keep_last_quant):
                        quantized, indices, _, loss = self.lfq(interpolate_residual)
                        quantized = quantized * out_fact
                        all_indices.append(indices)
                        all_losses.append(loss)
                    else:
                        quantized = torch.zeros_like(interpolate_residual)
                elif self.drop_when_test and drop_lvl_start <= si < drop_lvl_end:
                    continue                     
                else:
                    quantized, indices, bit_indices, loss = self.lfq(interpolate_residual)
                    if self.random_flip and si < self.max_flip_lvl:
                        quantized = self.flip_quant(quantized)
                    if self.random_flip_1lvl and si == self.flip_lvl_idx:
                        quantized = self.flip_quant(quantized)
                    quantized = quantized * out_fact
                    all_indices.append(indices)
                # quantized_list.append(torch.norm(quantized.detach(), dim=1).mean())
                if (pt, ph, pw) != (T, H, W):
                    quantized = F.interpolate(quantized, size=(T, H, W), mode=self.z_interplote_up).contiguous()
                
                if self.remove_residual_detach:
                    residual = residual - quantized
                else:
                    residual = residual - quantized.detach()
                quantized_out = quantized_out + quantized

                all_bit_indices.append(bit_indices)
                all_losses.append(loss)
                if si != scale_num - 1:
                    var_inputs.append(F.interpolate(quantized_out, size=scale_schedule[si+1], mode=self.z_interplote_down).contiguous())
                
                if self.use_decay_factor:
                    out_fact -= 0.1

        quantized_out = quantized_out.permute(0, 2, 3, 4, 1).contiguous() # (b, c, t, h, w) => (b, t, h, w, c)
        quantized_out = self.project_out(quantized_out)
        quantized_out = quantized_out.permute(0, 4, 1, 2, 3).contiguous() # (b, t, h, w, c) => (b, c, t, h, w)

        # image
        if quantized_out.size(2) == 1:
            quantized_out = quantized_out.squeeze(2)

        # stack all losses and indices

        all_losses = torch.stack(all_losses, dim = -1)

        ret = (quantized_out, all_indices, all_bit_indices, residual_norm_per_scale, all_losses, var_inputs)

        if not return_all_codes:
            return ret

        # whether to return all codes from all codebooks across layers
        all_codes = self.get_codes_from_indices(all_indices)

        # will return all codes in shape (quantizer, batch, sequence length, codebook dimension)

        return (*ret, all_codes)




################################
# Here is the vae part
################################

ptdtype = {None: torch.float32, 'fp32': torch.float32, 'bf16': torch.bfloat16}

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, cnn_type="2d", causal_offset=0, temporal_down=False):
        super().__init__()
        self.cnn_type = cnn_type
        self.slice_seq_len = 17
        
        if cnn_type == "2d":
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        if cnn_type == "3d":
            if temporal_down == False:
                stride = (1, stride, stride)
            else:
                stride = (stride, stride, stride)
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=0)
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size, kernel_size)
            self.padding = (
                kernel_size[0] - 1 + causal_offset,  # Temporal causal padding
                padding,  # Height padding
                padding  # Width padding
            )
        self.causal_offset = causal_offset
        self.stride = stride
        self.kernel_size = kernel_size
        
    def forward(self, x):
        if self.cnn_type == "2d":
            if x.ndim == 5:
                B, C, T, H, W = x.shape
                x = rearrange(x, "B C T H W -> (B T) C H W")
                x = self.conv(x)
                x = rearrange(x, "(B T) C H W -> B C T H W", T=T)
                return x
            else:
                return self.conv(x)
        if self.cnn_type == "3d":
            assert self.stride[0] == 1 or self.stride[0] == 2, f"only temporal stride = 1 or 2 are supported"
            xs = []
            for i in range(0, x.shape[2], self.slice_seq_len+self.stride[0]-1):
                st = i
                en = min(i+self.slice_seq_len, x.shape[2])
                _x = x[:,:,st:en,:,:]
                if i == 0:
                    _x = F.pad(_x, (self.padding[2], self.padding[2],  # Width
                            self.padding[1], self.padding[1],   # Height
                            self.padding[0], 0))                # Temporal
                else:
                    padding_0 = self.kernel_size[0] - 1
                    _x = F.pad(_x, (self.padding[2], self.padding[2],  # Width
                            self.padding[1], self.padding[1],   # Height
                            padding_0, 0))                      # Temporal
                    _x[:,:,:padding_0,
                        self.padding[1]:_x.shape[-2]-self.padding[1],
                        self.padding[2]:_x.shape[-1]-self.padding[2]] += x[:,:,i-padding_0:i,:,:]
                _x = self.conv(_x)
                xs.append(_x)
            try:
                x = torch.cat(xs, dim=2)
            except:
                device = x.device
                del x
                xs = [_x.cpu().pin_memory() for _x in xs]
                torch.cuda.empty_cache()
                x = torch.cat([_x.cpu() for _x in xs], dim=2).to(device=device)
            return x

class Normalize(nn.Module):
    def __init__(self, in_channels, norm_type, norm_axis="spatial"):
        super().__init__()
        self.norm_axis = norm_axis
        assert norm_type in ['group', 'batch', "no"]
        if norm_type == 'group':
            if in_channels % 32 == 0:
                self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
            elif in_channels % 24 == 0: 
                self.norm = nn.GroupNorm(num_groups=24, num_channels=in_channels, eps=1e-6, affine=True)
            else:
                raise NotImplementedError
        elif norm_type == 'batch':
            self.norm = nn.SyncBatchNorm(in_channels, track_running_stats=False) # Runtime Error: grad inplace if set track_running_stats to True
        elif norm_type == 'no':
            self.norm = nn.Identity()
    
    def forward(self, x):
        if self.norm_axis == "spatial":
            if x.ndim == 4:
                x = self.norm(x)
            else:
                B, C, T, H, W = x.shape
                x = rearrange(x, "B C T H W -> (B T) C H W")
                x = self.norm(x)
                x = rearrange(x, "(B T) C H W -> B C T H W", T=T)
        elif self.norm_axis == "spatial-temporal":
            x = self.norm(x)
        else:
            raise NotImplementedError
        return x

def swish(x: Tensor) -> Tensor:
    try:
        return x * torch.sigmoid(x)
    except:
        device = x.device
        x = x.cpu().pin_memory()
        return (x*torch.sigmoid(x)).to(device=device)


class AttnBlock(nn.Module):
    def __init__(self, in_channels, norm_type='group', cnn_param=None):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels, norm_type, norm_axis=cnn_param["cnn_norm_axis"])

        self.q = Conv(in_channels, in_channels, kernel_size=1)
        self.k = Conv(in_channels, in_channels, kernel_size=1)
        self.v = Conv(in_channels, in_channels, kernel_size=1)
        self.proj_out = Conv(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        B, _, T, _, _ = h_.shape
        h_ = self.norm(h_)
        h_ = rearrange(h_, "B C T H W -> (B T) C H W") # spatial attention only
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return rearrange(h_, "(b t) 1 (h w) c -> b c t h w", h=h, w=w, c=c, b=B, t=T)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_type='group', cnn_param=None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels, norm_type, norm_axis=cnn_param["cnn_norm_axis"])
        if cnn_param["res_conv_2d"] in ["half", "full"]:
            self.conv1 = Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, cnn_type="2d")
        else:
            self.conv1 = Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, cnn_type=cnn_param["cnn_type"])
        self.norm2 = Normalize(out_channels, norm_type, norm_axis=cnn_param["cnn_norm_axis"])
        if cnn_param["res_conv_2d"] in ["full"]:
            self.conv2 = Conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1, cnn_type="2d")
        else:
            self.conv2 = Conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1, cnn_type=cnn_param["cnn_type"])
        if self.in_channels != self.out_channels:
            self.nin_shortcut = Conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels, cnn_type="2d", spatial_down=False, temporal_down=False):
        super().__init__()
        assert spatial_down == True
        if cnn_type == "2d":
            self.pad = (0,1,0,1)
        if cnn_type == "3d":
            self.pad = (0,1,0,1,0,0) # add padding to the right for h-axis and w-axis. No padding for t-axis 
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = Conv(in_channels, in_channels, kernel_size=3, stride=2, padding=0, cnn_type=cnn_type, temporal_down=temporal_down)

    def forward(self, x: Tensor):
        x = nn.functional.pad(x, self.pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels, cnn_type="2d", spatial_up=False, temporal_up=False, use_pxsl=False):
        super().__init__()
        if cnn_type == "2d":
            self.scale_factor = 2
            self.causal_offset = 0
        else:
            assert spatial_up == True
            if temporal_up:
                self.scale_factor = (2,2,2)
                self.causal_offset = -1
            else:
                self.scale_factor = (1,2,2)
                self.causal_offset = 0
        self.use_pxsl = use_pxsl
        if self.use_pxsl:
            self.conv = Conv(in_channels, in_channels*4, kernel_size=3, stride=1, padding=1, cnn_type=cnn_type, causal_offset=self.causal_offset)
            self.pxsl = nn.PixelShuffle(2)
        else:
            self.conv = Conv(in_channels, in_channels, kernel_size=3, stride=1, padding=1, cnn_type=cnn_type, causal_offset=self.causal_offset)

    def forward(self, x: Tensor):
        if self.use_pxsl:
            x = self.conv(x)
            x = self.pxsl(x)
        else:
            try:
                x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
            except:
                # shard across channel
                _xs = []
                for i in range(x.shape[1]):
                    _x = F.interpolate(x[:,i:i+1,...], scale_factor=self.scale_factor, mode="nearest")
                    _xs.append(_x)
                x = torch.cat(_xs, dim=1)
            x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        ch: int,
        ch_mult: List[int],
        num_res_blocks: int,
        z_channels: int,
        in_channels = 3,
        patch_size=8, temporal_patch_size=4, 
        norm_type='group', cnn_param=None,
        use_checkpoint=False,
        use_vae=True,
    ):
        super().__init__()
        self.max_down = np.log2(patch_size)
        self.temporal_max_down = np.log2(temporal_patch_size)
        self.temporal_down_offset = self.max_down - self.temporal_max_down
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.cnn_param = cnn_param
        self.use_checkpoint = use_checkpoint
        # downsampling
        # self.conv_in = Conv(in_channels, self.ch, kernel_size=3, stride=1, padding=1)
        # cnn_param["cnn_type"] = "2d" for images, cnn_param["cnn_type"] = "3d" for videos
        if cnn_param["conv_in_out_2d"] == "yes": # "yes" for video
            self.conv_in = Conv(in_channels, ch, kernel_size=3, stride=1, padding=1, cnn_type="2d")
        else:
            self.conv_in = Conv(in_channels, ch, kernel_size=3, stride=1, padding=1, cnn_type=cnn_param["cnn_type"])

        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, norm_type=norm_type, cnn_param=cnn_param))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            # downsample, stride=1, stride=2, stride=2 for 4x8x8 Video VAE
            spatial_down = True if i_level < self.max_down else False
            temporal_down = True if i_level < self.max_down and i_level >= self.temporal_down_offset else False
            if spatial_down or temporal_down:
                down.downsample = Downsample(block_in, cnn_type=cnn_param["cnn_type"], spatial_down=spatial_down, temporal_down=temporal_down)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, norm_type=norm_type, cnn_param=cnn_param)
        if cnn_param["cnn_attention"] == "yes":
            self.mid.attn_1 = AttnBlock(block_in, norm_type, cnn_param=cnn_param)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, norm_type=norm_type, cnn_param=cnn_param)

        # end
        self.norm_out = Normalize(block_in, norm_type, norm_axis=cnn_param["cnn_norm_axis"])
        if cnn_param["conv_inner_2d"] == "yes":
            self.conv_out = Conv(block_in, (int(use_vae) + 1) * z_channels, kernel_size=3, stride=1, padding=1, cnn_type="2d")
        else:
            self.conv_out = Conv(block_in, (int(use_vae) + 1) * z_channels, kernel_size=3, stride=1, padding=1, cnn_type=cnn_param["cnn_type"])

    def forward(self, x, return_hidden=False):
        if not self.use_checkpoint:
            return self._forward(x, return_hidden=return_hidden)
        else:
            return checkpoint.checkpoint(self._forward, x, return_hidden, use_reentrant=False)

    def _forward(self, x: Tensor, return_hidden=False) -> Tensor:
        # downsampling
        h0 = self.conv_in(x)
        hs = [h0]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if hasattr(self.down[i_level], "downsample"):
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        hs_mid = [h]
        h = self.mid.block_1(h)
        if self.cnn_param["cnn_attention"] == "yes":
            h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        hs_mid.append(h)
        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        if return_hidden:
            return h, hs, hs_mid
        else:
            return h


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        ch_mult: List[int],
        num_res_blocks: int,
        z_channels: int,
        out_ch = 3, 
        patch_size=8, temporal_patch_size=4, 
        norm_type="group", cnn_param=None,
        use_checkpoint=False,
        use_freq_dec=False, # use frequency features for decoder
        use_pxsf=False
    ):
        super().__init__()
        self.max_up = np.log2(patch_size)
        self.temporal_max_up = np.log2(temporal_patch_size)
        self.temporal_up_offset = self.max_up - self.temporal_max_up
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.ffactor = 2 ** (self.num_resolutions - 1)
        self.cnn_param = cnn_param
        self.use_checkpoint = use_checkpoint
        self.use_freq_dec = use_freq_dec
        self.use_pxsf = use_pxsf

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]

        # z to block_in
        if cnn_param["conv_inner_2d"] == "yes":
            self.conv_in = Conv(z_channels, block_in, kernel_size=3, stride=1, padding=1, cnn_type="2d")
        else:
            self.conv_in = Conv(z_channels, block_in, kernel_size=3, stride=1, padding=1, cnn_type=cnn_param["cnn_type"])

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, norm_type=norm_type, cnn_param=cnn_param)
        if cnn_param["cnn_attention"] == "yes":
            self.mid.attn_1 = AttnBlock(block_in, norm_type=norm_type, cnn_param=cnn_param)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, norm_type=norm_type, cnn_param=cnn_param)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, norm_type=norm_type, cnn_param=cnn_param))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            # upsample, stride=1, stride=2, stride=2 for 4x8x8 Video VAE, offset 1 compared with encoder
            # https://github.com/black-forest-labs/flux/blob/b4f689aaccd40de93429865793e84a734f4a6254/src/flux/modules/autoencoder.py#L228
            spatial_up = True if 1 <= i_level <= self.max_up else False
            temporal_up = True if 1 <= i_level <= self.max_up and i_level >= self.temporal_up_offset+1 else False
            if spatial_up or temporal_up:
                up.upsample = Upsample(block_in, cnn_type=cnn_param["cnn_type"], spatial_up=spatial_up, temporal_up=temporal_up, use_pxsl=self.use_pxsf)
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in, norm_type, norm_axis=cnn_param["cnn_norm_axis"])
        if cnn_param["conv_in_out_2d"] == "yes":
            self.conv_out = Conv(block_in, out_ch, kernel_size=3, stride=1, padding=1, cnn_type="2d")
        else:
            self.conv_out = Conv(block_in, out_ch, kernel_size=3, stride=1, padding=1, cnn_type=cnn_param["cnn_type"])

    def forward(self, z):
        if not self.use_checkpoint:
            return self._forward(z)
        else:
            return checkpoint(self._forward, z, use_reentrant=False)

    def _forward(self, z: Tensor) -> Tensor:
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        if self.cnn_param["cnn_attention"] == "yes":
            h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if hasattr(self.up[i_level], "upsample"):
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class AutoEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        cnn_param = dict(
            cnn_type=args.cnn_type,
            conv_in_out_2d=args.conv_in_out_2d,
            res_conv_2d=args.res_conv_2d,
            cnn_attention=args.cnn_attention,
            cnn_norm_axis=args.cnn_norm_axis,
            conv_inner_2d=args.conv_inner_2d,
        )
        self.encoder = Encoder(
            ch=args.base_ch,
            ch_mult=args.encoder_ch_mult,
            num_res_blocks=args.num_res_blocks,
            z_channels=args.codebook_dim,
            patch_size=args.patch_size,
            temporal_patch_size=args.temporal_patch_size,
            cnn_param=cnn_param,
            use_checkpoint=args.use_checkpoint,
            use_vae=args.use_vae,
        )
        self.decoder = Decoder(
            ch=args.base_ch,
            ch_mult=args.decoder_ch_mult,
            num_res_blocks=args.num_res_blocks,
            z_channels=args.codebook_dim,
            patch_size=args.patch_size,
            temporal_patch_size=args.temporal_patch_size,
            cnn_param=cnn_param,
            use_checkpoint=args.use_checkpoint,
            use_freq_dec=args.use_freq_dec,
            use_pxsf=args.use_pxsf # pixelshuffle for upsampling
        )
        self.z_drop = nn.Dropout(args.z_drop)
        self.scale_factor = 0.3611
        self.shift_factor = 0.1159
        self.codebook_dim = self.embed_dim = args.codebook_dim

        self.gan_feat_weight = args.gan_feat_weight
        self.video_perceptual_weight = args.video_perceptual_weight
        self.recon_loss_type = args.recon_loss_type
        self.l1_weight = args.l1_weight
        self.use_vae = args.use_vae
        self.kl_weight = args.kl_weight
        self.lfq_weight = args.lfq_weight
        self.image_gan_weight = args.image_gan_weight # image GAN loss weight
        self.video_gan_weight = args.video_gan_weight # video GAN loss weight
        self.perceptual_weight = args.perceptual_weight
        self.flux_weight = args.flux_weight
        self.cycle_weight = args.cycle_weight
        self.cycle_feat_weight = args.cycle_feat_weight
        self.cycle_gan_weight = args.cycle_gan_weight

        self.flux_image_encoder = None
        
        if not args.use_vae:
            if args.quantizer_type == 'MultiScaleBSQ':
                self.quantizer = MultiScaleBSQ(
                    dim = args.codebook_dim,                        # this is the input feature dimension, defaults to log2(codebook_size) if not defined  
                    codebook_size = args.codebook_size,             # codebook size, must be a power of 2
                    entropy_loss_weight = args.entropy_loss_weight, # how much weight to place on entropy loss
                    diversity_gamma = args.diversity_gamma,         # within entropy loss, how much weight to give to diversity of codes, taken from https://arxiv.org/abs/1911.05894
                    preserve_norm=args.preserve_norm,               # preserve norm of the input for BSQ
                    ln_before_quant=args.ln_before_quant,           # use layer norm before quantization
                    ln_init_by_sqrt=args.ln_init_by_sqrt,           # layer norm init value 1/sqrt(d)
                    commitment_loss_weight=args.commitment_loss_weight, # loss weight of commitment loss
                    new_quant=args.new_quant,
                    use_decay_factor=args.use_decay_factor,
                    mask_out=args.mask_out,
                    use_stochastic_depth=args.use_stochastic_depth,
                    drop_rate=args.drop_rate,
                    schedule_mode=args.schedule_mode,
                    keep_first_quant=args.keep_first_quant,
                    keep_last_quant=args.keep_last_quant,
                    remove_residual_detach=args.remove_residual_detach,
                    use_out_phi=args.use_out_phi,
                    use_out_phi_res=args.use_out_phi_res,
                    random_flip = args.random_flip,
                    flip_prob = args.flip_prob,
                    flip_mode = args.flip_mode,
                    max_flip_lvl = args.max_flip_lvl,
                    random_flip_1lvl = args.random_flip_1lvl,
                    flip_lvl_idx = args.flip_lvl_idx,
                    drop_when_test = args.drop_when_test,
                    drop_lvl_idx = args.drop_lvl_idx,
                    drop_lvl_num = args.drop_lvl_num,
                )
                self.quantize = self.quantizer
                self.vocab_size = args.codebook_size
            else:
                raise NotImplementedError(f"{args.quantizer_type} not supported")


    def forward(self, x):
        is_image = x.ndim == 4
        if not is_image:
            B, C, T, H, W = x.shape
        else:
            B, C, H, W = x.shape
            T = 1
        enc_dtype = ptdtype[self.args.encoder_dtype]

        with torch.amp.autocast("cuda", dtype=enc_dtype):
            h, hs, hs_mid = self.encoder(x, return_hidden=True) # B C H W or B C T H W
        hs = [_h.detach() for _h in hs]
        hs_mid = [_h.detach() for _h in hs_mid]
        h = h.to(dtype=torch.float32)
        # print(z.shape)
        # Multiscale LFQ            
        z, all_indices, all_loss = self.quantizer(h)
        x_recon = self.decoder(z)
        vq_output = {
            "commitment_loss": torch.mean(all_loss) * self.lfq_weight, # here commitment loss is sum of commitment loss and entropy penalty
            "encodings": all_indices, 
        }
        return x_recon, vq_output

    def encode_for_raw_features(self, x, scale_schedule, return_residual_norm_per_scale=False):
        is_image = x.ndim == 4
        if not is_image:
            B, C, T, H, W = x.shape
        else:
            B, C, H, W = x.shape
            T = 1

        enc_dtype = ptdtype[self.args.encoder_dtype]
        with torch.amp.autocast("cuda", dtype=enc_dtype):
            h, hs, hs_mid = self.encoder(x, return_hidden=True) # B C H W or B C T H W

        hs = [_h.detach() for _h in hs]
        hs_mid = [_h.detach() for _h in hs_mid]
        h = h.to(dtype=torch.float32)
        return h, hs, hs_mid
    
    def encode(self, x, scale_schedule, return_residual_norm_per_scale=False):
        h, hs, hs_mid = self.encode_for_raw_features(x, scale_schedule, return_residual_norm_per_scale)
        # Multiscale LFQ
        z, all_indices, all_bit_indices, residual_norm_per_scale, all_loss, var_input = self.quantizer(h, scale_schedule=scale_schedule, return_residual_norm_per_scale=return_residual_norm_per_scale)
        return h, z, all_indices, all_bit_indices, residual_norm_per_scale, var_input

    def decode(self, z):
        x_recon = self.decoder(z)
        x_recon = torch.clamp(x_recon, min=-1, max=1)
        return x_recon
    
    def decode_from_indices(self, all_indices, scale_schedule, label_type):
        summed_codes = 0
        for idx_Bl in all_indices:
            codes = self.quantizer.lfq.indices_to_codes(idx_Bl, label_type)
            summed_codes += F.interpolate(codes, size=scale_schedule[-1], mode=self.quantizer.z_interplote_up)
        assert summed_codes.shape[-3] == 1
        x_recon = self.decoder(summed_codes.squeeze(-3))
        x_recon = torch.clamp(x_recon, min=-1, max=1)
        return summed_codes, x_recon

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--flux_weight", type=float, default=0)
        parser.add_argument("--cycle_weight", type=float, default=0)
        parser.add_argument("--cycle_feat_weight", type=float, default=0)
        parser.add_argument("--cycle_gan_weight", type=float, default=0)
        parser.add_argument("--cycle_loop", type=int, default=0)
        parser.add_argument("--z_drop", type=float, default=0.)
        return parser

def load_cnn(model, state_dict, prefix, expand=False, use_linear=False):
    delete_keys = []
    loaded_keys = []
    for key in state_dict:
        if key.startswith(prefix):
            _key = key[len(prefix):]
            if _key in model.state_dict():
                # load nn.Conv2d or nn.Linear to nn.Linear
                if use_linear and (".q.weight" in key or ".k.weight" in key or ".v.weight" in key or ".proj_out.weight" in key):
                    load_weights = state_dict[key].squeeze()
                elif _key.endswith(".conv.weight") and expand:
                    if model.state_dict()[_key].shape == state_dict[key].shape:
                        # 2D cnn to 2D cnn
                        load_weights = state_dict[key]
                    else:
                        # 2D cnn to 3D cnn
                        _expand_dim = model.state_dict()[_key].shape[2]
                        load_weights = state_dict[key].unsqueeze(2).repeat(1, 1, _expand_dim, 1, 1)
                else:
                    load_weights = state_dict[key]
                model.state_dict()[_key].copy_(load_weights)
                delete_keys.append(key)
                loaded_keys.append(prefix+_key)
            # load nn.Conv2d to Conv class
            conv_list = ["conv"] if use_linear else ["conv", ".q.", ".k.", ".v.", ".proj_out.", ".nin_shortcut."]
            if any(k in _key for k in conv_list):
                if _key.endswith(".weight"):
                    conv_key = _key.replace(".weight", ".conv.weight")
                    if conv_key and conv_key in model.state_dict():
                        if model.state_dict()[conv_key].shape == state_dict[key].shape:
                            # 2D cnn to 2D cnn
                            load_weights = state_dict[key]
                        else:
                            # 2D cnn to 3D cnn
                            _expand_dim = model.state_dict()[conv_key].shape[2]
                            load_weights = state_dict[key].unsqueeze(2).repeat(1, 1, _expand_dim, 1, 1)
                        model.state_dict()[conv_key].copy_(load_weights)
                        delete_keys.append(key)
                        loaded_keys.append(prefix+conv_key)
                if _key.endswith(".bias"):
                    conv_key = _key.replace(".bias", ".conv.bias")
                    if conv_key and conv_key in model.state_dict():
                        model.state_dict()[conv_key].copy_(state_dict[key])
                        delete_keys.append(key)
                        loaded_keys.append(prefix+conv_key)
            # load nn.GroupNorm to Normalize class
            if "norm" in _key:
                if _key.endswith(".weight"):
                    norm_key = _key.replace(".weight", ".norm.weight")
                    if norm_key and norm_key in model.state_dict():
                        model.state_dict()[norm_key].copy_(state_dict[key])
                        delete_keys.append(key)
                        loaded_keys.append(prefix+norm_key)
                if _key.endswith(".bias"):
                    norm_key = _key.replace(".bias", ".norm.bias")
                    if norm_key and norm_key in model.state_dict():
                        model.state_dict()[norm_key].copy_(state_dict[key])
                        delete_keys.append(key)
                        loaded_keys.append(prefix+norm_key)
            
    for key in delete_keys:
        del state_dict[key]

    return model, state_dict, loaded_keys

def vae_model(vqgan_ckpt, schedule_mode, codebook_dim, codebook_size, test_mode=True, patch_size=16, encoder_ch_mult=[1, 2, 4, 4, 4], decoder_ch_mult=[1, 2, 4, 4, 4],):
    args=argparse.Namespace(
        vqgan_ckpt=vqgan_ckpt,
        sd_ckpt=None,
        inference_type='image',
        save='./imagenet_val_bsq',
        save_prediction=True,
        image_recon4video=False,
        junke_old=False,
        device='cuda',
        max_steps=1000000.0,
        log_every=1,
        visu_every=1000,
        ckpt_every=1000,
        default_root_dir='',
        compile='no',
        ema='no',
        lr=0.0001,
        beta1=0.9,
        beta2=0.95,
        warmup_steps=0,
        optim_type='Adam',
        disc_optim_type=None,
        lr_min=0.0,
        warmup_lr_init=0.0,
        max_grad_norm=1.0,
        max_grad_norm_disc=1.0,
        disable_sch=False,
        patch_size=patch_size,
        temporal_patch_size=4,
        embedding_dim=256,
        codebook_dim=codebook_dim,
        num_quantizers=8,
        quantizer_type='MultiScaleBSQ',
        use_vae=False,
        use_freq_enc=False,
        use_freq_dec=False,
        preserve_norm=False,
        ln_before_quant=False,
        ln_init_by_sqrt=False,
        use_pxsf=False,
        new_quant=True,
        use_decay_factor=False,
        mask_out=False,
        use_stochastic_depth=False,
        drop_rate=0.0,
        schedule_mode=schedule_mode,
        lr_drop=None,
        lr_drop_rate=0.1,
        keep_first_quant=False,
        keep_last_quant=False,
        remove_residual_detach=False,
        use_out_phi=False,
        use_out_phi_res=False,
        use_lecam_reg=False,
        lecam_weight=0.05,
        perceptual_model='vgg16',
        base_ch_disc=64,
        random_flip=False,
        flip_prob=0.5,
        flip_mode='stochastic',
        max_flip_lvl=1,
        not_load_optimizer=False,
        use_lecam_reg_zero=False,
        freeze_encoder=False,
        rm_downsample=False,
        random_flip_1lvl=False,
        flip_lvl_idx=0,
        drop_when_test=False,
        drop_lvl_idx=0,
        drop_lvl_num=1,
        disc_version='v1',
        magvit_disc=False,
        sigmoid_in_disc=False,
        activation_in_disc='leaky_relu',
        apply_blur=False,
        apply_noise=False,
        dis_warmup_steps=0,
        dis_lr_multiplier=1.0,
        dis_minlr_multiplier=False,
        disc_channels=64,
        disc_layers=3,
        discriminator_iter_start=0,
        disc_pretrain_iter=0,
        disc_optim_steps=1,
        disc_warmup=0,
        disc_pool='no',
        disc_pool_size=1000,
        advanced_disc=False,
        recon_loss_type='l1',
        video_perceptual_weight=0.0,
        image_gan_weight=1.0,
        video_gan_weight=1.0,
        image_disc_weight=0.0,
        video_disc_weight=0.0,
        l1_weight=4.0,
        gan_feat_weight=0.0,
        perceptual_weight=0.0,
        kl_weight=0.0,
        lfq_weight=0.0,
        entropy_loss_weight=0.1,
        commitment_loss_weight=0.25,
        diversity_gamma=1,
        norm_type='group',
        disc_loss_type='hinge',
        use_checkpoint=False,
        precision='fp32',
        encoder_dtype='fp32',
        upcast_attention='',
        upcast_tf32=False,
        tokenizer='flux',
        pretrained=None,
        pretrained_mode='full',
        inflation_pe=False,
        init_vgen='no',
        no_init_idis=False,
        init_idis='keep',
        init_vdis='no',
        enable_nan_detector=False,
        turn_on_profiler=False,
        profiler_scheduler_wait_steps=10,
        debug=True,
        video_logger=False,
        bytenas='',
        username='',
        seed=1234,
        vq_to_vae=False,
        load_not_strict=False,
        zero=0,
        bucket_cap_mb=40,
        manual_gc_interval=1000,
        data_path=[''],
        data_type=[''],
        dataset_list=['imagenet'],
        fps=-1,
        dataaug='resizecrop',
        multi_resolution=False,
        random_bucket_ratio=0.0,
        sequence_length=16,
        resolution=[256, 256],
        batch_size=[1],
        num_workers=0,
        image_channels=3,
        codebook_size=codebook_size,
        codebook_l2_norm=True,
        codebook_show_usage=True,
        commit_loss_beta=0.25,
        entropy_loss_ratio=0.0,
        base_ch=128,
        num_res_blocks=2,
        encoder_ch_mult=encoder_ch_mult,
        decoder_ch_mult=decoder_ch_mult,
        dropout_p=0.0,
        cnn_type='2d',
        cnn_version='v1',
        conv_in_out_2d='no',
        conv_inner_2d='no',
        res_conv_2d='no',
        cnn_attention='no',
        cnn_norm_axis='spatial',
        flux_weight=0,
        cycle_weight=0,
        cycle_feat_weight=0,
        cycle_gan_weight=0,
        cycle_loop=0,
        z_drop=0.0)
    
    vae = AutoEncoder(args)
    use_vae = vae.use_vae
    if not use_vae:
        num_codes = args.codebook_size
    if isinstance(vqgan_ckpt, str):
        state_dict = torch.load(args.vqgan_ckpt, map_location=torch.device("cpu"), weights_only=True)
    else:
        state_dict = args.vqgan_ckpt
    
    if state_dict:
        if args.ema == "yes":
            vae, new_state_dict, loaded_keys = load_cnn(vae, state_dict["ema"], prefix="", expand=False)
        else:
            vae, new_state_dict, loaded_keys = load_cnn(vae, state_dict["vae"], prefix="", expand=False)
    if test_mode:
        vae.eval()
        [p.requires_grad_(False) for p in vae.parameters()]
    return vae


def load_visual_tokenizer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load vae
    if args.vae_type in [16,18,20,24,32,64]:
        schedule_mode = "dynamic"
        codebook_dim = args.vae_type
        codebook_size = 2**codebook_dim
        if args.apply_spatial_patchify:
            patch_size = 8
            encoder_ch_mult=[1, 2, 4, 4]
            decoder_ch_mult=[1, 2, 4, 4]
        else:
            patch_size = 16
            encoder_ch_mult=[1, 2, 4, 4, 4]
            decoder_ch_mult=[1, 2, 4, 4, 4]
        vae = vae_model(args.vae_path, schedule_mode, codebook_dim, codebook_size, patch_size=patch_size, 
                        encoder_ch_mult=encoder_ch_mult, decoder_ch_mult=decoder_ch_mult, test_mode=True).to(device)
    else:
        raise ValueError(f'vae_type={args.vae_type} not supported')
    return vae


################################
# Here is main model building blocks part
################################

@torch.compile(fullgraph=True)
def fused_rms_norm(x: torch.Tensor, weight: nn.Parameter, eps: float):
    x = x.float()
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True).add_(eps))) * weight


@torch.compile(fullgraph=True)
def fused_ada_layer_norm(C: int, eps: float, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor):
    x = x.float()
    x = F.layer_norm(input=x, normalized_shape=(C,), weight=None, bias=None, eps=eps)
    return x.mul(scale.add(1)).add_(shift)


@torch.compile(fullgraph=True)
def fused_ada_rms_norm(C: int, eps: float, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor):
    x = x.float()
    x = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True).add_(eps)))
    return x.mul(scale.add(1)).add_(shift)

def precompute_rope2d_freqs_grid(dim, dynamic_resolution_h_w, rope2d_normalized_by_hw, pad_to_multiplier=1, max_height=2048 // 16, max_width=2048 // 16, base=10000.0, device=None, scaling_factor=1.0):
    # split the dimension into half, one for x and one for y
    half_dim = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, 2, dtype=torch.int64).float().to(device) / half_dim)) # namely theta, 1 / (10000^(i/half_dim)), i=0,2,..., half_dim-2
    t_height = torch.arange(max_height, device=device, dtype=torch.int64).type_as(inv_freq)
    t_width = torch.arange(max_width, device=device, dtype=torch.int64).type_as(inv_freq)
    t_height = t_height / scaling_factor
    freqs_height = torch.outer(t_height, inv_freq)  # (max_height, dim / (1 for 1d, 2 for 2d, 3 for 3d) / 2), namely y*theta
    t_width = t_width / scaling_factor
    freqs_width = torch.outer(t_width, inv_freq)  # (max_width, dim / (1 for 1d, 2 for 2d, 3 for 3d) / 2), namely x*theta
    freqs_grid_map = torch.concat([
        freqs_height[:, None, :].expand(-1, max_width, -1), # (max_height, max_width, dim / (1 for 1d, 2 for 2d, 3 for 3d) / 2)
        freqs_width[None, :, :].expand(max_height, -1, -1), # (max_height, max_width, dim / (1 for 1d, 2 for 2d, 3 for 3d) / 2)
    ], dim=-1)  # (max_height, max_width, dim / (1 for 1d, 2 for 2d, 3 for 3d))
    freqs_grid_map = torch.stack([torch.cos(freqs_grid_map), torch.sin(freqs_grid_map)], dim=0)
    # (2, max_height, max_width, dim / (1 for 1d, 2 for 2d, 3 for 3d))

    rope2d_freqs_grid = {}
    for h_div_w in dynamic_resolution_h_w:
        scale_schedule = dynamic_resolution_h_w[h_div_w]['1M']['scales']
        _, ph, pw = scale_schedule[-1]
        max_edge_length = freqs_grid_map.shape[1]
        if ph >= pw:
            uph, upw = max_edge_length, int(max_edge_length / ph * pw)
        else:
            uph, upw = int(max_edge_length / pw * ph), max_edge_length
        rope_cache_list = []
        for (_, ph, pw) in scale_schedule:
            ph_mul_pw = ph * pw
            if rope2d_normalized_by_hw == 1: # downsample
                rope_cache = F.interpolate(freqs_grid_map[:, :uph, :upw, :].permute([0,3,1,2]), size=(ph, pw), mode='bilinear', align_corners=True)
                rope_cache = rope_cache.permute([0,2,3,1]) # (2, ph, pw, half_head_dim)
            elif rope2d_normalized_by_hw == 2: # star stylee
                _, uph, upw = scale_schedule[-1]
                indices = torch.stack([
                    (torch.arange(ph) * (uph / ph)).reshape(ph, 1).expand(ph, pw),
                    (torch.arange(pw) * (upw / pw)).reshape(1, pw).expand(ph, pw),
                ], dim=-1).round().int() # (ph, pw, 2)
                indices = indices.reshape(-1, 2) # (ph*pw, 2)
                rope_cache = freqs_grid_map[:, indices[:,0], indices[:,1], :] # (2, ph*pw, half_head_dim)
                rope_cache = rope_cache.reshape(2, ph, pw, -1)
            elif rope2d_normalized_by_hw == 0:
                rope_cache = freqs_grid_map[:, :ph, :pw, :] # (2, ph, pw, half_head_dim)
            else:
                raise ValueError(f'Unknown rope2d_normalized_by_hw: {rope2d_normalized_by_hw}')
            rope_cache_list.append(rope_cache.reshape(2, ph_mul_pw, -1))
        cat_rope_cache = torch.cat(rope_cache_list, 1) # (2, seq_len, half_head_dim)
        if cat_rope_cache.shape[1] % pad_to_multiplier:
            pad = torch.zeros(2, pad_to_multiplier - cat_rope_cache.shape[1] % pad_to_multiplier, half_dim)
            cat_rope_cache = torch.cat([cat_rope_cache, pad], dim=1)
        cat_rope_cache = cat_rope_cache[:,None,None,None] # (2, 1, 1, 1, seq_len, half_dim)
        for pn in dynamic_resolution_h_w[h_div_w]:
            scale_schedule = dynamic_resolution_h_w[h_div_w][pn]['scales']
            tmp_scale_schedule = [(1, h, w) for _, h, w in scale_schedule]
            rope2d_freqs_grid[str(tuple(tmp_scale_schedule))] = cat_rope_cache
    return rope2d_freqs_grid

def apply_rotary_emb(q, k, scale_schedule, rope2d_freqs_grid, pad_to_multiplier, rope2d_normalized_by_hw, scale_ind):
    qk = torch.stack((q, k), dim=0)  #(2, batch_size, heads, seq_len, head_dim)
    device_type = qk.device.type
    device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        seq_len = qk.shape[3]
        start = 0
        if scale_ind >= 1:
            assert len(scale_schedule[0]) == 3
            start = np.sum([item[0] * item[1] * item[2] for item in scale_schedule[:scale_ind]])
        rope2d_freqs_grid[str(tuple(scale_schedule))] = rope2d_freqs_grid[str(tuple(scale_schedule))].to(qk.device)
        assert start+seq_len <= rope2d_freqs_grid[str(tuple(scale_schedule))].shape[4]
        rope_cache = rope2d_freqs_grid[str(tuple(scale_schedule))][:, :, :, :, start:start+seq_len] # rope_cache shape: [2, 1, 1, 1, seq_len, half_head_dim]
        qk = qk.reshape(*qk.shape[:-1], -1, 2) #(2, batch_size, heads, seq_len, half_head_dim, 2)
        qk = torch.stack([
            rope_cache[0] * qk[...,0] - rope_cache[1] * qk[...,1],
            rope_cache[1] * qk[...,0] + rope_cache[0] * qk[...,1],
        ], dim=-1) # (2, batch_size, heads, seq_len, half_head_dim, 2), here stack + reshape should not be concate
        qk = qk.reshape(*qk.shape[:-2], -1) #(2, batch_size, heads, seq_len, head_dim)
        q, k = qk.unbind(dim=0) # (batch_size, heads, seq_len, head_dim)
    return q, k


class FastRMSNorm(nn.Module):
    def __init__(self, C, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.C = C
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(C))
        else:
            self.register_buffer('weight', torch.ones(C))
    
    def forward(self, x):
        src_type = x.dtype
        return rms_norm_impl(x.float(), self.weight, epsilon=self.eps).to(src_type)
    
    def extra_repr(self) -> str:
        return f'C={self.C}, eps={self.eps:g}, elementwise_affine={self.elementwise_affine}'


def get_dropout_layer(p):
    return nn.Dropout(p, inplace=True) if p > 0 else nn.Identity()


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., fused_mlp=False):
        super().__init__()
        self.fused_mlp_func = fused_mlp_func if fused_mlp else None
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = get_dropout_layer(drop)
        self.heuristic = -1
    
    def forward(self, x):
        if self.fused_mlp_func is not None:
            return self.drop(self.fused_mlp_func(
                x=x,
                weight1=self.fc1.weight,
                weight2=self.fc2.weight,
                bias1=self.fc1.bias,
                bias2=self.fc2.bias,
                activation='gelu_approx',
                save_pre_act=self.training,
                return_residual=False,
                checkpoint_lvl=0,
                heuristic=self.heuristic,
                process_group=None,
            ))
        else:
            return self.drop(self.fc2( self.act(self.fc1(x)) ))
    
    def extra_repr(self) -> str:
        return f'fused_mlp={self.fused_mlp_func is not None}'


class FFNSwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=None, drop=0., fused_mlp=False):
        super().__init__()
        self.fused_mlp_func = None
        hidden_features = round(2 * hidden_features / 3 / 256) * 256
        
        out_features = out_features or in_features
        self.fcg = nn.Linear(in_features, hidden_features, bias=False)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
        self.drop = get_dropout_layer(drop)
    
    def forward(self, x):
        return self.drop(self.fc2( F.silu(self.fcg(x), inplace=True).mul_(self.fc1(x)) ))
    
    def extra_repr(self) -> str:
        return f'fused_mlp={self.fused_mlp_func is not None}'




def _causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

def _length_to_offsets(lengths, device):
    """Converts a list of lengths to a list of offsets.

    Args:
        lengths: A list of lengths.

    """
    offsets = [0]
    offsets.extend(lengths)
    offsets = torch.tensor(offsets, device=device, dtype=torch.int32)
    offsets = torch.cumsum(offsets, dim=-1)
    return offsets

def _generate_var_mask_mod(offsets):
    """Generates mask mods that apply to inputs to flex attention in the sequence stacked
    format.

    Args:
        offsets: This tensor should be of shape(num_documents + 1)
            this should contain the cumulative counts of document tokens.
            e.g. if you have 3 documents of length 2, 4, 3 then
            offsets = [0, 2, 6, 9]

    Note:
        What is the sequence stacked format? When assembling batches of inputs, we
        take multiple sequences and stack them together to form 1 large sequence. We then
        use masking to ensure that the attention scores are only applied to tokens within
        the same document.
    """

    def _offsets_to_doc_ids_tensor(offsets):
        device = offsets.device
        counts = offsets[1:] - offsets[:-1]
        return torch.repeat_interleave(
            torch.arange(len(counts), device=device, dtype=torch.int32), counts
        )

    document_id = _offsets_to_doc_ids_tensor(offsets)

    def var_mask_mod(b, h, q_idx, kv_idx):
        same_doc = document_id[q_idx] == document_id[kv_idx]
        causal_mask = _causal_mask(b, h, q_idx, kv_idx)
        return same_doc | causal_mask

    return var_mask_mod

def _generate_var_infer_mask_with_kv_cache(lengths):
    kv_len = sum(lengths)
    def var_mask_mod(b, h, q_idx, kv_idx):
        return kv_idx < kv_len

    return var_mask_mod

class FlexAttn(nn.Module):
    def __init__(
            self, block_scales:list, mask_type:str, B, H, L:int, auto_padding=False
    ):
        """
        :param block_scales: accept VAR's block sizes like [(1,1), (2,2), (3,3)]
        :param mask_type: var/causal
        :param B: batch size
        :param H: heads num
        :param L: sequence length
        """
        super().__init__()
        if not flex_attention_available:
            raise NotImplementedError((f"[Error] flex attention need pytorch 2.5.0+ but your version is {torch.__version__}"))

        self.support_mask_type = ["var", "causal", "var_infer_mask_with_kv_cache"]
        self.auto_padding = auto_padding

        self.flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")

        self.block_scales = block_scales
        self.lengths = [ x * y * z for x,y,z in block_scales]

        self.offsets = _length_to_offsets(self.lengths, device='cuda')

        # if L paded to align 128, block need to cover padding area
        if self.offsets[-1] < L:
            self.offsets = torch.cat((self.offsets, torch.tensor([L], device='cuda')), dim=0)

        if mask_type == "var":
            self.mask_mod = _generate_var_mask_mod(self.offsets)
            self.block_mask = create_block_mask(self.mask_mod, B = B, H = H, Q_LEN = L, KV_LEN = L, device = 'cuda', _compile = True)
        elif mask_type == "causal":
            self.mask_mod = _causal_mask
            self.block_mask = create_block_mask(self.mask_mod, B = B, H = H, Q_LEN = L, KV_LEN = L, device = 'cuda', _compile = True)
        elif mask_type == 'var_infer_mask_with_kv_cache':
            self.mask_mod = _generate_var_infer_mask_with_kv_cache(self.lengths)
            self.block_mask = create_block_mask(self.mask_mod, B = B, H = H, Q_LEN = L, KV_LEN = L, device = 'cuda', _compile = True)
        else:
            raise NotImplementedError(f"{mask_type} not supportted in FlexAttn, support type:{self.support_mask_type}")


    def forward(self, q, k, v, scale = None):
        if self.auto_padding:
            q_pad_len = (128 - q.shape[-2] % 128) % 128
            kv_pad_len = (128 - k.shape[-2] % 128) % 128
            q_pad = F.pad(q, (0, 0, 0, q_pad_len))
            k_pad = F.pad(k, (0, 0, 0, kv_pad_len))
            v_pad = F.pad(v, (0, 0, 0, kv_pad_len))
            oup = self.flex_attention(q_pad.to(v_pad.dtype), k_pad.to(v.dtype), v_pad, block_mask = self.block_mask, scale = scale)
            if q_pad_len > 0:
                oup = oup[:,:,:-q_pad_len]
        else:
            oup = self.flex_attention(q.to(v.dtype), k.to(v.dtype), v, block_mask = self.block_mask, scale = scale)
        return oup


class SelfAttention(nn.Module):
    def __init__(
        self, embed_dim=768, num_heads=12,
        proj_drop=0., tau=1, cos_attn=False, customized_flash_attn=True, use_flex_attn=False, 
        batch_size=2, pad_to_multiplier=1, rope2d_normalized_by_hw=0,
    ):
        """
        :param embed_dim: model's width
        :param num_heads: num heads of multi-head attention
        :param proj_drop: always 0 for testing
        :param tau: always 1
        :param cos_attn: always True: during attention, q and k will be L2-normalized and scaled by a head-wise learnable parameter self.scale_mul_1H11
        :param customized_flash_attn:
        """
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
        self.tau, self.cos_attn = tau, cos_attn
        if self.cos_attn:
            self.scale = 1
            size = (1, self.num_heads, 1, 1)
            # size: 11H1 or 1H11
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=size, fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 1 / math.sqrt(self.head_dim) / self.tau
        
        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(embed_dim)), nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = get_dropout_layer(proj_drop)
        
        self.caching = False    # kv caching: only used during inference
        self.cached_k = None    # kv caching: only used during inference
        self.cached_v = None    # kv caching: only used during inference

        self.batch_size = batch_size
        self.use_flex_attn = use_flex_attn
        self.pad_to_multiplier = pad_to_multiplier

        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw

    
    def kv_caching(self, enable: bool): # kv caching: only used during inference
        self.caching = enable
        self.cached_k = None
        self.cached_v = None
    
    # NOTE: attn_bias_or_two_vector is None during inference
    def forward(self, x, attn_bias_or_two_vector: Union[torch.Tensor, Tuple[torch.IntTensor, torch.IntTensor]], attn_fn=None, scale_schedule=None, rope2d_freqs_grid=None, scale_ind=0):
        """
        :param (fp32) x: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be shared
        :param (fp32) attn_bias_or_two_vector:
                if not using_flash:
                    a block-wise, lower-triangle matrix, like:
                    [[[[0, -, -, -, -, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]]
                    where 0 means visible and - means invisible (-inf)
                else:
                    a tuple of two 1-dim int vector (VAR_visible_kvlen, VAR_invisible_qlen)
        :return: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be shared
        """
        # x: fp32
        B, L, C = x.shape
        
        # qkv: amp, bf16
        qkv = F.linear(input=x, weight=self.mat_qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))).view(B, L, 3, self.num_heads, self.head_dim)  # BL3Hc
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0); L_dim = 2   # q or k or v: all are shaped in (B:batch_size, H:heads, L:seq_len, c:head_dim)
        
        scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp() # 11H1 (flash), or 1H11 (not flash)
        q = F.normalize(q, dim=-1, eps=1e-12).mul(scale_mul).contiguous()   # fp32
        k = F.normalize(k, dim=-1, eps=1e-12).contiguous()                  # fp32
        v = v.contiguous()                                                  # bf16
        
        if rope2d_freqs_grid is not None:
            q, k = apply_rotary_emb(q, k, scale_schedule, rope2d_freqs_grid, self.pad_to_multiplier, self.rope2d_normalized_by_hw, scale_ind) #, freqs_cis=freqs_cis)
        if self.caching:
            if not v.requires_grad: # during inference, we can concat
                if self.cached_k is None: self.cached_k = k; self.cached_v = v
                else: k = self.cached_k = torch.cat((self.cached_k, k), dim=L_dim); v = self.cached_v = torch.cat((self.cached_v, v), dim=L_dim)
            else: self.cached_k = k; self.cached_v = v # during training, kv_cache is used to return k,v, so always update in-place
            
        if self.use_flex_attn and attn_fn is not None:
            oup = attn_fn(q, k, v, scale=self.scale).transpose(1, 2).reshape(B, L, C)
        else:
            oup = slow_attn(query=q, key=k, value=v, scale=self.scale, attn_mask=attn_bias_or_two_vector, dropout_p=0).transpose(1, 2).reshape(B, L, C)
        # oup: bf16
        
        if self.caching:
            return self.proj_drop(self.proj(oup)), self.cached_k, self.cached_v
        else:
            return self.proj_drop(self.proj(oup)), None, None # NOTE: None are just placeholders
    

class CrossAttention(nn.Module):
    def __init__(
        self, for_attn_pool=False, embed_dim=768, kv_dim=4096, num_heads=12,
        proj_drop=0., cos_attn=False,
    ):
        """
        :param for_attn_pool: only used in VAR.text_proj_for_sos
        :param embed_dim: Q's dim
        :param kv_dim: K's and V's dim
        :param num_heads: num heads of multi-head attention
        :param proj_drop: proj drop out
        :param cos_attn: during attention, q and k will be L2-normalized and scaled by a head-wise learnable parameter self.scale_mul_1H11
        """
        super().__init__()
        self.for_attn_pool = for_attn_pool
        self.embed_dim = embed_dim
        self.kv_dim = kv_dim
        assert embed_dim % num_heads == 0
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads  # =64
        self.cos_attn = cos_attn
        self.scale = 1 / math.sqrt(self.head_dim)
        
        if for_attn_pool:
            q = torch.empty(1, self.num_heads, self.head_dim)
            nn.init.trunc_normal_(q, mean=0, std=math.sqrt(1 / embed_dim / 3))
            self.mat_q = nn.Parameter(q)
        else:
            self.mat_q = nn.Linear(embed_dim, embed_dim, bias=True)
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = get_dropout_layer(proj_drop)
    
    def forward(self, q, ca_kv):
        """
        :param q: shaped as (batch, seq_len, Q_dim)
        :param ca_kv: contains several vectors, each of which is shaped as (len_i, KV_dim). We have [len_1xKV_dim, len_2xKV_dim, len_3xKV_dim, ...] and lens == [len_1, len_2, len_3, ...]
            - kv_compact: shaped as (sum(lens), 2, KV_dim)
            - cu_seqlens_k: cumulated sum of lens
            - max_seqlen_k: int, max(lens)
        NOTE: seq_len (num of Qs) can reach 10k;  but len_i (num of KVs) must <= 256
        
        :return: shaped as (batch, seq_len, Q_dim)
        """
        kv_compact, cu_seqlens_k, max_seqlen_k = ca_kv
        N = kv_compact.shape[0]
        
        kv_compact = kv_compact.view(N, 2, self.num_heads, self.head_dim) # N2C => N2Hc
        
        if not self.for_attn_pool:
            B, Lq = q.shape[:2]
            q_compact = self.mat_q(q).view(-1, self.num_heads, self.head_dim)
        else:
            B = cu_seqlens_k.shape[0] - 1
            Lq = 1
            q_compact = self.mat_q.repeat(B, 1, 1).to(dtype=kv_compact.dtype)
        
        q_compact = q_compact.contiguous()
        kv_compact = kv_compact.contiguous()
        
        cu_seqlens_q = torch.arange(0, Lq * (B+1), Lq, dtype=torch.int32, device=q_compact.device)
        if q_compact.dtype == torch.float32:
            oup = flash_attn_varlen_kvpacked_func(q=q_compact.to(dtype=torch.bfloat16), kv=kv_compact.to(dtype=torch.bfloat16), cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=Lq, max_seqlen_k=max_seqlen_k, dropout_p=0, softmax_scale=self.scale).reshape(B, Lq, -1)
            oup = oup.float()
        else:
            oup = flash_attn_varlen_kvpacked_func(q=q_compact, kv=kv_compact, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=Lq, max_seqlen_k=max_seqlen_k, dropout_p=0, softmax_scale=self.scale).reshape(B, Lq, -1)
        
        return self.proj_drop(self.proj(oup))

class TextAttentivePool(nn.Module):
    def __init__(self, Ct5: int, D: int):
        super().__init__()
        self.Ct5, self.D = Ct5, D
        if D > 4096:
            self.head_dim = 64 
        else:
            self.head_dim = 128

        self.num_heads = Ct5 // self.head_dim
        self.ca = CrossAttention(for_attn_pool=True, embed_dim=self.D, kv_dim=Ct5, num_heads=self.num_heads)
    def forward(self, ca_kv):
        return self.ca(None, ca_kv).squeeze(1)

class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).reshape(-1, 1, 6, C)   # B16C


class MultipleLayers(nn.Module):
    def __init__(self, ls, num_blocks_in_a_chunk, index):
        super().__init__()
        self.module = nn.ModuleList()
        for i in range(index, index+num_blocks_in_a_chunk):
            self.module.append(ls[i])

    def forward(self, x, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn=None, scale_schedule=None, checkpointing_full_block=False, rope2d_freqs_grid=None):
        h = x
        for m in self.module:
            if checkpointing_full_block:
                h = torch.utils.checkpoint.checkpoint(m, h, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid, use_reentrant=False)
            else:
                h = m(h, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid)
        return h

class AdaLNBeforeHead(nn.Module):
    def __init__(self, C, D, act: bool, norm_layer: partial, fused_norm_func=None):   # C: embed_dim, D: cond_dim
        super().__init__()
        self.C, self.D = C, D
        self.ln_wo_grad = norm_layer(C, elementwise_affine=False)
        self.fused_norm_func = fused_norm_func
        self.norm_eps = norm_layer.keywords.get('eps', 1e-6)
        lin = nn.Linear(D, 2*C)
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin) if act else nn.Sequential(lin)
    
    def forward(self, x_BLC: torch.Tensor, cond_BD: Optional[torch.Tensor]):
        scale, shift = self.ada_lin(cond_BD).view(-1, 1, 2, self.C).unbind(2)
        if self.fused_norm_func is None:
            return self.ln_wo_grad(x_BLC).mul(scale.add(1)).add_(shift)
        else:
            return self.fused_norm_func(C=self.C, eps=self.norm_eps, x=x_BLC, scale=scale, shift=shift)

def sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV: torch.Tensor, top_k: int = 0, top_p: float = 0.0, rng=None, num_samples=1) -> torch.Tensor:  # return idx, shaped (B, l)
    B, l, V = logits_BlV.shape
    if top_k > 0:
        top_k = min(top_k, V)
        idx_to_remove = logits_BlV < logits_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
        logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
    if top_p > 0:
        sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        logits_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), -torch.inf)
    # sample (have to squeeze cuz multinomial can only be used on 2D tensor)
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(logits_BlV.softmax(dim=-1).view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)