"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ['PacConv2d', 'PacConvTranspose2d', 'PacPool2d',
           'pacconv2d', 'pacconv_transpose2d', 'pacpool2d', 'packernel2d', 'nd2col']

import math
from numbers import Number
from itertools import repeat

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function, once_differentiable
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
from torch._thnn import type2backend

try:
    import pyinn as P

    has_pyinn = True
except ImportError:
    P = None
    has_pyinn = False
    pass


def _neg_idx(idx):
    return None if idx == 0 else -idx


# 가우시안 필터를 리턴하는 함수
def np_gaussian_2d(width, sigma=-1):
    '''Truncated 2D Gaussian filter'''
    assert width % 2 == 1
    if sigma <= 0:
        sigma = float(width) / 4

    r = np.arange(-(width // 2), (width // 2) + 1, dtype=np.float32)
    gaussian_1d = np.exp(-0.5 * r * r / (sigma * sigma))
    gaussian_2d = gaussian_1d.reshape(-1, 1) * gaussian_1d
    gaussian_2d /= gaussian_2d.sum()

    return gaussian_2d

# https://cding.tistory.com/112 참조, 컨볼루션을 직접 구현할 때 컨볼루션을 적용하기 쉬운 형태로 변형해주는 함수.
def nd2col(input_nd, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, transposed=False,
           use_pyinn_if_possible=False):
    """
    Shape:
        - Input: :math:`(N, C, L_{in})` # 여기서 N은 배치 개수, C는 채널 개수, L_{in} 길이 튜플 (이미지의 h, w... 등의 길이)
        - Output: :math:`(N, C, *kernel_size, *L_{out})` where # 여기서 kernel_size 앞의 애스터리스크는 가변인자 사용 표기법이다.
          :math:`L_{out} = floor((L_{in} + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)` for non-transposed
          :math:`L_{out} = (L_{in} - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding` for transposed
    """
    n_dims = len(input_nd.shape[2:])
    kernel_size = (kernel_size,) * n_dims if isinstance(kernel_size, Number) else kernel_size # 튜플인지 숫자인지 검사해서 숫자이면 튜플로 변형.
    stride = (stride,) * n_dims if isinstance(stride, Number) else stride
    padding = (padding,) * n_dims if isinstance(padding, Number) else padding
    output_padding = (output_padding,) * n_dims if isinstance(output_padding, Number) else output_padding
    dilation = (dilation,) * n_dims if isinstance(dilation, Number) else dilation

    if transposed:
        assert n_dims == 2, 'Only 2D is supported for fractional strides.' # assertion 에러 메시지
        w_one = input_nd.new_ones(1, 1, 1, 1)
        pad = [(k - 1) * d - p for (k, d, p) in zip(kernel_size, dilation, padding)]
        input_nd = F.conv_transpose2d(input_nd, w_one, stride=stride)
        input_nd = F.pad(input_nd, (pad[1], pad[1] + output_padding[1], pad[0], pad[0] + output_padding[0]))
        stride = _pair(1) # 숫자를 튜플로 변형. 위에 isinstance(, Number) 해놓은거랑 용도 비슷한듯
        padding = _pair(0)

    (bs, nch), in_sz = input_nd.shape[:2], input_nd.shape[2:]
    out_sz = tuple([((i + 2 * p - d * (k - 1) - 1) // s + 1) # 출력 사이즈 계산식
                    for (i, k, d, p, s) in zip(in_sz, kernel_size, dilation, padding, stride)]) # (h, w...에 대해 반복)
    # Use PyINN if possible (about 15% faster) TODO confirm the speed-up
    if n_dims == 2 and dilation == 1 and has_pyinn and torch.cuda.is_available() and use_pyinn_if_possible:
        output = P.im2col(input_nd, kernel_size, stride, padding)
    else:
        output = F.unfold(input_nd, kernel_size, dilation, padding, stride)
        out_shape = (bs, nch) + tuple(kernel_size) + out_sz
        output = output.view(*out_shape).contiguous()
    return output


# 출력: unfold된 유사도 가우시안 커널.
class GaussKernel2dFn(Function):
    @staticmethod
    def forward(ctx, input, kernel_size, stride, padding, dilation, channel_wise):
        ctx.kernel_size = _pair(kernel_size)
        ctx.dilation = _pair(dilation)
        ctx.padding = _pair(padding)
        ctx.stride = _pair(stride)
        bs, ch, in_h, in_w = input.shape
        out_h = (in_h + 2 * ctx.padding[0] - ctx.dilation[0] * (ctx.kernel_size[0] - 1) - 1) // ctx.stride[0] + 1 # 출력 사이즈 계산식, 아래의 L은 out_h * out_w
        out_w = (in_w + 2 * ctx.padding[1] - ctx.dilation[1] * (ctx.kernel_size[1] - 1) - 1) // ctx.stride[1] + 1
        # https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
        # Convolution is equivalent with Unfold + Matrix Multiplication + Fold (or view to output shape)
        # https://cding.tistory.com/112
        cols = F.unfold(input, ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride) # im2col, 입력 차원: (N, C, *) 일때 출력 차원: (N, C * 커널 크기, L), 기호의 의미는 위 링크 참조.
        cols = cols.view(bs, ch, ctx.kernel_size[0], ctx.kernel_size[1], out_h, out_w) # (N, C * 커널 크기, L) -> (N, C, kernel_h, kernel_w, out_h, out_w)
        center_y, center_x = ctx.kernel_size[0] // 2, ctx.kernel_size[1] // 2
        feat_0 = cols.contiguous()[:, :, center_y:center_y + 1, center_x:center_x + 1, :, :] # 커널 중앙에 위치하는 input의 feature
        diff_sq = (cols - feat_0).pow(2) # 차분의 제곱
        if not channel_wise:
            diff_sq = diff_sq.sum(dim=1, keepdim=True) # 채널 축에 대한 노름으로. (N, C=1, kernel_h, kernel_w, out_h, out_w)
        output = torch.exp(-0.5 * diff_sq) # 차이가 작으면 1, 차이가 크면 0에 가까운 마스크
        ctx._backend = type2backend[input.type()]
        ctx.save_for_backward(input, output)

        return output # (N, C=1, kernel_h, kernel_w, out_h, out_w), unfold 상태. 이거를 unfold된 input과 곱하면 각 위치마다 위치에 적응된 마스크를 씌우는 효과를 냄.

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        bs, ch, in_h, in_w = input.shape
        out_h, out_w = output.shape[-2:]
        cols = F.unfold(input, ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)
        cols = cols.view(bs, ch, ctx.kernel_size[0], ctx.kernel_size[1], out_h, out_w)
        center_y, center_x = ctx.kernel_size[0] // 2, ctx.kernel_size[1] // 2
        feat_0 = cols.contiguous()[:, :, center_y:center_y + 1, center_x:center_x + 1, :, :]
        diff = cols - feat_0
        grad = -0.5 * grad_output * output
        grad_diff = grad.expand_as(cols) * (2 * diff)
        grad_diff[:, :, center_y:center_y + 1, center_x:center_x + 1, :, :] -= \
            grad_diff.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
        grad_input = grad_output.new()
        ctx._backend.Im2Col_updateGradInput(ctx._backend.library_state,
                                            grad_diff.view(bs, ch * ctx.kernel_size[0] * ctx.kernel_size[1], -1),
                                            grad_input,
                                            in_h, in_w,
                                            ctx.kernel_size[0], ctx.kernel_size[1],
                                            ctx.dilation[0], ctx.dilation[1],
                                            ctx.padding[0], ctx.padding[1],
                                            ctx.stride[0], ctx.stride[1])

        return grad_input, None, None, None, None, None

# GU
class EdgeKernel2d(nn.Module):
    
    def __init__(self, in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1):
        super(EdgeKernel2d, self).__init__()
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
        self.convt = nn.ConvTranspose2d(out_channels, 2, kernel_size=kernel_size, stride=stride, padding=0) # TODO 2
        
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        
    def forward(self, input):
        bs, ch, in_h, in_w = input.shape

        assert ch == 1 or ch == 2, "ch: " + str(ch)

        out_h = (in_h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1 # 출력 사이즈 계산식, 아래의 L은 out_h * out_w
        out_w = (in_w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        cols = F.unfold(input, self.kernel_size, self.dilation, self.padding, self.stride) # im2col, 입력 차원: (N, C, *) 일때 출력 차원: (N, C * 커널 크기, L)
        cols = cols.view(bs * self.kernel_size[0] * self.kernel_size[1], ch, out_h, out_w) # (N, C * 커널 크기, L) -> (N, C, kernel_h, kernel_w, out_h, out_w)
        
        enc = self.conv(cols)
        enc = self.softmax(enc)
        dec = self.convt(enc)
        y = self.softmax(dec)

        w = torch.tensor(np.array([0.,1.]).reshape(1,2,1,1), dtype=torch.float, device=input.device)
        y = F.conv2d(y, weight=w) # TODO 2
        
        y = y.view(bs, 1, self.kernel_size[0], self.kernel_size[1], out_h, out_w)
        
        return y

# 입력을 unfold한 다음 이미 unfold된 커널을 받아서 곱함.
class PacConv2dFn(Function):
    @staticmethod
    def forward(ctx, input, kernel, weight, bias=None, stride=1, padding=0, dilation=1, shared_filters=False):
        (bs, ch), in_sz = input.shape[:2], input.shape[2:]
        if kernel.size(1) > 1:
            raise ValueError('Non-singleton channel is not allowed for kernel.') # (N, C=1, kernel_h, kernel_w, out_h, out_w)
        ctx.input_size = in_sz
        ctx.in_ch = ch
        ctx.kernel_size = tuple(weight.shape[-2:])        
        ctx.dilation = _pair(dilation)
        ctx.padding = _pair(padding)
        ctx.stride = _pair(stride)
        ctx.shared_filters = shared_filters
        ctx.save_for_backward(input if (ctx.needs_input_grad[1] or ctx.needs_input_grad[2]) else None,
                              kernel if (ctx.needs_input_grad[0] or ctx.needs_input_grad[2]) else None,
                              weight if (ctx.needs_input_grad[0] or ctx.needs_input_grad[1]) else None)
        ctx._backend = type2backend[input.type()]

        cols = F.unfold(input, ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)
        
        in_mul_k = cols.view(bs, ch, *kernel.shape[2:]) * kernel # 유사도 가우시안 필터 적용
        # (N, C, kernel_h, kernel_w, out_h, out_w) * (N, C=1, kernel_h, kernel_w, out_h, out_w)

        # https://pytorch.org/docs/stable/generated/torch.einsum.html
        # https://rockt.github.io/2018/04/30/einsum
        # fold 작업을 포함함.
        # matrix multiplication, written as an einsum to avoid repeated view() and permute()
        if shared_filters:
            output = torch.einsum('ijklmn,zykl->ijmn', (in_mul_k, weight))
        else:
            output = torch.einsum('ijklmn,ojkl->iomn', (in_mul_k, weight))
            # (N, C(in), kernel_h, kernel_w, out_h, out_w), (out, in, kernel_h, kernel_w) => (N, C(out), out_h, out_w)

        if bias is not None:
            output += bias.view(1, -1, 1, 1) # 바이어스 적용

        return output.clone()  # TODO understand why a .clone() is needed here

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_input = grad_kernel = grad_weight = grad_bias = None
        (bs, out_ch), out_sz = grad_output.shape[:2], grad_output.shape[2:]
        in_ch = ctx.in_ch

        input, kernel, weight = ctx.saved_tensors
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            if ctx.shared_filters:
                grad_in_mul_k = grad_output.view(bs, out_ch, 1, 1, out_sz[0], out_sz[1]) \
                                * weight.view(ctx.kernel_size[0], ctx.kernel_size[1], 1, 1)
            else:
                grad_in_mul_k = torch.einsum('iomn,ojkl->ijklmn', (grad_output, weight))
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            in_cols = F.unfold(input, ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)
            in_cols = in_cols.view(bs, in_ch, ctx.kernel_size[0], ctx.kernel_size[1], out_sz[0], out_sz[1])
        if ctx.needs_input_grad[0]: # grad_input
            grad_input = grad_output.new()
            grad_im2col_output = grad_in_mul_k * kernel
            grad_im2col_output = grad_im2col_output.view(bs, -1, out_sz[0] * out_sz[1])
            ctx._backend.Im2Col_updateGradInput(ctx._backend.library_state,
                                                grad_im2col_output,
                                                grad_input,
                                                ctx.input_size[0], ctx.input_size[1],
                                                ctx.kernel_size[0], ctx.kernel_size[1],
                                                ctx.dilation[0], ctx.dilation[1],
                                                ctx.padding[0], ctx.padding[1],
                                                ctx.stride[0], ctx.stride[1])
        if ctx.needs_input_grad[1]: # grad_kernel
            grad_kernel = in_cols * grad_in_mul_k
            grad_kernel = grad_kernel.sum(dim=1, keepdim=True)
        if ctx.needs_input_grad[2]: # grad_weight
            in_mul_k = in_cols * kernel
            if ctx.shared_filters:
                grad_weight = torch.einsum('ijmn,ijklmn->kl', (grad_output, in_mul_k))
                grad_weight = grad_weight.view(1, 1, ctx.kernel_size[0], ctx.kernel_size[1]).contiguous()
            else:
                grad_weight = torch.einsum('iomn,ijklmn->ojkl', (grad_output, in_mul_k))
        if ctx.needs_input_grad[3]:
            grad_bias = torch.einsum('iomn->o', (grad_output,))

        return grad_input, grad_kernel, grad_weight, grad_bias, None, None, None, None


class PacConvTranspose2dFn(Function):
    @staticmethod
    def forward(ctx, input, kernel, weight, bias=None, stride=1, padding=0, output_padding=0, dilation=1,
                shared_filters=False):
        (bs, ch), in_sz = input.shape[:2], input.shape[2:]
        if kernel.size(1) > 1:
            raise ValueError('Non-singleton channel is not allowed for kernel.')
        ctx.in_ch = ch
        ctx.kernel_size = tuple(weight.shape[-2:])
        ctx.dilation = _pair(dilation)
        ctx.padding = _pair(padding)
        ctx.output_padding = _pair(output_padding)
        ctx.stride = _pair(stride)
        ctx.shared_filters = shared_filters
        ctx.save_for_backward(input if (ctx.needs_input_grad[1] or ctx.needs_input_grad[2]) else None,
                              kernel if (ctx.needs_input_grad[0] or ctx.needs_input_grad[2]) else None,
                              weight if (ctx.needs_input_grad[0] or ctx.needs_input_grad[1]) else None)
        ctx._backend = type2backend[input.type()]

        w = input.new_ones((ch, 1, 1, 1))
        x = F.conv_transpose2d(input, w, stride=stride, groups=ch)
        pad = [(k - 1) * d - p for (k, d, p) in zip(ctx.kernel_size, ctx.dilation, ctx.padding)]
        x = F.pad(x, (pad[1], pad[1] + ctx.output_padding[1], pad[0], pad[0] + ctx.output_padding[0]))

        cols = F.unfold(x, ctx.kernel_size, ctx.dilation, _pair(0), _pair(1))

        in_mul_k = cols.view(bs, ch, *kernel.shape[2:]) * kernel

        # matrix multiplication, written as an einsum to avoid repeated view() and permute()
        if shared_filters:
            output = torch.einsum('ijklmn,jokl->iomn', (in_mul_k, weight))
        else:
            output = torch.einsum('ijklmn,jokl->iomn', (in_mul_k, weight))

        if bias is not None:
            output += bias.view(1, -1, 1, 1)

        return output.clone()  # TODO understand why a .clone() is needed here

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_input = grad_kernel = grad_weight = grad_bias = None
        (bs, out_ch), out_sz = grad_output.shape[:2], grad_output.shape[2:]
        in_ch = ctx.in_ch
        pad = [(k - 1) * d - p for (k, d, p) in zip(ctx.kernel_size, ctx.dilation, ctx.padding)]
        pad = [(p, p + op) for (p, op) in zip(pad, ctx.output_padding)]

        input, kernel, weight = ctx.saved_tensors
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            if ctx.shared_filters:
                grad_in_mul_k = grad_output.view(bs, out_ch, 1, 1, out_sz[0], out_sz[1]) \
                                * weight.view(ctx.kernel_size[0], ctx.kernel_size[1], 1, 1)
            else:
                grad_in_mul_k = torch.einsum('iomn,jokl->ijklmn', (grad_output, weight))
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            w = input.new_ones((in_ch, 1, 1, 1))
            x = F.conv_transpose2d(input, w, stride=ctx.stride, groups=in_ch)
            x = F.pad(x, (pad[1][0], pad[1][1], pad[0][0], pad[0][1]))
            in_cols = F.unfold(x, ctx.kernel_size, ctx.dilation, _pair(0), _pair(1))
            in_cols = in_cols.view(bs, in_ch, ctx.kernel_size[0], ctx.kernel_size[1], out_sz[0], out_sz[1])
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.new()
            grad_im2col_output = grad_in_mul_k * kernel
            grad_im2col_output = grad_im2col_output.view(bs, -1, out_sz[0] * out_sz[1])
            im2col_input_sz = [o + (k - 1) * d for (o, k, d) in zip(out_sz, ctx.kernel_size, ctx.dilation)]
            ctx._backend.Im2Col_updateGradInput(ctx._backend.library_state,
                                                grad_im2col_output,
                                                grad_input,
                                                im2col_input_sz[0], im2col_input_sz[1],
                                                ctx.kernel_size[0], ctx.kernel_size[1],
                                                ctx.dilation[0], ctx.dilation[1],
                                                0, 0,
                                                1, 1)
            grad_input = grad_input[:, :, pad[0][0]:-pad[0][1]:ctx.stride[0], pad[1][0]:-pad[1][1]:ctx.stride[1]]
        if ctx.needs_input_grad[1]:
            grad_kernel = in_cols * grad_in_mul_k
            grad_kernel = grad_kernel.sum(dim=1, keepdim=True)
        if ctx.needs_input_grad[2]:
            in_mul_k = in_cols * kernel
            if ctx.shared_filters:
                grad_weight = torch.einsum('ijmn,ijklmn->kl', (grad_output, in_mul_k))
                grad_weight = grad_weight.view(1, 1, ctx.kernel_size[0], ctx.kernel_size[1]).contiguous()
            else:
                grad_weight = torch.einsum('iomn,ijklmn->jokl', (grad_output, in_mul_k))
        if ctx.needs_input_grad[3]:
            grad_bias = torch.einsum('iomn->o', (grad_output,))
        return grad_input, grad_kernel, grad_weight, grad_bias, None, None, None, None, None


class PacPool2dFn(Function):
    @staticmethod
    def forward(ctx, input, kernel, kernel_size, stride=1, padding=0, dilation=1):
        (bs, ch), in_sz = input.shape[:2], input.shape[2:]
        if kernel.size(1) > 1 and kernel.size(1) != ch:
            raise ValueError('Incompatible input and kernel sizes.')
        ctx.input_size = in_sz
        ctx.kernel_size = _pair(kernel_size)
        ctx.kernel_ch = kernel.size(1)
        ctx.dilation = _pair(dilation)
        ctx.padding = _pair(padding)
        ctx.stride = _pair(stride)
        ctx.save_for_backward(input if ctx.needs_input_grad[1] else None,
                              kernel if ctx.needs_input_grad[0] else None)
        ctx._backend = type2backend[input.type()]

        cols = F.unfold(input, ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)

        output = cols.view(bs, ch, *kernel.shape[2:]) * kernel
        output = torch.einsum('ijklmn->ijmn', (output,))

        return output.clone()  # TODO check whether a .clone() is needed here

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, kernel = ctx.saved_tensors
        grad_input = grad_kernel = None
        (bs, ch), out_sz = grad_output.shape[:2], grad_output.shape[2:]
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.new()
            grad_im2col_output = torch.einsum('ijmn,izklmn->ijklmn', (grad_output, kernel))
            grad_im2col_output = grad_im2col_output.view(bs, -1, out_sz[0] * out_sz[1])
            ctx._backend.Im2Col_updateGradInput(ctx._backend.library_state,
                                                grad_im2col_output,
                                                grad_input,
                                                ctx.input_size[0], ctx.input_size[1],
                                                ctx.kernel_size[0], ctx.kernel_size[1],
                                                ctx.dilation[0], ctx.dilation[1],
                                                ctx.padding[0], ctx.padding[1],
                                                ctx.stride[0], ctx.stride[1])
        if ctx.needs_input_grad[1]:
            cols = F.unfold(input, ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)
            cols = cols.view(bs, ch, ctx.kernel_size[0], ctx.kernel_size[1], out_sz[0], out_sz[1])
            grad_kernel = torch.einsum('ijmn,ijklmn->ijklmn', (grad_output, cols))
            if ctx.kernel_ch == 1:
                grad_kernel = grad_kernel.sum(dim=1, keepdim=True)

        return grad_input, grad_kernel, None, None, None, None


# 출력: unfold된 유사도 가우시안 커널(필터), 별도 마스크 사용 여부
def packernel2d(input, mask=None, kernel_size=0, stride=1, padding=0, output_padding=0, dilation=1,
                kernel_type='gaussian', smooth_kernel_type='none', smooth_kernel=None, inv_alpha=None, inv_lambda=None,
                channel_wise=False, normalize_kernel=False, transposed=False, native_impl=False):
    kernel_size = _pair(kernel_size)
    dilation = _pair(dilation)
    padding = _pair(padding)
    output_padding = _pair(output_padding)
    stride = _pair(stride)
    output_mask = False if mask is None else True
    norm = None

    if mask is not None and mask.dtype != input.dtype: # 데이터 타입 다를 시 캐스팅
        mask = torch.tensor(mask, dtype=input.dtype, device=input.device)

    if transposed:
        in_sz = tuple(int((o - op - 1 - (k - 1) * d + 2 * p) // s) + 1 for (o, k, s, p, op, d) in
                      zip(input.shape[-2:], kernel_size, stride, padding, output_padding, dilation))
    else:
        in_sz = input.shape[-2:] # spatial dimensions

    if mask is not None or normalize_kernel:
        mask_pattern = input.new_ones(1, 1, *in_sz)
        mask_pattern = nd2col(mask_pattern, kernel_size, stride=stride, padding=padding, output_padding=output_padding,
                              dilation=dilation, transposed=transposed)
        if mask is not None:
            mask = nd2col(mask, kernel_size, stride=stride, padding=padding, output_padding=output_padding,
                          dilation=dilation, transposed=transposed)
            if not normalize_kernel:
                norm = mask.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True) \
                       / mask_pattern.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
        else:
            mask = mask_pattern

    if transposed:
        stride = _pair(1)
        padding = tuple((k - 1) * d // 2 for (k, d) in zip(kernel_size, dilation))

    if native_impl: # native_impl와 else의 코드 내용 차이는 그냥 클래스를 썼느냐 아니냐의 차이인 듯? 사실상 else의 GaussKernel2dFn만 보면 됨. # 차이: 속도가 다르다.
        bs, k_ch, in_h, in_w = input.shape

        x = nd2col(input, kernel_size, stride=stride, padding=padding, dilation=dilation)
        x = x.view(bs, k_ch, -1, *x.shape[-2:]).contiguous()

        if smooth_kernel_type == 'none':
            self_idx = kernel_size[0] * kernel_size[1] // 2
            feat_0 = x[:, :, self_idx:self_idx + 1, :, :]
        else:
            smooth_kernel_size = smooth_kernel.shape[2:]
            smooth_padding = (int(padding[0] - (kernel_size[0] - smooth_kernel_size[0]) / 2),
                              int(padding[1] - (kernel_size[1] - smooth_kernel_size[1]) / 2))
            crop = tuple(-1 * np.minimum(0, smooth_padding))
            input_for_kernel_crop = input.view(-1, 1, in_h, in_w)[:, :,
                                    crop[0]:_neg_idx(crop[0]), crop[1]:_neg_idx(crop[1])]
            smoothed = F.conv2d(input_for_kernel_crop, smooth_kernel,
                                stride=stride, padding=tuple(np.maximum(0, smooth_padding)))
            feat_0 = smoothed.view(bs, k_ch, 1, *x.shape[-2:])
        x = x - feat_0 # 차분 계산
        if kernel_type.find('_asym') >= 0:
            x = F.relu(x, inplace=True)
        # x.pow_(2)  # this causes an autograd issue in pytorch>0.4
        x = x * x # 제곱
        if not channel_wise:
            x = torch.sum(x, dim=1, keepdim=True) # 거리로 계산, 브로드캐스팅.
        if kernel_type == 'gaussian':
            x = torch.exp_(x.mul_(-0.5))  # TODO profiling for identifying the culprit of 5x slow down # 가우시안 적용. TODO 가중치와 편향을 적용한다면?
            # x = torch.exp(-0.5 * x)
        elif kernel_type.startswith('inv_'):
            epsilon = 1e-4
            x = inv_alpha.view(1, -1, 1, 1, 1) \
                + torch.pow(x + epsilon, 0.5 * inv_lambda.view(1, -1, 1, 1, 1))
        else:
            raise ValueError()
        output = x.view(*(x.shape[:2] + tuple(kernel_size) + x.shape[-2:])).contiguous()
    else:
        assert (smooth_kernel_type == 'none' and
                kernel_type == 'gaussian')
        output = GaussKernel2dFn.apply(input, kernel_size, stride, padding, dilation, channel_wise) # unfold된 유사도 가우시안 커널

    if mask is not None:
        output = output * mask  # avoid numerical issue on masked positions

    if normalize_kernel:
        norm = output.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)

    if norm is not None:
        empty_mask = (norm == 0)
        output = output / (norm + torch.tensor(empty_mask, dtype=input.dtype, device=input.device))
        output_mask = (1 - empty_mask) if output_mask else None
    else:
        output_mask = None

    return output, output_mask


def pacconv2d(input, kernel, weight, bias=None, stride=1, padding=0, dilation=1, shared_filters=False,
              native_impl=False):
    kernel_size = tuple(weight.shape[-2:])
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    if native_impl:
        # im2col on input
        im_cols = nd2col(input, kernel_size, stride=stride, padding=padding, dilation=dilation)

        # main computation
        if shared_filters:
            output = torch.einsum('ijklmn,zykl->ijmn', (im_cols * kernel, weight))
        else:
            output = torch.einsum('ijklmn,ojkl->iomn', (im_cols * kernel, weight))

        if bias is not None:
            output += bias.view(1, -1, 1, 1)
    else:
        output = PacConv2dFn.apply(input, kernel, weight, bias, stride, padding, dilation, shared_filters)

    return output

# GU
def pacconv2d_my(input, kernel, weight, bias=None, stride=1, padding=0, dilation=1, depth=None):
    kernel_size = tuple(weight.shape[-2:])
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    # im2col on input
    im_cols = nd2col(input, kernel_size, stride=stride, padding=padding, dilation=dilation)

    in_mul_k = im_cols * kernel # (N, C(in), kernel_h, kernel_w, out_h, out_w)
    dominate = torch.einsum('ijklmn->ijmn', [in_mul_k]) / torch.einsum('ijklmn->ijmn', [kernel])
    corrected = in_mul_k + dominate.unsqueeze(2).unsqueeze(2) * (1 - kernel)

    # TODO if depth is not None:

    output = torch.einsum('ijklmn,ojkl->iomn', (corrected, weight))
    # (N, C(in), kernel_h, kernel_w, out_h, out_w), (C(out), C(in), kernel_h, kernel_w) => (N, C(out), out_h, out_w)

    if bias is not None:
        output += bias.view(1, -1, 1, 1)

    return output


def pacconv_transpose2d(input, kernel, weight, bias=None, stride=1, padding=0, output_padding=0, dilation=1,
                        shared_filters=False, native_impl=False):
    kernel_size = tuple(weight.shape[-2:])
    stride = _pair(stride)
    padding = _pair(padding)
    output_padding = _pair(output_padding)
    dilation = _pair(dilation)

    if native_impl:
        ch = input.shape[1]
        w = input.new_ones((ch, 1, 1, 1))
        x = F.conv_transpose2d(input, w, stride=stride, groups=ch)
        pad = [(kernel_size[i] - 1) * dilation[i] - padding[i] for i in range(2)]
        x = F.pad(x, (pad[1], pad[1] + output_padding[1], pad[0], pad[0] + output_padding[0]))
        output = pacconv2d(x, kernel, weight.permute(1, 0, 2, 3), bias, dilation=dilation,
                           shared_filters=shared_filters, native_impl=True)
    else:
        output = PacConvTranspose2dFn.apply(input, kernel, weight, bias, stride, padding, output_padding, dilation,
                                            shared_filters)

    return output


def pacpool2d(input, kernel, kernel_size, stride=1, padding=0, dilation=1, native_impl=False):
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    if native_impl:
        bs, in_ch, in_h, in_w = input.shape
        out_h = (in_h + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
        out_w = (in_w + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1

        # im2col on input
        im_cols = nd2col(input, kernel_size, stride=stride, padding=padding, dilation=dilation)

        # main computation
        im_cols *= kernel
        output = im_cols.view(bs, in_ch, -1, out_h, out_w).sum(dim=2, keepdim=False)
    else:
        output = PacPool2dFn.apply(input, kernel, kernel_size, stride, padding, dilation)

    return output


class _PacConvNd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, bias,
                 pool_only, kernel_type, smooth_kernel_type,
                 channel_wise, normalize_kernel, shared_filters, filler):
        super(_PacConvNd, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.pool_only = pool_only
        self.kernel_type = kernel_type
        self.smooth_kernel_type = smooth_kernel_type
        self.channel_wise = channel_wise
        self.normalize_kernel = normalize_kernel
        self.shared_filters = shared_filters
        self.filler = filler
        if any([k % 2 != 1 for k in kernel_size]):
            raise ValueError('kernel_size only accept odd numbers')
        if smooth_kernel_type.find('_') >= 0 and int(smooth_kernel_type[smooth_kernel_type.rfind('_') + 1:]) % 2 != 1:
            raise ValueError('smooth_kernel_type only accept kernels of odd widths')
        if shared_filters:
            assert in_channels == out_channels, 'when specifying shared_filters, number of channels should not change'
        if any([p > d * (k - 1) / 2 for (p, d, k) in zip(padding, dilation, kernel_size)]):
            # raise ValueError('padding ({}) too large'.format(padding))
            pass  # TODO verify that this indeed won't cause issues
        if not pool_only:
            if self.filler in {'pool', 'crf_pool'}:
                assert shared_filters
                self.register_buffer('weight', torch.ones(1, 1, *kernel_size))
                if self.filler == 'crf_pool':
                    self.weight[(0, 0) + tuple(k // 2 for k in kernel_size)] = 0  # Eq.5, DenseCRF
            elif shared_filters:
                self.weight = Parameter(torch.Tensor(1, 1, *kernel_size))
            elif transposed:
                self.weight = Parameter(torch.Tensor(in_channels, out_channels, *kernel_size))
            else:
                self.weight = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
            if bias:
                self.bias = Parameter(torch.Tensor(out_channels))
            else:
                self.register_parameter('bias', None)
        if kernel_type.startswith('inv_'):
            self.inv_alpha_init = float(kernel_type.split('_')[1])
            self.inv_lambda_init = float(kernel_type.split('_')[2])
            if self.channel_wise and kernel_type.find('_fixed') < 0:
                if out_channels <= 0:
                    raise ValueError('out_channels needed for channel_wise {}'.format(kernel_type))
                inv_alpha = self.inv_alpha_init * torch.ones(out_channels)
                inv_lambda = self.inv_lambda_init * torch.ones(out_channels)
            else:
                inv_alpha = torch.tensor(float(self.inv_alpha_init))
                inv_lambda = torch.tensor(float(self.inv_lambda_init))
            if kernel_type.find('_fixed') < 0:
                self.register_parameter('inv_alpha', Parameter(inv_alpha))
                self.register_parameter('inv_lambda', Parameter(inv_lambda))
            else:
                self.register_buffer('inv_alpha', inv_alpha)
                self.register_buffer('inv_lambda', inv_lambda)
        elif kernel_type != 'gaussian':
            raise ValueError('kernel_type set to invalid value ({})'.format(kernel_type))
        if smooth_kernel_type.startswith('full_'):
            smooth_kernel_size = int(smooth_kernel_type.split('_')[-1])
            self.smooth_kernel = Parameter(torch.Tensor(1, 1, *repeat(smooth_kernel_size, len(kernel_size))))
        elif smooth_kernel_type == 'gaussian':
            smooth_1d = torch.tensor([.25, .5, .25])
            smooth_kernel = smooth_1d
            for d in range(1, len(kernel_size)):
                smooth_kernel = smooth_kernel * smooth_1d.view(-1, *repeat(1, d))
            self.register_buffer('smooth_kernel', smooth_kernel.unsqueeze(0).unsqueeze(0))
        elif smooth_kernel_type.startswith('average_'):
            smooth_kernel_size = int(smooth_kernel_type.split('_')[-1])
            smooth_1d = torch.tensor((1.0 / smooth_kernel_size,) * smooth_kernel_size)
            smooth_kernel = smooth_1d
            for d in range(1, len(kernel_size)):
                smooth_kernel = smooth_kernel * smooth_1d.view(-1, *repeat(1, d))
            self.register_buffer('smooth_kernel', smooth_kernel.unsqueeze(0).unsqueeze(0))
        elif smooth_kernel_type != 'none':
            raise ValueError('smooth_kernel_type set to invalid value ({})'.format(smooth_kernel_type))

        self.reset_parameters()

    def reset_parameters(self):
        if not (self.pool_only or self.filler in {'pool', 'crf_pool'}):
            if self.filler == 'uniform':
                n = self.in_channels
                for k in self.kernel_size:
                    n *= k
                stdv = 1. / math.sqrt(n)
                if self.shared_filters:
                    stdv *= self.in_channels
                self.weight.data.uniform_(-stdv, stdv)
                if self.bias is not None:
                    self.bias.data.uniform_(-stdv, stdv)
            elif self.filler == 'linear':
                effective_kernel_size = tuple(2 * s - 1 for s in self.stride)
                pad = tuple(int((k - ek) // 2) for k, ek in zip(self.kernel_size, effective_kernel_size))
                assert self.transposed and self.in_channels == self.out_channels
                assert all(k >= ek for k, ek in zip(self.kernel_size, effective_kernel_size))
                w = 1.0
                for i, (p, s, k) in enumerate(zip(pad, self.stride, self.kernel_size)):
                    d = len(pad) - i - 1
                    w = w * (np.array((0.0,) * p + tuple(range(1, s)) + tuple(range(s, 0, -1)) + (0,) * p) / s).reshape(
                        (-1,) + (1,) * d)
                    if self.normalize_kernel:
                        w = w * np.array(tuple(((k - j - 1) // s) + (j // s) + 1.0 for j in range(k))).reshape(
                            (-1,) + (1,) * d)
                self.weight.data.fill_(0.0)
                for c in range(1 if self.shared_filters else self.in_channels):
                    self.weight.data[c, c, :] = torch.tensor(w)
                if self.bias is not None:
                    self.bias.data.fill_(0.0)
            elif self.filler in {'crf', 'crf_perturbed'}:
                assert len(self.kernel_size) == 2 and self.kernel_size[0] == self.kernel_size[1] \
                       and self.in_channels == self.out_channels
                perturb_range = 0.001
                n_classes = self.in_channels
                gauss = np_gaussian_2d(self.kernel_size[0]) * self.kernel_size[0] * self.kernel_size[0]
                gauss[self.kernel_size[0] // 2, self.kernel_size[1] // 2] = 0
                if self.shared_filters:
                    self.weight.data[0, 0, :] = torch.tensor(gauss)
                else:
                    compat = 1.0 - np.eye(n_classes, dtype=np.float32)
                    self.weight.data[:] = torch.tensor(compat.reshape(n_classes, n_classes, 1, 1) * gauss)
                if self.filler == 'crf_perturbed':
                    self.weight.data.add_((torch.rand_like(self.weight.data) - 0.5) * perturb_range)
                if self.bias is not None:
                    self.bias.data.fill_(0.0)
            else:
                raise ValueError('Initialization method ({}) not supported.'.format(self.filler))
        if hasattr(self, 'inv_alpha') and isinstance(self.inv_alpha, Parameter):
            self.inv_alpha.data.fill_(self.inv_alpha_init)
            self.inv_lambda.data.fill_(self.inv_lambda_init)
        if hasattr(self, 'smooth_kernel') and isinstance(self.smooth_kernel, Parameter):
            self.smooth_kernel.data.fill_(1.0 / np.multiply.reduce(self.smooth_kernel.shape))

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', kernel_type={kernel_type}')
        if self.stride != (1,) * len(self.stride):
            s += ', stride={stride}'
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.bias is None:
            s += ', bias=False'
        if self.smooth_kernel_type != 'none':
            s += ', smooth_kernel_type={smooth_kernel_type}'
        if self.channel_wise:
            s += ', channel_wise=True'
        if self.normalize_kernel:
            s += ', normalize_kernel=True'
        if self.shared_filters:
            s += ', shared_filters=True'
        return s.format(**self.__dict__)


class PacConv2d(_PacConvNd):
    r"""
    Args (in addition to those of Conv2d):
        kernel_type (str): 'gaussian' | 'inv_{alpha}_{lambda}[_asym][_fixed]'. Default: 'gaussian'
        smooth_kernel_type (str): 'none' | 'gaussian' | 'average_{sz}' | 'full_{sz}'. Default: 'none'
        normalize_kernel (bool): Default: False
        shared_filters (bool): Default: False
        filler (str): 'uniform'. Default: 'uniform'

    Note:
        - kernel_size only accepts odd numbers
        - padding should not be larger than :math:`dilation * (kernel_size - 1) / 2`
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, compute_kernel=False,
                 kernel_type='gaussian', smooth_kernel_type='none', normalize_kernel=False, shared_filters=False,
                 filler='uniform', native_impl=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(PacConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, False, _pair(0), bias,
            False, kernel_type, smooth_kernel_type, False, normalize_kernel, shared_filters, filler)

        self.native_impl = native_impl

        if compute_kernel is True:
            self.compute_kernel = EdgeKernel2d(in_channels=1)

    def forward(self, input_2d, input_for_kernel=None, kernel=None, mask=None, return_kernel=False):
        output_mask = None
        if kernel is None:
            kernel = self.compute_kernel(input_for_kernel)

        # GU
        output = pacconv2d_my(input_2d, kernel, self.weight, self.bias, self.stride, self.padding, self.dilation)

        return (output, kernel) if return_kernel is True else output


class PacConvTranspose2d(_PacConvNd):
    r"""
    Args (in addition to those of ConvTranspose2d):
        kernel_type (str): 'gaussian' | 'inv_{alpha}_{lambda}[_asym][_fixed]'. Default: 'gaussian'
        smooth_kernel_type (str): 'none' | 'gaussian' | 'average_{sz}' | 'full_{sz}'. Default: 'none'
        normalize_kernel (bool): Default: False
        shared_filters (bool): Default: False
        filler (str): 'uniform' | 'linear'. Default: 'uniform'

    Note:
        - kernel_size only accepts odd numbers
        - padding should not be larger than :math:`dilation * (kernel_size - 1) / 2`
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1,
                 bias=True, kernel_type='gaussian', smooth_kernel_type='none', normalize_kernel=False,
                 shared_filters=False, filler='uniform', native_impl=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        output_padding = _pair(output_padding)
        dilation = _pair(dilation)
        super(PacConvTranspose2d, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, True, output_padding, bias,
            False, kernel_type, smooth_kernel_type, False, normalize_kernel, shared_filters, filler)

        self.native_impl = native_impl

    def compute_kernel(self, input_for_kernel, input_mask=None):
        return packernel2d(input_for_kernel, input_mask,
                           kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                           output_padding=self.output_padding, dilation=self.dilation, kernel_type=self.kernel_type,
                           smooth_kernel_type=self.smooth_kernel_type,
                           smooth_kernel=self.smooth_kernel if hasattr(self, 'smooth_kernel') else None,
                           inv_alpha=self.inv_alpha if hasattr(self, 'inv_alpha') else None,
                           inv_lambda=self.inv_lambda if hasattr(self, 'inv_lambda') else None,
                           channel_wise=False, normalize_kernel=self.normalize_kernel, transposed=True,
                           native_impl=self.native_impl)

    def forward(self, input_2d, input_for_kernel=None, kernel=None, mask=None):
        output_mask = None
        if kernel is None:
            kernel, output_mask = self.compute_kernel(input_for_kernel, mask)

        output = pacconv_transpose2d(input_2d, kernel, self.weight, self.bias, self.stride, self.padding,
                                     self.output_padding, self.dilation, self.shared_filters, self.native_impl)

        return output if output_mask is None else (output, output_mask)


class PacPool2d(_PacConvNd):
    r"""
    Args:
        kernel_size, stride, padding, dilation
        kernel_type (str): 'gaussian' | 'inv_{alpha}_{lambda}[_asym][_fixed]'. Default: 'gaussian'
        smooth_kernel_type (str): 'none' | 'gaussian' | 'average_{sz}' | 'full_{sz}'. Default: 'none'
        channel_wise (bool): Default: False
        normalize_kernel (bool): Default: False
        out_channels (int): needs to be specified for channel_wise 'inv_*' (non-fixed) kernels. Default: -1

    Note:
        - kernel_size only accepts odd numbers
        - padding should not be larger than :math:`dilation * (kernel_size - 1) / 2`
    """

    def __init__(self, kernel_size, stride=1, padding=0, dilation=1,
                 kernel_type='gaussian', smooth_kernel_type='none',
                 channel_wise=False, normalize_kernel=False, out_channels=-1, native_impl=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(PacPool2d, self).__init__(
            -1, out_channels, kernel_size, stride,
            padding, dilation, False, _pair(0), False,
            True, kernel_type, smooth_kernel_type, channel_wise, normalize_kernel, False, None)

        self.native_impl = native_impl

    def compute_kernel(self, input_for_kernel, input_mask=None):
        return packernel2d(input_for_kernel, input_mask,
                           kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                           dilation=self.dilation, kernel_type=self.kernel_type,
                           smooth_kernel_type=self.smooth_kernel_type,
                           smooth_kernel=self.smooth_kernel if hasattr(self, 'smooth_kernel') else None,
                           inv_alpha=self.inv_alpha if hasattr(self, 'inv_alpha') else None,
                           inv_lambda=self.inv_lambda if hasattr(self, 'inv_lambda') else None,
                           channel_wise=self.channel_wise, normalize_kernel=self.normalize_kernel, transposed=False,
                           native_impl=self.native_impl)

    def forward(self, input_2d, input_for_kernel, kernel=None, mask=None):
        output_mask = None
        if kernel is None:
            kernel, output_mask = self.compute_kernel(input_for_kernel, mask)

        bs, in_ch, in_h, in_w = input_2d.shape
        if self.channel_wise and (kernel.shape[1] != in_ch):
            raise ValueError('input and kernel must have the same number of channels when channel_wise=True')
        assert self.out_channels <= 0 or self.out_channels == in_ch

        output = pacpool2d(input_2d, kernel, self.kernel_size, self.stride, self.padding, self.dilation,
                           self.native_impl)

        return output if output_mask is None else (output, output_mask)
