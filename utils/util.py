import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function, once_differentiable
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
from torch._thnn import type2backend

from models.myPac import GaussKernel2dFn, pacconv2d

class HingeLoss(nn.Module):

    def __init__(self, margin, reduction='mean', ignore_pad=True, pad_value=0):
        super(HingeLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
        self.ignore_pad = ignore_pad
        self.pad_value = pad_value

    # labels: 0, 1
    def forward(self, inputs, labels, mask=None):

        assert inputs.shape == labels.shape

        targets = torch.sum(inputs * labels, dim=1, keepdim=True) # (N, 1, H, W)
        loss = inputs - targets + self.margin * (1 - labels) # (N, D, H, W)
        loss = F.relu(loss)

        if mask is not None:
            loss *= mask.float()
        if self.ignore_pad:
            pad_mask = targets != self.pad_value # 패딩된 곳이 true disp일때는 loss 계산하지 않음.
            # print(torch.sum(targets == self.pad_value))
            loss *= pad_mask.float()

        if self.reduction == 'none':
            return loss
        if self.reduction == 'mean':
            ones = torch.ones_like(inputs, dtype=torch.int, device=inputs.device)
            if mask is not None:
                ones *= mask.int()
            if self.ignore_pad:
                ones *= (targets != self.pad_value).int()
            return torch.sum(loss) / (torch.sum(ones).float() + 1e-12)
        if self.reduction == 'sum':
            return torch.sum(loss)

def one_hot(indices, depth, on_value=1, off_value=0, dim=1, dtype=None):
    assert indices.shape[dim] == 1
    eye = torch.eye(depth, dtype=dtype, device=indices.device)
    if on_value != 1 or off_value != 0:
        eye = eye * on_value + (1-eye) * off_value
    if dim == -1:
        out = eye[indices.long()]
    else:
        out = torch.squeeze(eye[indices.long()].transpose(dim,-1), -1)
    return out

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

class BilateralFilter(nn.Module):

    def __init__(self, kernel_size, stride):
        super(BilateralFilter, self).__init__()
        assert kernel_size % 2 == 1
        self.kernel_size = kernel_size
        self.stride = stride

        self.sigma_for_kernel = Parameter(torch.tensor(float(self.kernel_size) / 4))
        self.sigma_for_guide = 1 # Parameter(torch.tensor(1.))

        self.register_buffer('r', torch.tensor(np.arange(-(self.kernel_size // 2), (self.kernel_size // 2) + 1, dtype=np.float32)))
    
    def gaussian_2d_kernel(self, sigma):
        '''Truncated 2D Gaussian filter'''
        gaussian_1d = torch.exp(-0.5 * self.r * self.r / (sigma * sigma))
        gaussian_2d = gaussian_1d.reshape(-1, 1) * gaussian_1d
        gaussian_2d /= gaussian_2d.sum()

        return gaussian_2d # (kernel_size, kernel_size)

    def compute_guide(self, input, sigma):
        input = input / sigma
        # 파라미터: input, kernel_size, stride, padding, dilation, channel_wise
        output = GaussKernel2dFn.apply(input, self.kernel_size, self.stride, self.kernel_size//2, 1, False) # 패딩되어도 바이리터럴 특성상 영향 거의 없을 듯?
        return output # (N, C=1, kernel_size, kernel_size, out_h, out_w)

    def forward(self, input, input_for_kernel=None):
        # print("sigma for kernel:", float(self.sigma_for_kernel), "sigma for guide:", float(self.sigma_for_guide))
        kernel = self.gaussian_2d_kernel(self.sigma_for_kernel).view(1,1,self.kernel_size,self.kernel_size,1,1)
        if input_for_kernel is None:
            guide = self.compute_guide(input, self.sigma_for_guide)
        else:
            guide = self.compute_guide(input_for_kernel, self.sigma_for_guide)
        ker_mul_gd = kernel * guide
        norm = ker_mul_gd.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
        empty_mask = (norm == 0)
        ker_mul_gd = ker_mul_gd / (norm + torch.tensor(empty_mask, dtype=input.dtype, device=input.device))

        # 파라미터: input, kernel, stride, padding, dilation
        output = ApplyKernelFn.apply(input, ker_mul_gd, self.stride, self.kernel_size//2, 1)

        return output

class AdaptiveBilateralFilter(nn.Module):

    def __init__(self, kernel_size, stride):
        super(AdaptiveBilateralFilter, self).__init__()
        assert kernel_size % 2 == 1
        self.kernel_size = kernel_size
        self.stride = stride

        self.register_buffer('r', torch.tensor(np.arange(-(self.kernel_size // 2), (self.kernel_size // 2) + 1, dtype=np.float32)))
    
    def gaussian_2d_kernel(self, sigma):
        '''Truncated 2D Gaussian filter'''
        _1d = -0.5 * self.r * self.r
        _2d = (_1d.reshape(-1, 1) + _1d.reshape(1,-1)).view(1, 1, self.kernel_size, self.kernel_size, 1, 1)
        _2d = _2d * (sigma*sigma).unsqueeze(2).unsqueeze(3)
        gaussian_2d = torch.exp(_2d).clone() # TODO 이런 ㅆ발
        gaussian_2d /= gaussian_2d.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)

        return gaussian_2d # (N, C=1, kernel_size, kernel_size, out_h, out_w)

    def compute_guide(self, input, sigma, stride):
        # 파라미터: input, kernel_size, stride, padding, dilation, channel_wise
        output = AdaptiveGaussKernel2dFn.apply(input, sigma, self.kernel_size, stride, self.kernel_size//2, 1, False) # 패딩되어도 바이리터럴 특성상 영향 거의 없을 듯?
        return output # (N, C=1, kernel_size, kernel_size, out_h, out_w)

    def forward(self, input, sigmas, stride=None):
        if stride is None:
            stride = self.stride
        assert sigmas.shape[1] == 2
        kernel = self.gaussian_2d_kernel(1 / (torch.abs(sigmas[:,0:1,::stride,::stride]) + 1e-12)) # prevent division by zero
        guide = self.compute_guide(input, 1 / (torch.abs(sigmas[:,1:2,::stride,::stride]) + 1e-12), stride)
        ker_mul_gd = kernel * guide
        norm = ker_mul_gd.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
        empty_mask = (norm == 0)
        ker_mul_gd = ker_mul_gd / (norm + torch.tensor(empty_mask, dtype=input.dtype, device=input.device))

        # 파라미터: input, kernel, stride, padding, dilation
        output = ApplyKernelFn.apply(input, ker_mul_gd, stride, self.kernel_size//2, 1)

        return output

class ApplyKernelFn(Function):
    @staticmethod
    def forward(ctx, input, kernel, stride=1, padding=0, dilation=1):
        (bs, ch), in_sz = input.shape[:2], input.shape[2:]
        if kernel.size(1) > 1:
            raise ValueError('Non-singleton channel is not allowed for kernel.') # (N, C=1, kernel_h, kernel_w, out_h, out_w)
        ctx.input_size = in_sz
        ctx.in_ch = ch
        ctx.kernel_size = tuple(kernel.shape[2:4])
        ctx.dilation = _pair(dilation)
        ctx.padding = _pair(padding)
        ctx.stride = _pair(stride)
        ctx.save_for_backward(input if (ctx.needs_input_grad[1]) else None,
                              kernel if (ctx.needs_input_grad[0]) else None)
        ctx._backend = type2backend[input.type()]

        cols = F.unfold(input, ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)
        in_mul_k = cols.view(bs, ch, *kernel.shape[2:]) * kernel # 커널 적용
        # (N, C, kernel_h, kernel_w, out_h, out_w) * (N, C=1, kernel_h, kernel_w, out_h, out_w)

        # matrix multiplication, written as an einsum to avoid repeated view() and permute()
        output = torch.einsum('ijklmn->ijmn', [in_mul_k])
        # (N, C(in), kernel_h, kernel_w, out_h, out_w), (out, in, kernel_h, kernel_w) => (N, C(out), out_h, out_w)

        return output.clone()  # TODO understand why a .clone() is needed here

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_input = grad_kernel = None
        (bs, out_ch), out_sz = grad_output.shape[:2], grad_output.shape[2:]
        in_ch = ctx.in_ch

        input, kernel = ctx.saved_tensors
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            grad_in_mul_k = grad_output.unsqueeze(2).unsqueeze(3) # torch.einsum('iomn,ojkl->ijklmn', (grad_output, weight))
        if ctx.needs_input_grad[1]:
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

        return grad_input, grad_kernel, None, None, None

class AdaptiveGaussKernel2dFn(Function):
    @staticmethod
    def forward(ctx, input, sigma, kernel_size, stride, padding, dilation, channel_wise):
        assert padding == kernel_size // 2 and dilation == 1 and channel_wise is False
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
        output = torch.exp(-0.5 * diff_sq * sigma.pow(2).unsqueeze(2).unsqueeze(3)) # GU 차이가 작으면 1, 차이가 크면 0에 가까운 마스크
        ctx._backend = type2backend[input.type()]
        ctx.save_for_backward(input, output, sigma)

        return output # (N, C=1, kernel_h, kernel_w, out_h, out_w), unfold 상태. 이거를 unfold된 input과 곱하면 각 위치마다 위치에 적응된 마스크를 씌우는 효과를 냄.

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_input = grad_sigma = None
        input, output, sigma = ctx.saved_tensors
        bs, ch, in_h, in_w = input.shape
        out_h, out_w = output.shape[-2:]
        cols = F.unfold(input, ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)
        cols = cols.view(bs, ch, ctx.kernel_size[0], ctx.kernel_size[1], out_h, out_w)
        center_y, center_x = ctx.kernel_size[0] // 2, ctx.kernel_size[1] // 2
        feat_0 = cols.contiguous()[:, :, center_y:center_y + 1, center_x:center_x + 1, :, :]
        diff = cols - feat_0
        if ctx.needs_input_grad[0]:
            grad = -0.5 * grad_output * output * sigma.pow(2).unsqueeze(2).unsqueeze(3) # GU
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
        if ctx.needs_input_grad[1]:
            grad_sigma_sq = (-0.5 * grad_output * output * diff.pow(2).sum(dim=1, keepdim=True)).sum(dim=3).sum(dim=2) # GU (N, C=1, out_h, out_w)
            grad_sigma = grad_sigma_sq * (2 * sigma) # GU

        return grad_input, grad_sigma, None, None, None, None, None