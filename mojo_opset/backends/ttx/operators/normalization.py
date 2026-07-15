import os

import torch

from mojo_opset.backends.ttx.kernels import fused_add_layernorm_infer
from mojo_opset.backends.ttx.kernels import fused_add_layernorm_tle_infer_impl
from mojo_opset.backends.ttx.kernels import fused_add_rmsnorm_infer
from mojo_opset.backends.ttx.kernels import fused_add_rmsnorm_tle_infer_impl
from mojo_opset.backends.ttx.kernels import group_rmsnorm
from mojo_opset.backends.ttx.kernels import layernorm_infer
from mojo_opset.backends.ttx.kernels import layernorm_tle_infer_impl
from mojo_opset.backends.ttx.kernels import rmsnorm_infer
from mojo_opset.backends.ttx.kernels import rmsnorm_tle_infer_impl
from mojo_opset.core import MojoGroupRMSNorm
from mojo_opset.core import MojoLayerNorm
from mojo_opset.core import MojoResidualAddLayerNorm
from mojo_opset.core import MojoResidualAddRMSNorm
from mojo_opset.core import MojoRMSNorm
from mojo_opset.utils.platform import get_platform


def _norm_tle_enabled() -> bool:
    return os.getenv("MOJO_TTX_NORM_TLE", "1").lower() not in ("0", "false", "off", "no")


def _is_not_impl(fn) -> bool:
    try:
        fn.__name__
    except AttributeError:
        return False
    return fn.__name__ == "_not_impl"


def _can_use_norm_tle(*tensors: torch.Tensor) -> bool:
    if not _norm_tle_enabled() or get_platform() != "npu":
        return False
    for tensor in tensors:
        if tensor is None:
            return False
        if tensor.device.type != "npu":
            return False
    return True


class TTXGroupRMSNorm(MojoGroupRMSNorm):
    supported_platforms_list = ["mlu", "ilu"]

    def forward(self, input_groups) -> list[torch.tensor]:
        return group_rmsnorm(input_groups, weight=self.weight, eps=self.variance_epsilon)


class TTXLayerNorm(MojoLayerNorm):
    supported_platforms_list = ["npu", "ilu", "mlu"]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if _can_use_norm_tle(hidden_state, self.weight, self.bias) and not _is_not_impl(layernorm_tle_infer_impl):
            try:
                return layernorm_tle_infer_impl(hidden_state, self.weight, self.bias, self.variance_epsilon)
            except NotImplementedError:
                pass
        return layernorm_infer(hidden_state, self.weight, self.bias, self.variance_epsilon)


class TTXRMSNorm(MojoRMSNorm):
    supported_platforms_list = ["npu", "ilu"]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if _can_use_norm_tle(hidden_state, self.weight) and not _is_not_impl(rmsnorm_tle_infer_impl):
            try:
                return rmsnorm_tle_infer_impl(hidden_state, self.weight, self.variance_epsilon)
            except NotImplementedError:
                pass
        return rmsnorm_infer(hidden_state, self.weight, self.variance_epsilon)


class TTXResidualAddRMSNorm(MojoResidualAddRMSNorm):
    supported_platforms_list = ["npu", "ilu"]

    def forward(self, hidden_state: torch.Tensor, residual: torch.Tensor = None):
        if _can_use_norm_tle(hidden_state, residual, self.weight) and not _is_not_impl(fused_add_rmsnorm_tle_infer_impl):
            try:
                return fused_add_rmsnorm_tle_infer_impl(
                    hidden_state,
                    residual,
                    self.weight,
                    self.norm_pos,
                    self.variance_epsilon,
                )
            except NotImplementedError:
                pass

        output, res = fused_add_rmsnorm_infer(
            hidden_state,
            residual,
            self.weight,
            self.norm_pos,
            self.variance_epsilon,
        )

        return output, res


class TTXResidualAddLayerNorm(MojoResidualAddLayerNorm):
    supported_platforms_list = ["npu", "ilu"]

    def forward(self, hidden_state: torch.Tensor, residual: torch.Tensor = None):
        if _can_use_norm_tle(hidden_state, residual, self.weight, self.bias) and not _is_not_impl(
            fused_add_layernorm_tle_infer_impl
        ):
            try:
                return fused_add_layernorm_tle_infer_impl(
                    hidden_state,
                    residual,
                    self.weight,
                    self.bias,
                    self.norm_pos,
                    self.variance_epsilon,
                )
            except NotImplementedError:
                pass

        output, res = fused_add_layernorm_infer(
            hidden_state,
            residual,
            self.weight,
            self.bias,
            self.norm_pos,
            self.variance_epsilon,
        )

        return output, res
