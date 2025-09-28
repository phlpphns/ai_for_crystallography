import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import numpy as np
import math
import os
import random


# =================================================================================


# model_file = os.path.join(OV.BaseDir(), "PhAI_model.pth")
model_file = os.path.join(os.path.dirname(os.path.abspath( __file__ )), "PhAI_model.pth")
if not os.path.exists(model_file):
    # end the script
    raise FileNotFoundError(
        "PhAI model file not found. Please download the model from the PhAI repository and place it in the base directory to use it."
    )

# model definition
model_args = {
    "max_index": 10,
    "filters": 96,
    "kernel_size": 3,
    "cnn_depth": 6,
    "dim": 1024,
    "dim_exp": 2048,
    "dim_token_exp": 512,
    "mlp_depth": 8,
    "reflections": 1205,
}



# =================================================================================


class ConvolutionalBlock(nn.Module):
    def __init__(self, filters, kernel_size, padding):
        super().__init__()

        self.act = nn.GELU()

        self.conv1 = nn.Conv3d(
            filters, filters, kernel_size=kernel_size, padding=padding
        )
        self.conv2 = nn.Conv3d(
            filters, filters, kernel_size=kernel_size, padding=padding
        )

        self.norm1 = nn.GroupNorm(filters, filters)
        self.norm2 = nn.GroupNorm(filters, filters)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.act(x)
        x = self.norm1(x)

        x = self.conv2(x)
        x = self.act(x)
        x = self.norm2(x)

        x = x + identity
        return x


class MLPLayer(nn.Module):
    def __init__(self, token_nr, dim, dim_exp, mix_type):
        super().__init__()

        self.act = nn.GELU()

        self.norm1 = nn.GroupNorm(token_nr, token_nr)

        if mix_type == "token":
            self.layer1 = nn.Conv1d(
                kernel_size=1, in_channels=token_nr, out_channels=dim_exp
            )
            self.layer2 = nn.Conv1d(
                kernel_size=1, in_channels=dim_exp, out_channels=token_nr
            )
        else:
            self.layer1 = nn.Linear(dim, dim_exp)
            self.layer2 = nn.Linear(dim_exp, dim)

        self.mix_type = mix_type

    def forward(self, x):
        identity = x

        x = self.norm1(x)

        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)

        x = x + identity

        return x


class PhAINeuralNetwork(nn.Module):
    def __init__(
        self,
        *,
        max_index,
        filters,
        kernel_size,
        cnn_depth,
        dim,
        dim_exp,
        dim_token_exp,
        mlp_depth,
        reflections
    ):
        super().__init__()

        hkl = [max_index * 2 + 1, max_index + 1, max_index + 1]
        mlp_token_nr = filters
        padding = int((kernel_size - 1) / 2)

        self.net_a = nn.Sequential(
            Rearrange("b x y z  -> b 1 x y z "),
            nn.Conv3d(1, filters, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.GroupNorm(filters, filters),
        )

        self.net_p = nn.Sequential(
            Rearrange("b x y z  -> b 1 x y z "),
            nn.Conv3d(1, filters, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.GroupNorm(filters, filters),
        )

        self.net_convolution_layers = nn.Sequential(
            *[
                nn.Sequential(
                    ConvolutionalBlock(
                        filters, kernel_size=kernel_size, padding=padding
                    ),
                )
                for _ in range(cnn_depth)
            ],
        )

        self.net_projection_layer = nn.Sequential(
            Rearrange("b c x y z  -> b c (x y z)"),
            nn.Linear(hkl[0] * hkl[1] * hkl[2], dim),
        )

        self.net_mixer_layers = nn.Sequential(
            *[
                nn.Sequential(
                    MLPLayer(mlp_token_nr, dim, dim_token_exp, "token"),
                    MLPLayer(mlp_token_nr, dim, dim_exp, "channel"),
                )
                for _ in range(mlp_depth)
            ],
            nn.LayerNorm(dim),
        )

        self.net_output = nn.Sequential(
            Rearrange("b t x -> b x t"),
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b x 1 -> b x"),
            nn.Linear(dim, reflections * 2),
            Rearrange("b (c h) -> b c h ", h=reflections),
        )

    def forward(self, input_amplitudes, input_phases):
        a = self.net_a(input_amplitudes)
        p = self.net_p(input_phases)

        x = a + p

        x = self.net_convolution_layers(x)

        x = self.net_projection_layer(x)

        x = self.net_mixer_layers(x)

        phases = self.net_output(x)

        return phases


def randomize_output(output):
    shape = output[0].shape
    rand_mask = torch.randint(0, 2, shape)
    output[0][rand_mask == 1] = -180.0
    output[0][rand_mask == 0] = 0.0
    return output


def create_hkl_array(max_index):
    hkl_array = []
    for h in range(-max_index, max_index + 1):
        for k in range(0, max_index + 1):
            for l in range(0, max_index + 1):
                if not (h == 0 and k == 0 and l == 0):
                    if math.sqrt(h**2 + k**2 + l**2) <= max_index:
                        hkl_array.append([h, k, l])
    hkl_array = np.array(hkl_array, dtype=np.int32)
    return hkl_array


def _create_amplitudes(Fabs, H, max_index):
    amplitudes = torch.zeros(1, 21, 11, 11)
    for i in range(len(H)):
        amplitudes[0][H[i][0] + 10][H[i][1]][H[i][2]] = Fabs[i]
    return amplitudes


def create_amplitudes(Fs, indices, max_index):
    I_max = max(Fs)
    amplitudes = torch.zeros(1, 21, 11, 11)
    for f, hkl in zip(Fs, indices):
        h, k, l = hkl
        if (
            h < -max_index
            or h > max_index
            or k < 0
            or k > max_index
            or l < 0
            or l > max_index
        ):
            continue
        if not (h == 0 and k == 0 and l == 0):
            if f != 0:
                amplitudes[0][h + max_index][k][l] = f / I_max
    return amplitudes


def create_amplitudes_ord(amplitudes, max_index):
    amplitudes_ord = []
    for h in range(-max_index, max_index + 1):
        for k in range(0, max_index + 1):
            for l in range(0, max_index + 1):
                if not (h == 0 and k == 0 and l == 0):
                    if math.sqrt(h**2 + k**2 + l**2) <= max_index:
                        amplitudes_ord.append(amplitudes[0][h + 10][k][l])
    amplitudes_ord = np.array(amplitudes_ord)
    return amplitudes_ord


def phases(output_phases):
    bin_size = 180.0
    offset = bin_size / 2
    # bin_nr = int(360 / bin_size)
    output_phases = output_phases.permute(0, 2, 1)
    output_phases = torch.argmax(output_phases, dim=2)
    return offset + (output_phases * bin_size) - 180.00 - (bin_size / 2)


def copy_new_phases_to_initial(hkl_array, init_phases, output, max_index=10):
    for j in range(len(hkl_array)):
        init_phases[0][hkl_array[j][0] + max_index][hkl_array[j][1]][
            hkl_array[j][2]
        ] = output[0][j]
    return init_phases


def reindex_monoclinic(H):  # same as in crystallography_module!
    # to: (-h, h), (0, k), (0, l)
    # bug: -2, -1, 0 -> -2, 1, 0 (incorrect) corrected?
    H_new = np.array([[0, 0, 0]], dtype=int)
    symm_eq = [(-1, 1, -1), (1, -1, 1), (-1, -1, -1)]
    for h in H:
        if h[1] < 0 or h[2] < 0:
            for eq in symm_eq:
                h_new = (h[0] * eq[0], h[1] * eq[1], h[2] * eq[2])
                if h_new[1] >= 0 and h_new[2] >= 0:
                    H_new = np.append(
                        H_new,
                        np.array([[h[0] * eq[0], h[1] * eq[1], h[2] * eq[2]]]),
                        axis=0,
                    )
                    break
        else:
            H_new = np.append(H_new, np.array([[h[0], h[1], h[2]]]), axis=0)
    H_new = np.delete(H_new, 0, axis=0)
    for i in range(len(H_new)):
        if H_new[i][2] == 0 and H_new[i][0] < 0:  # locus layer = hk0 and not -hk0
            H_new[i][0] = -H_new[i][0]

    return H_new


def prepare_reflections(H, F):  # same as in crystallography_module!
    for i in range(len(F)):
        F[i] = F[i] / np.max(F)
    H_reind = reindex_monoclinic(H)
    sort_array = np.lexsort((H_reind[:, 2], H_reind[:, 1], H_reind[:, 0]))
    H_reind = H_reind[sort_array]
    F = F[sort_array]

    H_final = np.array([[0, 0, 0]])
    F_final = np.array([])
    group = [F[0]]
    H_curr = H_reind[0]

    for i in range(len(H_reind)):
        if (H_reind[i] == H_curr).all():
            group.append(F[i])
        else:
            H_final = np.append(H_final, np.array([H_curr]), axis=0)
            F_final = np.append(F_final, sum(group) / len(group))
            H_curr = H_reind[i]
            group = [F[i]]
    H_final = np.append(H_final, np.array([H_curr]), axis=0)
    F_final = np.append(F_final, sum(group) / len(group))
    H_final = np.delete(H_final, 0, axis=0)

    max_F_final = max(F_final)
    F_final = F_final / max_F_final

    return H_final, F_final


# def ___output_files(amplitudes_ord, output, fname, fname_ext):
#     # output
#     file_out = open(fname, "w")
#     file_out_ext = open(fname_ext, "w")
#     for n in range(0, len(hkl_array)):
#         # remove locus artefact
#         if hkl_array[n][2] == 0 and hkl_array[n][0] < 0:
#             continue
#         # write
#         if amplitudes_ord[n] != 0.0:
#             if output[0][n] == 0:
#                 F = complex(amplitudes_ord[n], 0)
#             elif output[0][n] == -180:
#                 F = complex(-amplitudes_ord[n], 0)
#             else:
#                 print("Wrong phase!?", output[0][n])
#                 input()
#             file_out.write("{} {} {} {}\n".format(*hkl_array[n], F))
#         else:
#             # write extended phases
#             if output[0][n] == 0:
#                 F = complex(amplitudes_ord[n], 0)
#             elif output[0][n] == -180:
#                 F = complex(-amplitudes_ord[n], 0)
#             else:
#                 print("Wrong phase!?", output[0][n])
#                 input()
#             file_out_ext.write("{} {} {} {}\n".format(*hkl_array[n], F))
#     file_out.close()
#     file_out_ext.close()


def output_files(amplitudes_ord, output, fname, fname_ext, hkl_array):
    # output
    file_out = open(fname, "w")
    file_out_ext = open(fname_ext, "w")
    for n in range(0, len(hkl_array)):
        # remove locus artefact
        if hkl_array[n][2] == 0 and hkl_array[n][0] < 0:
            continue
        # write
        if amplitudes_ord[n] != 0.0:
            if output[0][n] == 0:
                F = complex(amplitudes_ord[n], 0)
            elif output[0][n] == -180:
                F = complex(-amplitudes_ord[n], 0)
            else:
                print("Wrong phase!?", output[0][n])
                input()
            file_out.write("{} {} {} {}\n".format(*hkl_array[n], F))
        else:
            # write extended phases
            if output[0][n] == 0:
                F = complex(amplitudes_ord[n], 0)
            elif output[0][n] == -180:
                F = complex(-amplitudes_ord[n], 0)
            else:
                print("Wrong phase!?", output[0][n])
                input()
            file_out_ext.write("{} {} {} {}\n".format(*hkl_array[n], F))
    file_out.close()
    file_out_ext.close()


# =================================================================================

model = PhAINeuralNetwork(**model_args)
state = torch.load(model_file, weights_only=True)
model.load_state_dict(state)

def get_PhAI_phases_phil(f_sq_obs, randomize_phases=0, cycles=1, t=False, name_infile="", INPUT_IS_SQUARED=True):
    Fs = f_sq_obs[0]
    indices = f_sq_obs[1]

    if INPUT_IS_SQUARED:
        Fs = np.sqrt(np.abs(Fs))
    max_index = model_args["max_index"]
    hkl_array = create_hkl_array(max_index)

    # H, Fabs = crystallography_module.merge_reflections(H_tmp, Fabs_tmp)
    H, Fabs = prepare_reflections(indices, Fs)

    amplitudes = create_amplitudes(Fabs, H, max_index)
    amplitudes_ord = create_amplitudes_ord(amplitudes, max_index)

    # if p == 0:
    #     init_phases = torch.zeros(1, 21, 11, 11)
    # else:
    #     init_phases = randomize_output(torch.zeros(1, 21, 11, 11))

    if randomize_phases:
        init_phases = randomize_output(torch.zeros_like(amplitudes))
    else:
        init_phases = torch.zeros_like(amplitudes)

    for i in range(cycles):
        print("cycle: ", i + 1)
        if i == 0:
            output = phases(model(amplitudes, init_phases))
            if t == True and cycles != 1:
                output_files(
                    amplitudes_ord,
                    output,
                    name_infile[: len(name_infile) - 4] + "_" + str(i + 1) + ".F",
                    name_infile[: len(name_infile) - 4] + "_phase_extension_" + str(i + 1) + ".F",
                    hkl_array,
                )
        else:
            init_phases = copy_new_phases_to_initial(
                hkl_array, init_phases, output, max_index
            )
            output = phases(model(amplitudes, init_phases))
            if t == True and i + 1 != cycles:
                output_files(
                    amplitudes_ord,
                    output,
                    name_infile[: len(name_infile) - 4] + "_" + str(i + 1) + ".F",
                    name_infile[: len(name_infile) - 4] + "_phase_extension_" + str(i + 1) + ".F",
                    hkl_array,
                )

    ph = output[0].cpu().numpy().flatten()

    output_files(
        amplitudes_ord,
        [ph],
        name_infile[: len(name_infile) - 4] + ".F",
        name_infile[: len(name_infile) - 4] + "_phase_extension.F",
        hkl_array,
    )

    return hkl_array, amplitudes_ord, ph


# def get_PhAI_phases_phil(f_sq_obs, randomize_phases=0, cycles=1, F=False, INPUT_IS_SQUARED=True):

def get_PhAI_phases(f_sq_obs, randomize_phases=0, cycles=1, name_infile="", INPUT_IS_SQUARED=True, **kwargs):
    """
    Get PhAI phases for the loaded file.

    Parameters:
    - randomize_phases: If set to 1, randomizes the initial phases.
    - cycles: Number of cycles to run the phase extension.

    Returns:
    - None
    """
    try:
        # in case we're in olex2
        Fs = f_sq_obs.as_amplitude_array().data().as_numpy_array()
        indices = f_sq_obs.indices()
    except:
        Fs = f_sq_obs[0]
        indices = f_sq_obs[1]

    if INPUT_IS_SQUARED:
        Fs = np.sqrt(np.abs(Fs))

    max_index = model_args["max_index"]
    hkl_array = create_hkl_array(max_index)

    indices, Fs = prepare_reflections(
        indices, Fs
    )  # comes from: crystallography_module.merge_reflections(H_tmp, Fabs_tmp)

    amplitudes = create_amplitudes(Fs, indices, max_index)
    amplitudes_ord = create_amplitudes_ord(amplitudes, max_index)

    # print(hkl_array)
    # print(amplitudes_ord)
    # print(np.sum(amplitudes_ord))
    # print(dddd)

    if randomize_phases:
        init_phases = randomize_output(torch.zeros_like(amplitudes))
    else:
        init_phases = torch.zeros_like(amplitudes)

    for i in range(cycles):
        print("cycle: ", i + 1)
        if i == 0:
            output = phases(model(amplitudes, init_phases))
        else:
            # COMPARE HERE!
            init_phases = copy_new_phases_to_initial(
                hkl_array, init_phases, output, max_index
            )
            # has an error:
            # init_phases = copy_new_phases_to_initial_flo(hkl_array, init_phases, output)
            output = phases(model(amplitudes, init_phases))

            if name_infile:
                output_files(
                    amplitudes_ord,
                    output,
                    name_infile[: len(name_infile) - 4] + "_" + str(i + 1) + ".F",
                    name_infile[: len(name_infile) - 4]
                    + "_phase_extension_"
                    + str(i + 1)
                    + ".F",
                    hkl_array,
                )

    ph = output[0].cpu().numpy().flatten()

    if name_infile:
        output_files(
            amplitudes_ord,
            [ph],
            name_infile[: len(name_infile) - 4] + ".F",
            name_infile[: len(name_infile) - 4] + "_phase_extension.F",
            hkl_array,
        )

    return hkl_array, amplitudes_ord, ph

