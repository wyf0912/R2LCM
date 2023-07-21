from compressai.zoo import raw_hyperprior, raw_context, learned_context, learned_context_journal, learned_context_journal2, learned_context_journal3, learned_context_journal4

journal_cvpr22_J4_list = [
    {
        "path": "checkpoints/learned_context_journal4_qual_1_lamd_2.40e-01_stride_2_down2_raw_linear_l1_sample4_reduce-c8__nocompress_p256_use_deconv_CVPR22_AuxForward_repeat_olympus_noise_adaptQuant_bs8.pth.tar",
        "model": learned_context_journal4,  # 0
        "down_num": 2,
        "quality": 1,
        "nocompress": True,
        "reduce_c": 8,
        "demosaic": True,
        "sampling_num": 4,
        "discrete_context": True,
        "use_deconv": True,
        "only_demosaic": True,
        "stride": 2,
        "rounding": "noise",
        "adaptive_quant": True,
        "only_demosaic": False,
        # "drop_pixel": True,
        # "gmm_num": 3
    },
    {
        "path": "checkpoints/learned_context_journal4_qual_1_lamd_2.40e-01_stride_2_down2_raw_linear_l1_sample4_reduce-c8__nocompress_p256_use_deconv_CVPR22_AuxForward_samsung_noise_adaptQuant_bs8.pth.tar",
        "model": learned_context_journal4,  # 1
        "down_num": 2,
        "quality": 1,
        "nocompress": True,
        "reduce_c": 8,
        "demosaic": True,
        "sampling_num": 4,
        "discrete_context": True,
        "use_deconv": True,
        "only_demosaic": True,
        "stride": 2,
        "rounding": "noise",
        "adaptive_quant": True,
        "only_demosaic": False,
        # "drop_pixel": True,
        # "gmm_num": 3
    },
    {
        "path": "checkpoints/learned_context_journal4_qual_1_lamd_2.40e-01_stride_2_down2_raw_linear_l1_sample4_reduce-c8__nocompress_p256_use_deconv_CVPR22_AuxForward_sony_noise_adaptQuant_bs8.pth.tar",
        "model": learned_context_journal4,  # 2
        "down_num": 2,
        "quality": 1,
        "nocompress": True,
        "reduce_c": 8,
        "demosaic": True,
        "sampling_num": 4,
        "discrete_context": True,
        "use_deconv": True,
        "only_demosaic": True,
        "stride": 2,
        "rounding": "noise",
        "adaptive_quant": True,
        "only_demosaic": False,
        # "drop_pixel": True,
        # "gmm_num": 3
    }
]


