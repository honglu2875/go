"""Analytical ledger for proposed models; no JAX implementation is qualified.

FMA=2, dense SAME convolution including boundary padding. This excludes
elementwise operations, compiler padding, training and rematerialization work.
The block stack is reused on its own output; the stem and connector run once.
"""
import json


def candidate(unique_blocks, passes, transformer_layers):
    width, connector_channels, expansion, ffn = 768, 64, 4, 2048
    board_points, spatial_channels, global_channels = 81, 22, 19
    kv_width, past_moves = 256, 128
    reference_parameters, reference_flops = 232431872, 37348255744

    # ConvNeXt-style block: DW3x3+bias, affine LN, pointwise expansion+bias,
    # GELU, pointwise contraction+bias, learned channel residual scaling.
    block_matrix = 9 * width + 2 * width * (expansion * width)
    block_other = 9 * width
    stem_matrix = 9 * spatial_channels * width
    stem_other = 3 * width  # convolution bias and affine LayerNorm
    compress_matrix = width * connector_channels
    flat_matrix = board_points * connector_channels * width
    global_matrix = global_channels * width
    encoder_parameters = (
        unique_blocks * (block_matrix + block_other)
        + stem_matrix + stem_other + compress_matrix + connector_channels
        + flat_matrix + width + 2 * width + global_matrix + width
    )

    # GQA: 12 query heads, four KV heads, head dimension 64; SwiGLU FFN.
    trunk_matrix = 2 * width**2 + 2 * width * kv_width + 3 * width * ffn
    transformer_parameters = transformer_layers * (trunk_matrix + 2 * width)
    # Tied action embedding/policy output, action type, final RMSNorm.
    other_parameters = (board_points + 1) * width + 2 * width
    parameters = encoder_parameters + transformer_parameters + other_parameters

    encoder_flops = (
        2 * board_points * (stem_matrix + unique_blocks * passes * block_matrix
                            + compress_matrix)
        + 2 * flat_matrix + 2 * global_matrix
    )
    # Cached previous action and new board; the policy reads the board token.
    new_tokens = 2
    keys = (past_moves + 1) * new_tokens - 1
    transformer_flops = (
        2 * transformer_layers * new_tokens * trunk_matrix
        + 4 * transformer_layers * new_tokens * keys * width
    )
    head_flops = 2 * width * (board_points + 1)
    flops = encoder_flops + transformer_flops + head_flops
    return {
        'unique_encoder_blocks': unique_blocks,
        'encoder_passes': passes,
        'executed_encoder_blocks': unique_blocks * passes,
        'transformer_layers': transformer_layers,
        'encoder_parameters_including_connector': encoder_parameters,
        'transformer_parameters': transformer_parameters,
        'other_parameters': other_parameters,
        'total_parameters': parameters,
        'transformer_to_encoder_parameter_ratio': transformer_parameters / encoder_parameters,
        'parameter_difference_percent': 100 * (parameters / reference_parameters - 1),
        'dense_flops_per_move': flops,
        'dense_flops_by_component': {
            'encoder': encoder_flops, 'transformer': transformer_flops, 'head': head_flops,
        },
        'flop_difference_percent': 100 * (flops / reference_flops - 1),
    }


if __name__ == '__main__':
    print(json.dumps({
        'status': 'proposed_unimplemented_analytical_design',
        'reference': {
            'board_size': 9, 'batch': 128, 'past_moves': 128,
            'cnn_parameters': 232431872, 'cnn_dense_flops_per_move': 37348255744,
        },
        'common': {
            'encoder_width': 768, 'encoder_kernel': 3, 'encoder_expansion': 4,
            'encoder_normalization': 'affine LayerNorm', 'encoder_stride': 1,
            'connector_channels': 64, 'visual_tokens_per_board': 1,
            'transformer_width': 768, 'ffn_hidden': 2048,
            'query_heads': 12, 'kv_heads': 4, 'tokens_per_move': 2,
            'policy_heads': 1,
        },
        'proposed_matched_candidates': [candidate(24, 2, 18), candidate(16, 3, 24)],
        'unmatched_single_pass_arithmetic_only': [candidate(24, 1, 18), candidate(16, 1, 24)],
        'limits': [
            'Shape ledger, not an executable model, measured latency, MFU or learning result.',
            'Elementwise operations, compiler padding and differentiated training work are excluded.',
            'Weight sharing and encoder/transformer allocation are explicit interventions.',
            'The flattening connector and action vocabulary here are specific to the 9x9 study.',
            'A reference-point FLOP match does not imply matching at other history lengths.',
        ],
    }, indent=2, sort_keys=True))
