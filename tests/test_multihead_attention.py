import tensorflow as tf
import numpy as np
import pytest

from ModernGTETFModel import MultiHeadAttention


@pytest.mark.parametrize(
    "batch_size, seq_len, d_model, num_heads, dropout_rate, training, layer_id",
    [
        (
            1,
            128,
            64,
            8,
            0.0,
            False,
            0,
        ),  # Case 1: 배치를 먼저 작게하고, 기본 설정으로 만들어봅니다. layer_id가 0인 경우 layer norm을 적용하지 않고 원시 임베딩.
        (2, 256, 128, 16, 0.0, False, 0),  # Case 2: 중간 배치로 올려보겠습니다.
        (
            3,
            512,
            64,
            4,
            0.1,
            True,
            1,
        ),  # Case 3 : 3 개의 배치에 대해 dropout 0.1, training = True 이고 layer_id가 1 인 상태입니다.
        # layer_id가 1 인 경우 입력을 먼저 layer norm 하고 q, k, v를 계산합니다. 안정적인 학습을 위한 전략입니다.
    ],
)
def test_ROPE가_존재하지_않는_멀티헤드_어텐션을_테스트한다(
    batch_size, seq_len, d_model, num_heads, dropout_rate, training, layer_id
):
    """
    test_ROPE가_존재하지_않는_멀티헤드_어텐션을_테스트한다: rope_embed 없이 MHA 의 기본 출력을 검증해봅시다.

    Given:
      - 임의의 값으로 채워진 dummy input tensor (shape: [batch_size, seq_len, d_model]).
      - d_model은 num_heads로 나누어 떨어지는 값이어야 하며,
      - MultiHeadAttention 레이어가 주어진 파라미터로 초기화됩니다.
    When:
      - 해당 레이어가 mask와 rope_embeds 없이 dummy input에 대해 forward pass를 수행합니다.
    Then:
      - 출력 tensor의 shape는 [batch_size, seq_len, d_model]이어야 합니다.
      - 최종 구현에서는 입력(dummy input)과 비교하여, 내부의 QKV 프로젝션 및 attention, 출력 프로젝션을 통해
        값이 변환되어야 하므로, output은 dummy input과 거의 동일하지 않아야 합니다.
    """
    # Given : d_model 이 num_heads로 나누어 떨어져야만 합니다.
    if d_model % num_heads != 0:
        pytest.fail("d_model must be divisible by num_heads")

    # Given : 입력 텐서를 무작위의 값으로 생성합니다. shape는 [batch_size, seq_len, d_model] 으로 구성되어야 합니다.
    dummy_input = tf.random.uniform(
        shape=(batch_size, seq_len, d_model), dtype=tf.float32
    )

    # Given : MHA 인스턴스를 생성합니다.
    multi_head_attention: MultiHeadAttention = MultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        layer_id=layer_id,
    )

    # When : 입력 텐서를 전달하여 레이어를 호출해봅니다.
    output: tf.Tensor = multi_head_attention(
        inputs=dummy_input, mask=None, rope_embeds=None, training=training
    )

    # Then : 출력 텐서의 shape 가 (batch_size, seq_len, d_model) 과 동일해야만 합니다.
    expected_shape: tuple = (batch_size, seq_len, d_model)
    assert (
        output.shape == expected_shape
    ), f"Expected output shape {expected_shape}, but got {output.shape}"

    # Then : 출력값이 수치적으로 안정적인지를 shape에 대하여 확인해본다.
    np.testing.assert_allclose(
        output.numpy(), output.numpy(), err_msg="Output values are not consistent."
    )

    # Then : 만약 layer_id가 0이 아니라면, 실제 어텐션의 연산과 출력 프로젝션이 적용되어 dummy_input 과 output이 달라져야 합니다.
    if np.allclose(dummy_input.numpy(), output.numpy(), atol=1e-6):
        pytest.fail(
            "Output is almost identical to input; transformation did not occur as expected."
        )

    # ------------------------------------------------------------------------------------------------ #

    # Given : 각 헤드의 차원을 계산합니다.
    # head_dim: int = d_model // num_heads

    # Given : rope_embeds를 위한 dummy cos, sin 텐서를 생성해봅시다. dummy_cos 의 모든 값은 1이고, dummy_sin 의 모든 값은 0 입니다.
    # rotary 임베딩에서는 각 헤드의 차원을 두 부분으로 나누어 처리하는데, 일반적으로 절반은 원래 값이고 나머지 절반은 rotation 적용을 위해 씁니다.
    # shape 가 (seq_len, 2 * head_dim) 인 Tensor 를 생성하고 그 모든 요소를 1로 채웁니다. 각 위치마다 1이 들어있는 2배 크기의 값을 만듭니다.
    # dummy_cos = tf.ones((seq_len, 2 * head_dim), dtype=tf.float32)
    # dummy_sin = tf.zeros((seq_len, 2 * head_dim), dtype=tf.float32)

    # Q: dummy_cos 가 모두 1 이라는 cosine tensor 의 의미는 ?
    # 실제로는 각 토큰의 위치가 cos 값에 따라 달라지겠지만, 더미 값으로 모든 위치를 1을 반환하게 해서 회전 효과가 없다는 것을 테스트한다.

    # Q: dummy_sin 이 모두 0 인 sin tensor 의 의미는 ?
    # 실제로는 임베딩이 sine 값 이 위치에 따라 달라지지만 모든 값을 0 으로 반환하게 해서 회전 효과가 없다는 것을 테스트한다.
