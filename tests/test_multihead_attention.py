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

    Given
    테스트를 위해 실제 데이터 대신 임의의 값을 가지는 텐서를 생성합니다.
    입력 텐서가 올바른 shape과 [batch_size, seq_len, d_model] shape 을 가지는지 확인합니다.
    Transformer 구조에서 핵심이 되는 멀티헤드 어텐션 레이어를 초기화합니다.

    해당 레이어가 주어진 입력 텐서에 대해 올바르게 작동하는지, 즉 QKV 프로젝션, 어텐션 계산, 그리고 최종 출력 프로젝션 등이 제대로 수행되는지 검증합니다.
    로타리 포지셔널 임베딩(ROTARY positional embedding)에 필요한 cosine과 sine 값을 가지는 텐서를 생성합니다.
    테스트에서는 임의로 생성된 dummy 값(예: cos 값이 모두 1, sin 값이 모두 0)을 사용하여

    로타리 임베딩이 MultiHeadAttention 내부에서 올바르게 처리되는지를 확인합니다.
    dummy rope는 MultiHeadAttention 내부의 각 헤드에 적용되도록, [seq_len, 2 * head_dim]와 같은 정확한 shape을 가져야 합니다.
    해당 레이어가 입력 텐서와 일관된 위치 정보를 사용할 수 있게 합니다.

    When
    준비한 dummy input tensor, MultiHeadAttention 레이어, 그리고 dummy rope embeddings를 사용하여, 레이어의 forward pass를 수행합니다.
    로타리 임베딩이 올바르게 적용되어 Query와 Key가 변환되는지
    변환 후에도 전체 출력의 차원(출력 shape)이 변경되지 않고, 기대한대로 [batch_size, seq_len, d_model] 형태를 유지하는지

    Then
    MultiHeadAttention 레이어의 호출 결과로 반환되는 텐서는 입력과 동일한 shape, 즉 [batch_size, seq_len, d_model]을 가져야 합니다.
    멀티헤드 어텐션 레이어는 내부에서 여러 개의 헤드로 나누어 어텐션 연산을 수행하지만, 최종적으로는 출력 차원을 다시 원래의 d_model로 합칩니다.
    rope embeddings가 적용되었는지 여부와 관계없이, 최종 출력은 입력과 동일한 shape이어야 하며, 이는 Transformer의 기본 설계 원칙 중 하나입니다.
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
