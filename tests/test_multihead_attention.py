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


@pytest.mark.parametrize(
    "batch_size, seq_len, d_model, num_heads",
    [
        (1, 10, 64, 8),  # Case 1: 간단한 설정
        (2, 15, 128, 8),  # Case 2: 조금 더 큰 설정
    ],
)
def test_ROPE가_존재하는_멀티헤드_어텐션을_테스트한다(
    batch_size, seq_len, d_model, num_heads
):
    """
    Given a dummy input tensor and a MultiHeadAttention layer,
      and given dummy rope embeddings (cos, sin) with the correct shape,
    When the layer is invoked with these rope embeddings,
    Then the output tensor should have the same shape as (batch_size, seq_len, d_model).
    """
    # Given: d_model이 num_heads로 나누어 떨어져야 합니다.
    if d_model % num_heads != 0:
        pytest.skip("d_model must be divisible by num_heads")

    # Given: 입력 텐서를 무작위로 생성 (shape: [batch_size, seq_len, d_model])
    dummy_input = tf.random.uniform((batch_size, seq_len, d_model), dtype=tf.float32)

    # Given: dropout 없이, layer_id=1 (즉, 로타리 임베딩 적용 케이스)로 MultiHeadAttention 인스턴스를 생성합니다.
    mha = MultiHeadAttention(d_model, num_heads, dropout_rate=0.0, layer_id=1)

    # Given: 각 헤드의 차원 계산 (d_model // num_heads)
    head_dim = d_model // num_heads
    # Given: rope_embeds를 위한 더미 cos, sin 텐서를 생성합니다.
    # - 기대하는 rope_embeds의 shape는 [seq_len, 2 * head_dim]

    # Q: dummy_cos 가 모두 1 이라는 cosine tensor 의 의미는 ?
    # 실제로는 각 토큰의 위치가 cos 값에 따라 달라지겠지만, 더미 값으로 모든 위치를 1을 반환하게 해서 회전 효과가 없다는 것을 테스트한다.

    # Q: dummy_sin 이 모두 0 인 sin tensor 의 의미는 ?
    # 실제로는 임베딩이 sine 값 이 위치에 따라 달라지지만 모든 값을 0 으로 반환하게 해서 회전 효과가 없다는 것을 테스트한다.

    dummy_cos = tf.ones((seq_len, 2 * head_dim), dtype=tf.float32)  # 모든 값 1
    dummy_sin = tf.zeros((seq_len, 2 * head_dim), dtype=tf.float32)  # 모든 값 0
    rope_embeds = (dummy_cos, dummy_sin)

    # When: rope_embeds를 포함하여 MultiHeadAttention 레이어를 호출합니다.
    output = mha(dummy_input, mask=None, rope_embeds=rope_embeds, training=False)

    # Then: 출력 텐서의 shape가 (batch_size, seq_len, d_model)와 동일해야 합니다.
    expected_shape = (batch_size, seq_len, d_model)
    assert (
        output.shape == expected_shape
    ), f"Output shape mismatch with rope_embeds: expected {expected_shape}, got {output.shape}"

    # 그리고, 함수형 검증 도구를 활용하여 출력값의 일관성을 확인합니다.
    np.testing.assert_allclose(
        output.numpy(),
        output.numpy(),
        err_msg="Output values are inconsistent when using rope_embeds.",
    )


@pytest.mark.parametrize(
    "batch_size, seq_len, d_model, num_heads",
    [
        (1, 10, 64, 8),
        (2, 15, 128, 8),
    ],
)
def test_ROPE가_결과값에_영향을_미치는지_확인한다(
    batch_size, seq_len, d_model, num_heads
):
    """
    Given:
      - A dummy input tensor of shape [batch_size, seq_len, d_model] and a MultiHeadAttention layer (layer_id=1).
      - Dummy rope embeddings with non-trivial random values.
    When:
      - The layer is invoked with rope_embeds and without rope_embeds.
    Then:
      - The outputs should differ, demonstrating that rope embeddings are applied.
    """
    # Given: dummy input 생성
    dummy_input = tf.random.uniform((batch_size, seq_len, d_model), dtype=tf.float32)

    # And: MultiHeadAttention 레이어 생성 (layer_id=1)
    mha = MultiHeadAttention(d_model, num_heads, dropout_rate=0.0, layer_id=1)

    # When: rope_embeds 없이 호출
    output_no_rope = mha(
        inputs=dummy_input, mask=None, rope_embeds=None, training=False
    )

    # And: non-trivial dummy rope embeddings 생성
    head_dim = d_model // num_heads
    dummy_cos = tf.random.uniform((seq_len, 2 * head_dim), dtype=tf.float32)
    dummy_sin = tf.random.uniform((seq_len, 2 * head_dim), dtype=tf.float32)
    rope_embeds = (dummy_cos, dummy_sin)

    # When: rope_embeds를 적용하여 호출
    output_with_rope = mha(
        inputs=dummy_input, mask=None, rope_embeds=rope_embeds, training=False
    )

    # Then: 두 출력은 거의 동일하면 안 되어야 합니다.
    assert not np.allclose(
        output_no_rope.numpy(), output_with_rope.numpy(), atol=1e-6
    ), "Output with rope_embeds should differ from output without rope_embeds."


@pytest.mark.parametrize(
    "batch_size, num_heads, seq_len, head_dim",
    [
        (2, 4, 10, 16),
        (1, 8, 12, 8),
    ],
)
def test_scaled_dot_product_attention(batch_size, num_heads, seq_len, head_dim):
    """
    Given:
      - Dummy query, key, and value tensors of shape [batch_size, num_heads, seq_len, head_dim].
    When:
      - The scaled_dot_product_attention method is invoked with these tensors (mask=None, training=False).
    Then:
      - The output tensor should have shape [batch_size, num_heads, seq_len, head_dim].
    """
    # Given: dummy query, key, value 생성
    q = tf.random.uniform((batch_size, num_heads, seq_len, head_dim), dtype=tf.float32)
    k = tf.random.uniform((batch_size, num_heads, seq_len, head_dim), dtype=tf.float32)
    v = tf.random.uniform((batch_size, num_heads, seq_len, head_dim), dtype=tf.float32)

    # And: MultiHeadAttention 레이어를 생성 (d_model = num_heads * head_dim)
    mha = MultiHeadAttention(
        d_model=num_heads * head_dim, num_heads=num_heads, dropout_rate=0.0, layer_id=1
    )

    # When: scaled_dot_product_attention 호출
    output = mha.scaled_dot_product_attention(q, k, v, mask=None, training=False)

    # Then: 출력 shape가 [batch_size, num_heads, seq_len, head_dim]인지 검증
    expected_shape = (batch_size, num_heads, seq_len, head_dim)
    assert (
        output.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {output.shape}"


@pytest.mark.parametrize(
    "batch_size, num_heads, seq_len, head_dim",
    [
        (2, 4, 10, 16),
        (1, 8, 12, 8),
    ],
)
def test_apply_rotary_pos_emb(batch_size, num_heads, seq_len, head_dim):
    """
    Given:
      - Dummy query and key tensors of shape [batch_size, num_heads, seq_len, head_dim],
      - Dummy rope embeddings (cos: all ones, sin: all zeros) of shape [seq_len, 2 * head_dim].
    When:
      - apply_rotary_pos_emb is invoked with these dummy rope embeddings.
    Then:
      - The returned q and k tensors must have the same shape as input.
      - With dummy_cos=1 and dummy_sin=0, the rotation effect is null, so q and k should remain unchanged.
    """
    # Given: dummy q, k 생성
    q = tf.random.uniform((batch_size, num_heads, seq_len, head_dim), dtype=tf.float32)
    k = tf.random.uniform((batch_size, num_heads, seq_len, head_dim), dtype=tf.float32)

    # And: Dummy rope embeddings 생성 (cos는 모두 1, sin은 모두 0)
    dummy_cos = tf.ones((seq_len, 2 * head_dim), dtype=tf.float32)
    dummy_sin = tf.zeros((seq_len, 2 * head_dim), dtype=tf.float32)

    mha = MultiHeadAttention(
        d_model=num_heads * head_dim, num_heads=num_heads, dropout_rate=0.0, layer_id=1
    )

    # When: apply_rotary_pos_emb 호출
    q_new, k_new = mha.apply_rotary_pos_emb(q, k, dummy_cos, dummy_sin)

    # Then: 반환된 q_new, k_new의 shape가 원래 q, k와 동일해야 함
    assert q_new.shape == q.shape, f"Expected q shape {q.shape}, got {q_new.shape}"
    assert k_new.shape == k.shape, f"Expected k shape {k.shape}, got {k_new.shape}"

    # Then: dummy rope embeddings (cos=1, sin=0)이므로 회전 효과가 없어야 하므로 값이 그대로여야 합니다.
    np.testing.assert_allclose(
        q_new.numpy(), q.numpy(), atol=1e-6, err_msg="q values differ with dummy rope"
    )
    np.testing.assert_allclose(
        k_new.numpy(), k.numpy(), atol=1e-6, err_msg="k values differ with dummy rope"
    )
