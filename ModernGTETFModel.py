import keras
import tensorflow as tf


class MultiHeadAttention(keras.layers.Layer):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout_rate: float = 0.1,
        layer_id: int = 0,
        **kwargs,
    ) -> None:
        """
        Initializes the MultiHeadAttention layer.

        :param d_model: 모델의 전체 차원
        :param num_heads: 멀티헤드 어텐션에서의 헤드 개수
        :param dropout_rate: 드롭아웃 비율
        :param layer_id: 레이어의 식별자 (예: 0이면 첫 번째 레이어, 0이 아니면 Layer Normalization 적용)
        :param kwargs: 추가적인 keyword arguments
        """
        super().__init__(**kwargs)
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.d_model: int = d_model
        self.num_heads: int = num_heads
        self.layer_id: int = layer_id
        self.dropout_rate: float = dropout_rate
        self.depth: int = d_model // num_heads  # 각 헤드의 차원
        self.head_dim = self.depth

        # layer_id가 0이 아니면 입력 정규화를 적용 (원본 코드에서는 항상 적용함)
        # 여기서는 minimal 구현 단계에서는 layer_id에 관계없이 단순히 Dense 연산을 수행하도록 할 수 있지만,
        # 최종 구현에서는 입력을 정규화한 후 QKV 프로젝션을 진행하게 됩니다.
        self.attnNorm = keras.layers.LayerNormalization(epsilon=1e-5, center=False)

        # QKV 프로젝션: 출력 차원은 3 * d_model, bias 없음
        # 출력 차원이 3 * d_model인 이유는 입력 텐서로부터 한 번의 Dense 연산으로 Q, K, V 세 가지 벡터를 동시에 얻기 위함입니다.
        self.wqkv = keras.layers.Dense(3 * d_model, use_bias=False)

        # 최종 출력 프로젝션. 출력 하는 차원은 d_model
        # 최종 어텐션 결과를 다시 원래의 차원인 d_model로 투영(projection)하기 위한 Dense 레이어입니다.
        # 이 과정을 통해 멀티헤드 어텐션의 여러 헤드에서 나온 결과를 하나의 텐서로 합칩니다.
        self.o = keras.layers.Dense(d_model, use_bias=False)

        # Dropout Layer
        self.dropout = keras.layers.Dropout(rate=dropout_rate)

    def apply_rotary_pos_emb(
        self, q: tf.Tensor, k: tf.Tensor, cos: tf.Tensor, sin: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Applies the rotary positional embedding to query and key tensors.

        :param q: Query tensor
        :param k: Key tensor
        :param cos: Cosine tensor for rotary embedding
        :param sin: Sine tensor for rotary embedding
        :return: Tuple containing modified query and key tensors (q_embed, k_embed)
        """

        # 이 함수는 주어진 텐서를 마지막 차원에서 두 부분으로 분할한 후,
        # 두 번째 절반의 부호를 반전시키고, 순서를 뒤바꿔서 첫 번째 절반과 결합합니다.
        # 예를 들어, 입력 벡터가 [x1, x2]라면, 출력은 [-x2, x1]가 됩니다.
        def _rotate_half(x: tf.Tensor) -> tf.Tensor:
            # 마지막 차원(axis=-1)을 두 개의 동일한 크기의 텐서로 분할합니다.
            x1, x2 = tf.split(value=x, num_or_size_splits=2, axis=-1)

            # x2의 부호를 반전한 후, x1과 함께 이어 붙입니다.
            # Q: 왜 마지막 차원을 분할하냐?
            # Transformer에서는 각 토큰마다 하나의 임베딩 벡터가 있으며
            # MultiHeadAttention에서는 각 헤드마다 그 벡터의 일부(즉, head_dim)를 사용합니다.
            # 이 마지막 차원이 실제 피처(특징)들이 담긴 부분이기 때문에, 여기서 정보를 반으로 나누어 회전 변환을 적용하는 것이 자연스럽습니다.
            return tf.concat([-x2, x1], axis=-1)

        # 여기서 cos, sin은 원래 [seq_len, 2 * head_dim] 형태일 수 있습니다.
        # reshape을 통해 [1, 1, seq_len, head_dim] 형태로 바꿉니다.
        #  - 첫 번째 1: 배치 차원에 대해 확장 (모든 배치에 동일한 값을 사용)
        #  - 두 번째 1: 헤드 차원에 대해 확장 (모든 헤드에 동일하게 적용)
        #  - -1: seq_len을 그대로 유지
        #  - self.head_dim: 마지막 차원은 각 헤드의 차원 크기
        num_heads, seq_len, head_dim = 1, 1, -1

        cos = tf.reshape(cos, [num_heads, seq_len, head_dim, self.head_dim])
        sin = tf.reshape(sin, [num_heads, seq_len, head_dim, self.head_dim])

        # 위의 reshape 후, cos와 sin은 [1, 1, 2 * seq_len, head_dim]의 shape를 가지게 됩니다.
        # 하지만 실제로 적용할 때는 query q와 key k의 시퀀스 길이에 맞춰야 하므로,
        # 필요한 부분만 슬라이싱하여 [1, 1, seq_len, head_dim]의 shape로 맞춥니다.
        cos = tf.reshape(cos, [1, 1, -1, self.head_dim])
        sin = tf.reshape(sin, [1, 1, -1, self.head_dim])

        # tf.shape(q)[2]는 q 텐서의 세 번째 차원, 즉 시퀀스 길이 S를 나타냅니다.
        # 첫 번째와 두 번째 차원은 그대로 유지하고, 세 번째 차원에서 처음 seq_len (즉, tf.shape(q)[2]) 개의 값만 선택합니다.
        cos = cos[:, :, :tf.shape(q)[2], :]
        sin = sin[:, :, :tf.shape(q)[2], :]

        # 이제 rotary positional embedding을 적용합니다.
        # 각 query 벡터에 대해, cosine 값과 sine 값을 사용하여 회전 변환을 수행합니다.
        # 공식은 다음과 같습니다:
        #   q_embed = (q * cos) + (rotate_half(q) * sin)
        # 이는, 각 원소를 두 부분으로 나누고, 이 두 부분에 대해 회전 행렬의 효과를 벡터화한 형태라고 볼 수 있습니다.
        q_embed = (q * cos) + (_rotate_half(q) * sin)

        # 동일하게, key 벡터에도 같은 변환을 적용합니다.
        k_embed = (k * cos) + (_rotate_half(k) * sin)

        # 최종적으로, 변환된 query와 key 텐서를 반환합니다.
        return q_embed, k_embed


    def scaled_dot_product_attention(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        mask: tf.Tensor | None = None,
        training: bool | None = None,
    ) -> tf.Tensor:
        """
        Computes the scaled dot-product attention.

        :param query: Query tensor of shape (..., seq_len_q, depth)
        :param key: Key tensor of shape (..., seq_len_k, depth)
        :param value: Value tensor of shape (..., seq_len_v, depth_v)
        :param mask: Optional attention mask tensor
        :param training: Boolean flag indicating training mode
        :return: Tensor resulting from the attention operation
        """

        raise NotImplementedError()

    def call(
        self,
        inputs: tf.Tensor,
        mask: tf.Tensor | None = None,
        rope_embeds: tuple[tf.Tensor, tf.Tensor] | None = None,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Performs the forward pass of the MultiHeadAttention layer.

        :param inputs: Input tensor of shape (batch_size, seq_len, d_model)
        :param mask: Optional mask tensor of shape (batch_size, 1, seq_len, seq_len)
        :param rope_embeds: Optional tuple (cos, sin) for rotary positional embeddings
        :param training: Boolean flag indicating whether the layer is in training mode
        :return: Output tensor of shape (batch_size, seq_len, d_model)
        """

        # Dense 연산을 통해 입력이 변환되므로, minimal 구현보다 입력과 출력가 달라지게 됩니다.
        # 출력 shape는 여전히 [B, S, d_model]이어야 합니다.
        # 하지만 QKV 분리와 실제 attention 계산은 아직 없습니다.

        # 입력을 정규화 하는데 항상 적용해봅니다.
        x = self.attnNorm(inputs=inputs)

        # 입력 텐서의 shape가 [B, S, d_model] 에서, Dense 레이어를 통해 한 번에 Query, Key, Value를 계산하면
        # 결과는 [B, S, 3 * d_model]이 됩니다.
        # 여기서 3은 Q, K, V를 한 번에 계산했기 때문에 붙은 차원입니다.
        # QKV projection. [B, S, 3 * d_model] shape
        qkv = self.wqkv(inputs=x)

        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        # reshape: [B, S, 3, num_heads, depth]
        # 우리는 각 Q, K, V를 여러 헤드로 분할하여 병렬로 어텐션을 계산하려고 합니다.
        # 이를 위해, 전체 d_model을 num_heads개의 헤드로 나누면 각 헤드의 차원은
        # depth = d_model / num_heads가 됩니다.

        # 우리는 각 Q, K, V를 여러 헤드로 분할하여 병렬로 어텐션을 계산하려고 합니다.
        # 이를 위해, 전체 d_model을 num_heads개의 헤드로 나누면 각 헤드의 차원은
        # depth = d_model / num_heads가 됩니다.
        # Dense의 출력 [B, S, 3*d_model]을 [B, S, 3, num_heads, depth]로 reshape하면,
        # 마지막 두 차원으로 나눠서 각 헤드의 정보를 분리할 수 있습니다.
        # 이 때, 3이라는 차원은 Q, K, V를 구분하기 위한 것입니다.
        qkv = tf.reshape(qkv, [batch_size, seq_len, 3, self.num_heads, self.depth])

        # transpose: [3, B, num_heads, S, depth]
        # tf.transpose(qkv, perm=[2, 0, 3, 1, 4])를 수행하면
        # 첫 번째 차원이 3이 되어 Q, K, V를 구분하기 쉬워집니다.
        # 이때 각 텐서의 shape는
        # Q: [B, num_heads, S, depth]
        # K: [B, num_heads, S, depth]
        # V: [B, num_heads, S, depth]
        # 이렇게 하면 어텐션 연산(예를 들어, Query와 Key의 내적 계산)을 헤드 단위로 병렬 처리하기 용이해집니다.
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])

        # tf.unstack(qkv, axis=0)를 사용하여, 첫 번째 차원(3)을 기준으로 Q, K, V를 분리합니다.
        # 분리된 각 텐서의 shape는 [B, num_heads, S, depth]가 됩니다.
        q, k, v = tf.unstack(qkv, axis=0)

        # Step 3: 실제 어텐션 계산 전, 단순히 v 텐서를 사용하여 출력 구성
        # 원래 [B, num_heads, S, depth]에서
        # [B, S, num_heads, depth]로 재배열되어, 시퀀스 차원 S가 헤드 차원보다 앞쪽에 오게 됩니다.
        v_transposed = tf.transpose(v, perm=[0, 2, 1, 3])

        # reshape: [B, S, d_model]
        # v_transposed의 마지막 두 차원인 [num_heads, depth]를 하나의 차원으로 결합합니다.
        # Transformer의 최종 출력은 각 토큰마다 하나의 벡터가 되어야 하며, 이 벡터의 차원은 원래 모델 차원 d_model이어야 합니다.
        # 이 과정이 바로 여러 헤드의 결과를 하나의 텐서로 결합하는 역할을 합니다.
        # 따라서 reshape을 통해 [B, S, num_heads * depth] → [B, S, d_model] 으로 만듭니다.
        attention_output = tf.reshape(v_transposed, [batch_size, seq_len, self.d_model])

        # 최종 출력 프로젝션
        output = self.o(inputs=attention_output)

        return self.dropout(inputs=output, training=training) if training else output


def create_local_sliding_window_mask(
    global_mask_4d: tf.Tensor, window_size: int
) -> tf.Tensor:
    """
    ModernBERT 의 local attention 을 구현하기 위하여 local sliding window mask 를 만들어봅시다.

    :param global_mask_4d: (batch_size, 1, seq_len, seq_len) 을 가지는 4d tensor 입니다.
    global_mask_4d는 이미 글로벌 마스크로, 예를 들어 패딩 토큰의 위치를 -∞로 지정해 둔 텐서입니다.

    :param window_size: 128 토큰 사이즈 수준으로 확인하는 local attention window 의 크기 입니다.
    window_size // 2 에 존재하는 Position은 valid하다고 처리 합니다.

    :return: final_local_4d (tf.Tensor): (batch_size, 1, seq_len, seq_len) 를 가지는데, 다음의 형상을 지닙니다.
        - sliding window 내부에 있고 global mask에 valid 하다면 0.0이다.
        - sliding window 외부에 있다면 padding이 되므로 -∞ 이다.
    """

    global_mask_4d_shape: tf.shape = tf.shape(
        global_mask_4d
    )  # (batch_size, height, width, channels)
    batch_size: int = global_mask_4d_shape[0]
    seq_len: int = global_mask_4d_shape[-1]

    # (seq_len, 1) 과 (1, seq_len) 을 만들어서 두 텐서를 서로 빼거나 다른 연산을 수행하기에 용이하도록 만듭니다.
    rows: tf.range = tf.range(seq_len)[
        :, None
    ]  # (seq_len,) 인 1차원 텐서를 만들고 그 차원을 2차원 텐서로 확장해서 (seq_len, 1)
    cols: tf.range = tf.range(seq_len)[
        None, :
    ]  # 동일하게, (1, seq_len) 의 2차원 텐서를 만든다.
    distance = tf.abs(rows - cols)  # (seq_len, seq_len) 의 텐서가 됩니다.

    # distance의 내부는 True로, 외부는 False로 설정해본다.
    HALF_WINDOW_SIZE: int = window_size // 2
    window_bool_2d = tf.less_equal(distance, HALF_WINDOW_SIZE)
    inside_zero_values: tf.zeros = tf.zeros(
        [seq_len, seq_len], dtype=global_mask_4d.dtype
    )
    outside_infinite_values = tf.fill([seq_len, seq_len], global_mask_4d.dtype.min)

    # window_bool_2d는 (seq_len, seq_len) 크기의 Boolean 행렬입니다.
    # 두 토큰 간의 거리가 윈도우 내(즉, distance <= half_w)이면 True, 그렇지 않으면 False입니다.
    # tf.where를 사용하여, Boolean 마스크가 True인 위치는 0.0, False인 위치는 매우 작은 값(-∞와 유사한 값)을 할당합니다.
    local_mask_2d = tf.where(
        condition=window_bool_2d, x=inside_zero_values, y=outside_infinite_values
    )

    # local_mask_4d의 경우 현재 local_mask_2d는 (seq_len, seq_len) 의 shape을 가지고 있기에
    # 어텐션 계산 시점에서 각 배치에 대해 동일한 마스크를 적용해야 하므로 -> 마스크의 shape를 (batch_size, 1, seq_len, seq_len) 으로 합니다.
    # 1) local_mask_2d[None, None, :, :]를 통해 shape를 (1, 1, seq_len, seq_len)으로 확장합니다.
    # 2) tf.tile을 사용하여 배치 크기만큼 복제합니다.
    # 3) 모든 배치에 대해 동일한 슬라이딩 윈도우 마스크가 적용된 4D 텐서(local_mask_4d)가 만들어집니다.
    local_mask_4d = tf.tile(
        input=local_mask_2d[None, None, :, :], multiples=[batch_size, 1, 1, 1]
    )

    # 로컬 마스크와 글로벌 마스크를 합산함으로써, 두 조건 중 하나라도 마스킹해야 하면 최종적으로 해당 위치가 -∞가 되도록 만듭니다.
    # 두 마스크를 더하면, 내부(윈도우 내)에서는 0.0 + 0.0 = 0.0
    # 외부(윈도우를 벗어난 부분)에서는 0.0 + (-∞) = -∞ (혹은 매우 작은 값)이 되어 softmax 연산 시 무시됩니다.
    final_local_4d = global_mask_4d + local_mask_4d

    return final_local_4d
