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
        cos = cos[:, :, : tf.shape(q)[2], :]
        sin = sin[:, :, : tf.shape(q)[2], :]

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
        query, key: [B, num_heads, S, head_dim], value: [B, num_heads, S, head_dim]

        :param query: Query tensor of shape (..., seq_len_q, depth)
        :param key: Key tensor of shape (..., seq_len_k, depth)
        - 각각 [배치 크기(B), 헤드 수(num_heads), 시퀀스 길이(S), 각 헤드의 차원(head_dim)]의 텐서입니다.

        :param value: Value tensor of shape (..., seq_len_v, depth_v)
        - [B, num_heads, S, head_dim] 형태의 텐서입니다.

        :param mask: Optional attention mask tensor
        :param training: Boolean flag indicating training mode
        :return: Tensor resulting from the attention operation
        """

        # 1. query와 key의 내적: [B, num_heads, S, S]
        # 여기서 transpose_b=True는 key 텐서의 마지막 두 차원을 전치(transpose)하여 곱셈을 수행함을 의미합니다.
        # 내적을 계산하기 위해 Query 벡터는 [B, num_heads, S, head_dim]이고
        # Key 벡터는 내적 대상의 각 요소와 곱해질 수 있도록 마지막 두 차원이 전치되어 [B, num_heads, head_dim, S] 형태가 되어야 합니다.
        # 결과 텐서의 shape는 [B, num_heads, S, S]가 됩니다.
        # → 각 배치와 각 헤드별로 query의 각 시퀀스 토큰이 key의 각 시퀀스 토큰과 얼마나 관련 있는지를 나타내는 점수(score) 행렬을 생성합니다.
        matmul_qk: tf.Tensor = tf.matmul(query, key, transpose_b=True)

        # Scaling: key 텐서의 마지막 차원(즉, head_dim)을 얻습니다.
        head_dim: int = tf.shape(key)[-1]

        # Transformer의 점곱 어텐션(Scaled Dot-Product Attention)에서,
        # query와 key의 내적 값은 각 벡터의 차원 수에 비례하여 커지는 경향이 있습니다.
        # 이로 인해 softmax 함수에 입력되는 값들이 너무 커져서, 기울기 소실 문제가 발생할 수 있습니다.
        # 따라서, 내적 결과를 sqrt(dk) 로 나누어 스케일링(scaling)하는데, 이때 dk가 바로 각 key 벡터의 차원입니다.
        dk = tf.cast(head_dim, dtype=query.dtype)  # dimension of key

        # 이 연산은 내적 값이 너무 커지는 것을 sqrt(dk) 로 방지하여 softmax 연산 시 안정성을 높여줍니다.
        scaled_logits: tf.Tensor = matmul_qk / tf.math.sqrt(dk)

        # 결과는 scaled_logits에 저장되며, 여전히 shape은 [B, num_heads, S, S]입니다.
        # 마스크가 주어지면 추가 (mask는 보통 -∞ 또는 0.0으로 구성됨)
        scaled_logits += mask if mask is not None else 0

        # tf.nn.softmax를 사용하여, 스케일된 logits에 대해 softmax를 적용합니다.
        # axis=-1로 지정하여 마지막 차원(즉, 각 query 토큰에 대해 모든 key 토큰에 걸친 점수)에 대해 확률 분포를 만듭니다.
        # 이 결과는 각 query 토큰이 key의 각 토큰에 얼마나 집중해야 하는지를 나타내는 어텐션 가중치(attention_weights)가 됩니다.
        # shape은 여전히 [B, num_heads, S, S]입니다.
        attention_weights = tf.nn.softmax(scaled_logits, axis=-1)

        # 학습(training) 모드일 경우, 어텐션 가중치에 dropout을 적용하여 오버피팅(overfitting)을 방지합니다.
        # self.dropout은 생성자에서 설정한 dropout 레이어로, 지정된 dropout 비율에 따라 무작위로 일부 가중치를 0으로 만듭니다.
        # 이 과정은 모델의 일반화 성능을 향상시킵니다.
        attention_weights = (
            self.dropout(attention_weights, training=training)
            if training
            else attention_weights
        )

        # 최종 어텐션 가중치(attention_weights)와 value 텐서를 행렬 곱셈(tf.matmul)하여 최종 어텐션 출력을 계산합니다.
        # 이 연산은 각 query 토큰에 대해, key와의 유사도에 따라 value들의 가중 평균(weighted sum)을 계산하는 것과 같습니다.
        # 최종 출력의 shape는 [B, num_heads, S, head_dim]가 됩니다.
        return tf.matmul(attention_weights, value)

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

        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        # 1. 입력 정규화
        x = self.attnNorm(inputs)

        # 2. QKV 프로젝션: [B, S, 3*d_model]
        qkv = self.wqkv(x)

        # 3. reshape: [B, S, 3, num_heads, head_dim]
        # 위에서 얻은 qkv 텐서의 shape는 [B, S, 3*d_model]인데,
        # 이를 [B, S, 3, num_heads, head_dim]로 reshape합니다.
        # 여기서, d_model = num_heads * head_dim이므로,
        # 이 reshape를 통해 Query, Key, Value 각각을 분리하고, 동시에 여러 헤드로 나누기 위한 준비를 합니다.
        qkv = tf.reshape(qkv, [batch_size, seq_len, 3, self.num_heads, self.head_dim])

        # 4. transpose: [3, B, num_heads, S, head_dim]
        # tf.transpose를 사용해 텐서의 차원을 재배열합니다.
        # 여기서는 첫 번째 차원(인덱스 2)이 3개의 Q, K, V를 구분하는 차원으로 오게 하여,
        # 나중에 tf.unstack을 통해 쉽게 분리할 수 있도록 합니다.
        # 결과적으로, 텐서의 shape는 [3, B, num_heads, S, head_dim]가 됩니다.
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])

        # 5. q, k, v 분리: 각각 [B, num_heads, S, head_dim]
        # tf.unstack을 사용하여, 첫 번째 차원(크기 3)을 따라 텐서를 분리합니다.
        # 이로써 각각의 q, k, v 텐서의 shape는 [B, num_heads, S, head_dim]가 됩니다.
        # 이제 각 헤드별로 attention 연산을 수행할 수 있는 준비가 완료됩니다.
        q, k, v = tf.unstack(qkv, axis=0)

        # 6. rope_embeds 처리: 만약 rope_embeds (로타리 임베딩 값)가 제공되면,
        # self.apply_rotary_pos_emb 메서드를 호출하여 Query와 Key에 로타리 임베딩을 적용합니다.
        if rope_embeds is not None:
            cos, sin = rope_embeds
            q, k = self.apply_rotary_pos_emb(q, k, cos, sin)

        # 7. mask 처리: mask가 [B, 1, S, S]인 경우, head 차원에 브로드캐스트되도록 reshape
        if mask is not None:
            mask = tf.reshape(mask, [tf.shape(mask)[0], 1, seq_len, seq_len])

        # 8. Scaled Dot-Product Attention 적용: 결과 shape [B, num_heads, S, head_dim]
        attention_output = self.scaled_dot_product_attention(
            q, k, v, mask=mask, training=training
        )

        # 9. transpose: [B, S, num_heads, head_dim]
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])

        # 10. reshape: [B, S, d_model]
        attention_output = tf.reshape(
            attention_output, [batch_size, seq_len, self.d_model]
        )

        # 11. 최종 출력 프로젝션
        output = self.o(attention_output)
        if training:
            output = self.dropout(output, training=training)
        return output


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
