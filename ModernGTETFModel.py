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

        raise NotImplementedError()

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

        # QKV projection. [B, S, 3 * d_model] shape
        qkv = self.wqkv(inputs=x)

        # 최종 출력 프로젝션
        output = self.o(inputs=qkv)

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
