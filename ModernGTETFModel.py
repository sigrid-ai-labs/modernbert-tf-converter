import tensorflow as tf


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
