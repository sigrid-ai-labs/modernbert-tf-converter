import tensorflow as tf


def create_local_sliding_window_mask(
    global_mask_4d: tf.Tensor, window_size: int
) -> tf.Tensor:
    """
    ModernBERT 의 local attention 을 구현하기 위하여 local sliding window mask 를 만들어봅시다.

    :param global_mask_4d: (batch_size, 1, seq_len, seq_len) 을 가지는 4d tensor 입니다.
    :param window_size: 128 토큰 사이즈 수준으로 확인하는 local attention window 의 크기 입니다.
    window_size // 2 에 존재하는 Position은 valid하다고 처리 합니다.
    :return: final_local_4d (tf.Tensor): (batch_size, 1, seq_len, seq_len) 를 가지는데, 다음의 형상을 지닙니다.
        - sliding window 내부에 있고 global mask에 valid 하다면 0.0이다.
        - sliding window 외부에 있다면 padding이 되므로 -∞ 이다.
    """

    pass
