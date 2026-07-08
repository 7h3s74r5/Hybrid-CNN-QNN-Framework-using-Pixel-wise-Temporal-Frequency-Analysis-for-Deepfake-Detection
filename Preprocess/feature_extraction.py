from scipy.fft import fft

def generate_pwtf(frames):

    """
    frames shape

    (32,128,128)

    """

    fft_volume = fft(
        frames,
        axis=0
    )

    magnitude = np.abs(
        fft_volume
    )

    phase = np.angle(
        fft_volume
    )

    magnitude = magnitude.mean(
        axis=0
    )

    phase = phase.std(
        axis=0
    )

    magnitude = (
        magnitude -
        magnitude.min()
    ) / (
        magnitude.max()
        -
        magnitude.min()
        + 1e-4
    )

    phase = (
        phase -
        phase.min()
    ) / (
        phase.max()
        -
        phase.min()
        + 1e-8
    )

    return np.stack(
        [magnitude, phase]
    ).astype(np.float32)
