from scipy.fft import fft


def generate_pwtf(frames):

    fft_volume = fft(
        frames,
        axis=0
    )


    magnitude = np.log1p(
        np.abs(fft_volume)
    )

    phase = np.angle(
        fft_volume
    )


    
    magnitude = magnitude[1:9]

    phase = phase[1:9]


    # Aggregate low-frequency temporal bands

    magnitude = magnitude.mean(
        axis=0
    )

    phase = phase.mean(
        axis=0
    )


    # Normalize

    magnitude = (
        magnitude - magnitude.min()
    ) / (
        magnitude.max()
        - magnitude.min()
        + 1e-8
    )


    phase = (
        phase + np.pi
    ) / (
        2*np.pi
    )


    return np.stack(
        [
            magnitude,
            phase
        ]
    ).astype(np.float32)
