import numpy as np

from slassl.cli.visualize_events import render_events


def test_render_events_uses_distinct_polarity_colors() -> None:
    image = render_events(
        x=np.array([1, 2]),
        y=np.array([1, 2]),
        polarity=np.array([0, 1]),
        height=4,
        width=4,
    )
    assert image.shape == (4, 4, 3)
    assert image[1, 1, 0] > image[1, 1, 2]
    assert image[2, 2, 2] > image[2, 2, 0]
