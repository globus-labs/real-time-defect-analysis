import numpy as np
import pandas as pd
import trackpy as tp
from pytest import fixture

from rtdefects.analysis import convert_to_per_particle, compile_void_tracks, analyze_defects
from rtdefects.drift import compute_drifts_from_tracks


@fixture
def example_segmentation():
    """A few prototypical tracks over 3 frames:
    - Stay all frames (in the same point)
    - Move at a large rate
    - Disappear
    - Appear in new frame
    """

    # Make the frames
    tracks = [
        np.array([[0, 0], [1, 1], [2, 2]]),  # All except the appearing particle,
        np.array([[1.1, 1], [0, 0], [2, 2], [3, 3]]),  # All, with two switching positions
        np.array([[0, 0], [1.2, 1], [3, 3], [4, 4]]),  # All except the disappearing one
        np.array([[0, 0], [1.3, 1], [3, 3], [4, 4], [5, 5]]),  # All except the disappearing one, add a new singlet
    ]
    frames = np.arange(4)
    radii = [np.ones((t.shape[0],)) for t in tracks]

    # Mark which ones are on the side of the frame
    on_side = [
        (x < 0.5).any(axis=1)
        for x in tracks
    ]

    # Make the dataframe
    return pd.DataFrame({
        'positions': tracks,
        'frames': frames,
        'radii': radii,
        'touches_side': on_side
    })


def test_conversion(example_segmentation):
    """Test converting to a trackpy-ready format"""
    particles = pd.concat(list(convert_to_per_particle(example_segmentation)))
    assert len(particles) == 16


def test_tracking(example_segmentation):
    # Convert then run the analysis
    tracks = pd.concat(tp.link_df_iter(convert_to_per_particle(example_segmentation), search_range=0.2))
    assert len(tracks) == 16  # One per particle
    assert tracks['particle'].max() == 5  # 6 total particles

    # Gather the information
    track_ids = [g['local_id'].tolist() for _, g in tracks.groupby('particle')]
    assert (track_ids == [
        [0, 1, 0, 0],
        [1, 0, 1, 1],
        [2, 2],
        [3, 2, 2],
        [3, 3],
        [4]
    ])

    # Compute the drifts
    drifts = compute_drifts_from_tracks(tracks)
    assert np.isclose(drifts, 0, atol=0.5).all()  # Ensure that we know the objects don't move far relative to each other

    # Compile the tracks
    track_data = compile_void_tracks(tracks)

    # Make sure void #1 is known to touch the side
    assert all(track_data.iloc[0]['touches_side'] == [True] * 4)
    assert not track_data.iloc[1:]['touches_side'].apply(any).any()


def test_touch_sides(mask):
    """Test whether we detect if voids touch the sides"""

    # Test without an edge buffer
    output = analyze_defects(mask, edge_buffer=0)
    no_buffer = sum(output['touches_side'])
    assert mask.shape[0] > no_buffer > 0  # Not all touch the side

    # Test with
    output = analyze_defects(mask, edge_buffer=128)
    assert sum(output['touches_side']) > no_buffer  # The edge buffer should catch more near the edges
