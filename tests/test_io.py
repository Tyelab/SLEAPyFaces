from sleapyfaces.files import DAQData, SLEAPanalysis, BehMetadata, VideoMetadata


def test_daq_initialization():
    daq = DAQData(
        "/Users/annieehler/Projects/Jupyter_Notebooks/SLEAPyFaces/tests/data/20211105/DAQOutput.csv"
    )
    assert (
        daq.path
        == "/Users/annieehler/Projects/Jupyter_Notebooks/SLEAPyFaces/tests/data/20211105/DAQOutput.csv"
    )
    assert daq.cache.shape[1] == 13
    assert daq.cache.shape[0] > 13
    assert daq.columns == [
        "Shutter_on",
        "Shutter_off",
        "LED590_on",
        "LED590_off",
        "Lick_on",
        "Lick_off",
        "Speaker_on",
        "Speaker_off",
        "Airpuff_on",
        "Airpuff_off",
        "Sucrose_on",
        "Sucrose_off",
    ]


def test_sleap_initialization():
    sleap = SLEAPanalysis(
        "/Users/annieehler/Projects/Jupyter_Notebooks/SLEAPyFaces/tests/data/20211105/SLEAP.h5"
    )
    assert (
        sleap.path
        == "/Users/annieehler/Projects/Jupyter_Notebooks/SLEAPyFaces/tests/data/20211105/SLEAP.h5"
    )
    names = [0] * len(sleap.data["node_names"]) * 2
    for name, i in zip(
        sleap.data["node_names"], range(0, (len(sleap.data["node_names"]) * 2), 2)
    ):
        names[i] = f"{name.replace(' ', '_')}_x"
        names[i + 1] = f"{name.replace(' ', '_')}_y"
    assert len(sleap.data.keys()) > 4
    assert len(sleap.tracks.columns) == (len(sleap.data["node_names"]) * 2)
    assert sleap.track_names == names


def test_beh_initialization():
    beh = BehMetadata(
        "/Users/annieehler/Projects/Jupyter_Notebooks/SLEAPyFaces/tests/data/20211105/BehMetadata.json"
    )

    assert beh.ITIArrayKey == "ITIArray"
    assert beh.TrialArrayKey == "trialArray"
    assert beh.MetaDataKey == "beh_metadata"
    assert (
        beh.path
        == "/Users/annieehler/Projects/Jupyter_Notebooks/SLEAPyFaces/tests/data/20211105/BehMetadata.json"
    )
    assert beh.columns == ["trialArray", "ITIArray"]
    assert beh.cache.shape[1] == 2
    assert beh.cache.shape[0] > 2


def test_video_initialization():
    video = VideoMetadata(
        "/Users/annieehler/Projects/Jupyter_Notebooks/SLEAPyFaces/tests/data/20211105/video.mp4"
    )

    assert (
        video.path
        == "/Users/annieehler/Projects/Jupyter_Notebooks/SLEAPyFaces/tests/data/20211105/video.mp4"
    )

    assert type(video.cache) is dict
    assert len(video.cache.keys()) > 3
    assert eval(video.cache["avg_frame_rate"]) == video.fps
