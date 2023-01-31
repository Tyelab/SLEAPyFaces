from sleapyfaces.experiment import Experiment
from sleapyfaces.utils.structs import File, FileConstructor, CustomColumn
from sleapyfaces.files import SLEAPanalysis
import pandas as pd


def test_experiment():
    daq_file = File(
        "/Users/annieehler/Projects/Jupyter_Notebooks/SLEAPyFaces/tests/data/20211105",
        "DAQOutput.csv",
    )
    sleap_file = File(
        "/Users/annieehler/Projects/Jupyter_Notebooks/SLEAPyFaces/tests/data/20211105",
        "*.h5",
        True,
    )
    beh_file = File(
        "/Users/annieehler/Projects/Jupyter_Notebooks/SLEAPyFaces/tests/data/20211105",
        "BehMetadata.json",
    )
    video_file = File(
        "/Users/annieehler/Projects/Jupyter_Notebooks/SLEAPyFaces/tests/data/20211105",
        "video.mp4",
    )
    sleap = SLEAPanalysis(sleap_file.file)
    fc = FileConstructor(daq_file, sleap_file, beh_file, video_file)
    expr = Experiment("Test", fc)
    assert expr.name == "Test"
    assert expr.files == fc
    assert expr.sleap.tracks.equals(sleap.tracks)
    assert expr.numeric_columns == sleap.track_names
    cc = [CustomColumn("Mouse", "CSE008"), CustomColumn("Date", "20211105")]
    expr.buildData(cc)
    cc[0].buildColumn(len(expr.sleap.tracks.index))
    assert len(expr.sleap.tracks["Mouse"]) == len(cc[0].Column)
    assert expr.sleap.tracks.loc[
        :,
        ["Mouse", "Date", "Timestamps", "Frames"],
    ].equals(expr.custom_columns.loc[:, ["Mouse", "Date", "Timestamps", "Frames"]])
    expr.buildTrials(["Speaker_on", "LED590_on"], [False, True])
    assert len(expr.trialData) > 1
    assert type(expr.trialData[0]) is pd.DataFrame
    assert type(expr.trials) is pd.DataFrame
