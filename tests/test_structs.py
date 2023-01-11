from matplotlib.patches import Circle
from numpy import tri
from sleapyfaces.utils.structs import CustomColumn, File, FileConstructor
import pandas as pd


def test_custom_columns():
    cc = CustomColumn("Mouse", "CSE008")
    assert cc.ColumnTitle == "Mouse"
    assert cc.ColumnData == "CSE008"
    cc.buildColumn(10)
    assert type(cc.Column) is pd.DataFrame


def test_files_constructor():
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
    assert (
        daq_file.file
        == "/Users/annieehler/Projects/Jupyter_Notebooks/SLEAPyFaces/tests/data/20211105/DAQOutput.csv"
    )
    assert (
        sleap_file.file
        == "/Users/annieehler/Projects/Jupyter_Notebooks/SLEAPyFaces/tests/data/20211105/SLEAP.h5"
    )
    fc = FileConstructor(daq_file, sleap_file, beh_file, video_file)
    assert fc.daq == daq_file
    assert fc.sleap == sleap_file
