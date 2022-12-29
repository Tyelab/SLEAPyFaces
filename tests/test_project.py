from sleapyfaces.project import Project
import pandas as pd


def test_project():
    noGlob = Project(
        base="/Users/annieehler/Projects/Jupyter_Notebooks/SLEAPyFaces/tests/data",
        DAQFile="DAQOutput.csv",
        BehFile="BehMetadata.json",
        SLEAPFile="SLEAP.h5",
        VideoFile="video.mp4",
        get_glob=False,
    )

    withGlob = Project(
        base="/Users/annieehler/Projects/Jupyter_Notebooks/SLEAPyFaces/tests/data",
        iterator={"week 1": "20211105", "week 2": "20211112"},
        DAQFile="*.csv",
        BehFile="*.json",
        SLEAPFile="*.h5",
        VideoFile="*.mp4",
        get_glob=True,
    )

    assert noGlob.exprs[0].sleap.tracks.equals(withGlob.exprs[0].sleap.tracks)
    assert noGlob.exprs[0].sleap.path == noGlob.exprs[0].files.sleap.file
    assert withGlob.exprs[0].sleap.path == withGlob.exprs[0].files.sleap.file
    assert noGlob.exprs[0].sleap.path == withGlob.exprs[0].sleap.path

    noGlob.buildColumns(["Mouse"], ["CSE008"])
    withGlob.buildColumns(["Mouse"], ["CSE008"])

    assert noGlob.all_data.equals(withGlob.all_data)

    noGlob.buildTrials(["Speaker_on", "LED590_on"], [False, True])
    withGlob.buildTrials(["Speaker_on", "LED590_on"], [False, True])

    assert len(noGlob.exprs[0].trialData) == len(withGlob.exprs[0].trialData)
    assert noGlob.exprs[0].trials.equals(withGlob.exprs[0].trials)

    noGlob.meanCenter()
    noGlob.zScore()
    noGlob.visualize()

    withGlob.analyze()
    withGlob.visualize()

    assert type(noGlob.pcas["pca2d"]) is pd.DataFrame
