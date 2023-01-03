# SLEAPyFaces

----

*A package for extracting facial expressions from SLEAP analyses with sensible assumptions.*

Based on [these](https://github.com/annie444/Facial-Expression-Analyses) scripts

[![PyPI - Version](https://img.shields.io/pypi/v/sleapyfaces.svg)](https://pypi.org/project/sleapyfaces)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sleapyfaces.svg)](https://pypi.org/project/sleapyfaces)
[![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)

## Description

Sleapyfaces is a data analysis package for extracting facial expressions of mice from SLEAP analyses. It is designed to work with the SLEAP software package, which provides a graphical user interface for annotating animal behavioral videos. More information on SLEAP is available at [https://sleap.ai](https://sleap.ai). This package also depends on many assumptions about the data format and structure of the SLEAP analyses. It is not intended to be a general tool for extracting facial expressions from SLEAP analyses, but rather a tool for extracting facial expressions from the specific data format and structure used in the lab of Dr. Kay Tye at The Salk Institute for Biological Studies.

----

## Table of Contents

- [SLEAPyFaces](#sleapyfaces)
	- [Description](#description)
	- [Table of Contents](#table-of-contents)
	- [Pages](#pages)
		- [Citing:](#citing)
		- [License](#license)

## Pages
- [Getting Started](gettingstarted.md)
- [Project Structure](projectstructure.md)
- [API Reference](api.md)
- [Change Log](changelog.md)

----

### Citing:

If you use SLEAPyFaces in your research, this does not fall under *standard software* according to the *Publication Manual* for the APA, MLA, AMA, Turabian, IEEE, Vancouver style, Harvard style, or Chicago style guides. **Please cite the following:**

> A. Ehler,  J. Delahantey, A. Coley, D. LeDuke, L. Keyes, T.D. Pereira, and K. Tye. SLEAPyFaces: A package for extracting facial expressions from SLEAP analyses with sensible assumptions. SLEAPyFaces python package, v1.0.1, 2023. Retrieved from [https://github.com/annie444/sleapyfaces/](https://github.com/annie444/sleapyfaces/)

**BibTeX:**
```bibtex
 @misc{Ehler2023sleapyfaces,
    title={SLEAPyFaces: A package for extracting facial expressions from SLEAP analyses with sensible assumptions.},
    author={
        Ehler, Analetta and
        Delahantey, Jeremey and
        Coley, Austin and
        LeDuke, Deryn and
        Keyes, Laurel and
        Pereira, Talmo D and
        Tye, Kay},
    url={https://github.com/annie444/sleapyfaces/},
    journal={SLEAPyFaces python package},
    publisher={GitHub repository},
    volume={v1.0.1},
    year={2023},
    month={Dec},
    day={27}
}
```

**Please also cite the original SLEAP paper:**

>T.D. Pereira, N. Tabris, A. Matsliah, D. M. Turner, J. Li, S. Ravindranath, E. S. Papadoyannis, E. Normand, D. S. Deutsch, Z. Y. Wang, G. C. McKenzie-Smith, C. C. Mitelut, M. D. Castro, J. D’Uva, M. Kislin, D. H. Sanes, S. D. Kocher, S. S-H, A. L. Falkner, J. W. Shaevitz, and M. Murthy. Sleap: A deep learning system for multi-animal pose tracking. Nature Methods, 19(4), 2022

**BibTeX:**
```bibtex
@ARTICLE{Pereira2022sleap,
   title={SLEAP: A deep learning system for multi-animal pose tracking},
   author={Pereira, Talmo D and
      Tabris, Nathaniel and
      Matsliah, Arie and
      Turner, David M and
      Li, Junyu and
      Ravindranath, Shruthi and
      Papadoyannis, Eleni S and
      Normand, Edna and
      Deutsch, David S and
      Wang, Z. Yan and
      McKenzie-Smith, Grace C and
      Mitelut, Catalin C and
      Castro, Marielisa Diez and
      D'Uva, John and
      Kislin, Mikhail and
      Sanes, Dan H and
      Kocher, Sarah D and
      Samuel S-H and
      Falkner, Annegret L and
      Shaevitz, Joshua W and
      Murthy, Mala},
   journal={Nature Methods},
   volume={19},
   number={4},
   year={2022},
   publisher={Nature Publishing Group}
   }
}
```

### License

`sleapyfaces` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license. The full license text is available in the [LICENSE](LICENSE) file.
