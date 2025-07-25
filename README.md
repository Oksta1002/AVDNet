# AVDNet
### **[IEEE Signal Processing Letters, 2025] Adaptive Video Demoiréing Network with Subtraction-Guided Alignment**
Seung-Hun Ok, Young-Min Choi, Seung-Wook Kim, Se-Ho Lee

[Paper](https://doi.org/10.1109/LSP.2025.3585820) | [Supplementary Materials](https://drive.google.com/file/d/1Bk-R0x-ACmo8sU7rr86cbPRTgkrHsTy5/view?usp=drive_link)

###  Environment
***
The experiments were conducted using the following software environment:
* PyTorch: 

### Dataset
***
The dataset used in our project can be downloaded from the following link:
- **VDmoire**: [GitHub](https://github.com/CVMI-Lab/VideoDemoireing)

After downloading the dataset, please place the folders as follows:
```
project_root/
    ├── avdnet/
    │   ├── experiments/
    │   │   ...
    │   └── train.py
    └── datasets/
        ├── homo/
        └── optical/
            ├── iphone/
            └── tcl/
```

### Pretrained Models
***
You can download the pretrained models form the following links:
- **VDmoire**: [dropbox](https://www.dropbox.com/scl/fo/t8w8gfd1hz3m445z3tso2/ACHqeKW5iXnhLSaLlJXCMsY?rlkey=183mti7url38nvrt6gfkaajr5&st=fs9lzkqu&dl=0)
> **Note**: During testing, make sure to set `strict_load: false` in your option file to avoid mismatch errors when loading the pretrained weights.

### Citation
***
If you use this code or the results in your research, please cite the following paper:
```bibtex
@article{ok2025adaptive,
title = {Adaptive Video Demoiréing Network With Subtraction-Guided Alignment},
author = {Ok, Seung-Hun and Choi, Young-Min and Kim, Seung-Wook and Lee, Se-Ho},
journal = {IEEE Signal Processing Letters},
volume = {32},
pages = {2733--2737},
year = {2025}
}
```

### Contact
***
For any questions, please contact: cornking123@jbnu.ac.kr.