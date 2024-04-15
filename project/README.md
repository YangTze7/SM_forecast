Cuda version: cuda11.3


Torch package version:
torch                     1.13.0+cu116
torchaudio                0.13.0+cu116
torchvision               0.14.0+cu116


Project file structure:
├── README.md
├── code
│   ├── config.py # model configure file
│   ├── checkpoint099.pth # saved model checkpoint
│   └── main.py # main file for execution
├── data
│   └── test # input file path
└── submit
    └── output # output result path
	
895109Rivers
895109rsidd

{
    "registry-mirrors": [
        "https://g5p25cab.mirror.aliyuncs.com"
    ],
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    }
}