# ICA Experiments
Audio unmixing and source separation using ICA<br/>

## Introduction of each file:<br/>
main.py: work flow of  experiments<br/>
FastICA.py: all implements of ICA algorithms(negentropy_fastICA, Infomax_nature_ICA, kurtosis_ICA)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;method of de-correlation of data and PCA white of data<br/>
dataProcessing.py: method of load data, show data, save results.<br/>
data file: store all data<br/>
result file: store result<br/>

## Run
run on terminal for speech dataset: python main.py --dataset=Speech --ICA_model=All --eval=True<br/>
run on terminal for ICA mix dataset:python main.py --dataset=ICA_mix --ICA_model=All --eval=True<br/>
other candidate arguments show in main.py line 46-64<br/>

## Experiments environments
python 3.8.5 (at least >3.6)<br/>
    package argparse <br/>
    package numpy <br/>
    package sklearn (only use for test build-in fastica) <br/>
    package wave <br/>
    package struct <br/>
    package matplotlib <br/>