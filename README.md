# N-CMAPSS_DL
DL evaluation on N-CMAPSS
Turbo fan engine           |  CMAPSS [[1]](#1)
:----------------------------:|:----------------------:
![](turbo_engine.jpg)  |  ![](cmapss.png)

## Prerequisites


## Sample creator
Following the below instruction, you can create training/test sample arrays for machine learning model (especially for DL architectures that allow time-windowed data as input) from NASA's N-CMAPSS datafile. <br/>
Please download Turbofan Engine Degradation Simulation Data Set-2, so called N-CMAPSS dataset [[2]](#2), from [NASA's prognostic data repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/). In case the link does not work, please temporarily use this [shared drive](https://drive.google.com/drive/folders/1HtnDBGhMoAe53hl3t1XeCO9Z8IKd3-Q-)
<br/>
In the downloaded dataset, dataset DS01 has been used for the application of model-based diagnostics and dataset DS02 has been used for data-driven prognostics.   Therefore, we need only dataset DS02. <br/> 
Please locate "N-CMAPSS_DS02-006.h5"file to /N-CMAPSS folder. <br/>
Then, you can get npz files for each of 9 engines by running the python codes below. 
```bash
python3 sample_creator_unit_auto.py -w 50 -s 1 --test 0 --sampling 10
```
After that, you should run 
```bash
python3 sample_creator_unit_auto.py -w 50 -s 1 --test 1 --sampling 10
```
&ndash;  w : window length <br/>
&ndash;  s : stride of window <br/>
&ndash;  test : select train or test, if it is zero, then the code extracts samples from the engines used for training. Otherwise, it creates samples from test engines<br/>
&ndash;  sampling : subsampling the data before creating the output array so that we can set assume different sampling rate to mitigate memory issues. 


Please note that we used N = 6 units (u = 2, 5, 10, 16, 18 & 20) for training and M = 3  units (u = 11, 14 & 15) for test, same as for the setting used in [[3]](#3). <br/>

The size of the dataset is significantly large and it can cause memory issues by excessive memory use. Considering memory limitation that may occur when you load and create the samples, we set the data type as 'np.float32' to reduce the size of the data while the data type of the original data is 'np.float64'. Based on our experiments, this does not much affect to the performance when you use the data to train a DL network. If you want to change the type, please check 'data_preparation_unit.py' file in /utils folder.  <br/>

In addition, we offer the data subsampling to handle 'out-of-memory' issues from the given dataset that use the sampling rate of 1Hz. When you set this subsampling input as 10, then it indicates you only take only 1 sample for every 10, the sampling rate is then 0.1Hz. 

Finally, you can have 9 npz file in /N-CMAPSS/Samples_whole folder. <br/>

Each compressed file contains two arrays with different labels: 'sample' and 'label'. In the case of the test units, 'label' indicates the ground truth RUL of the test units for evaluation. 

For instance, one of the created file, Unit2_win50_str1_smp10.npz, its filename indicates that the file consists of a collection of the sliced time series by time window size 50 from the trajectory of engine (unit) 2 with the sampling rate of 0.1Hz. <br/>

## Load created samples
At first, you should load each of the npy files created in /Samples_whole folder. Then, the samples from the different engines should be aggregated. 
```bash
def load_part_array_merge (npz_units):
    sample_array_lst = []
    label_array_lst = []
    for npz_unit in npz_units:
      loaded = np.load(npz_unit)
      sample_array_lst.append(loaded['sample'])
      label_array_lst.append(loaded['label'])
    sample_array = np.dstack(sample_array_lst)
    label_array = np.concatenate(label_array_lst)
    sample_array = sample_array.transpose(2, 0, 1)
    return sample_array, label_array
```
The shape of your sample array should be (# of samples from all the units, window size, # of variables)


## References
<a id="1">[1]</a> 
Frederick, Dean & DeCastro, Jonathan & Litt, Jonathan. (2007). User's Guide for the Commercial Modular Aero-Propulsion System Simulation (C-MAPSS). NASA Technical Manuscript. 2007–215026. 

<a id="2">[2]</a> 
Chao, Manuel Arias, Chetan Kulkarni, Kai Goebel, and Olga Fink. "Aircraft Engine Run-to-Failure Dataset under Real Flight Conditions for Prognostics and Diagnostics." Data. 2021; 6(1):5. https://doi.org/10.3390/data6010005

<a id="3">[3]</a> 
Chao, Manuel Arias, Chetan Kulkarni, Kai Goebel, and Olga Fink. "Fusing physics-based and deep learning models for prognostics." Reliability Engineering & System Safety 217 (2022): 107961.

<a id="3">[4]</a> 
Mo, Hyunho, and Giovanni Iacca. "Multi-objective optimization of extreme learning machine for remaining useful life prediction." In Applications of Evolutionary Computation: 25th European Conference, EvoApplications 2022, Held as Part of EvoStar 2022, Madrid, Spain, April 20–22, 2022, Proceedings, pp. 191-206. Cham: Springer International Publishing, 2022.

Bibtex entry ready to be cited
```
@inproceedings{mo2022multi,
  title={Multi-objective optimization of extreme learning machine for remaining useful life prediction},
  author={Mo, Hyunho and Iacca, Giovanni},
  booktitle={Applications of Evolutionary Computation: 25th European Conference, EvoApplications 2022, Held as Part of EvoStar 2022, Madrid, Spain, April 20--22, 2022, Proceedings},
  pages={191--206},
  year={2022},
  organization={Springer}
}
```

