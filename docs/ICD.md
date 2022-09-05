# ICD:Incremental Cognitive Diagnosis for Intelligent Education 
This is our implementation for the paper:

Shiwei Tong,Jiayu Liu ,Yuting Hong, Zhenya Huang, Le Wu, Qi Liu, Wei Huang, Enhong Chen, Dan Zhang. Incremental Cognitive Diagnosis for Intelligent Education . The 28th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD'2022)

Please cite our KDD'2022 paper if you use our codes. Thanks!

Author: Shiwei Tong

Email: tongsw@mail.ustc.edu.cn



## Example to run the codes.
The instruction of commands and take a0910 dataset as an example

Go to the code directory:
```
cd EduCDM/EduCDM/ICD/ICD
```
Replace path_prefix by your project_url in ICD/constant.py.

Run baseline
```
python project_url/ICD/Base/pure_stream_run.py --dataset a0910 --cdm mirt --ctx cuda:2    --savename global --inc_type global
```

Run incremental method
```
python project_url/ICD/ICD/pure_stream_inc_run.py --dataset a0910 --cdm ncd --ctx cuda:3 --savename icd_v0 --alpha 0.2 --beta 0.9 --tolerance 0.2 --inner_metrics True --warmup_ratio 0
```

## Citation
```bibtex
@inproceedings{tong2022incremental,
  title={Incremental Cognitive Diagnosis for Intelligent Education},
  author={Tong, Shiwei and Liu, Jiayu and Hong, Yuting and Huang, Zhenya and Wu, Le and Liu, Qi and Huang, Wei and Chen, Enhong and Zhang, Dan},
  booktitle={Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={1760--1770},
  year={2022}
}
```