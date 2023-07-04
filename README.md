
# Intra-Source Style Augmentation for Improved Domain Generalization (ISSA)
Official PyTorch implementation of the WACV 2023 paper ["Intra-Source Style Augmentation for Improved Domain Generalization"](https://arxiv.org/pdf/2210.10175.pdf).  This repository provides the minimal code snippets of the masked noise encoder for GAN inversion. 

**:fire: Updates** We extended our WACV paper and add more applications, e.g., utilzing stylized data for assessing domain generalization performance. Please check our new extension ["Intra- & Extra-Source Exemplar-Based Style Synthesis for Improved Domain Generalization"](https://arxiv.org/pdf/2307.00648.pdf)!

[![arXiv](https://img.shields.io/badge/arXiv-2210.10175-red)](https://arxiv.org/pdf/2210.10175.pdf)  [![arXiv](https://img.shields.io/badge/arXiv-2307.00648-blue)](https://arxiv.org/pdf/2307.00648.pdf)    


![overview](figs/overview.png)   
![teaser](figs/teaser.png)   
![extra-style](figs/extra.png)   


## Getting Started
The code is tested for Python 3.9.
ISSA conda environment can be created via
```
conda env create --file environment.yml
source activate issa
```
**Note:** please read [how-to.pdf](how-to.pdf) for more detailed instruction. 

## Citation
If you use this code please cite

```
@inproceedings{li2023intra,
  title={Intra-Source Style Augmentation for Improved Domain Generalization},
  author={Li, Yumeng and Zhang, Dan and Keuper, Margret and Khoreva, Anna},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={509--519},
  year={2023}
}
```



## License

This project is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in this project, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).


## Purpose of the project

This software is a research prototype, solely developed for and published as
part of the publication cited above. It will neither be
maintained nor monitored in any way.


## Contact
Please feel free to open an issue or contact personally if you have questions, need help, or need explanations. Don't hesitate to write an email to the following email address:
liyumeng07@outlook.com  
