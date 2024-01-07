# DeepMRI-Denoise-vs-BM3D
This study involves the implementation and comparison of three distinct deep learning-based denoising models for MRI under different model parameters. The objective is to assist authors proposing superior MRI denoising algorithms by facilitating a comparative reimplementation against conventional methods. Please note that this project is solely intended for learning purposes.

Implemented denoising algorithms include NAFNet model from M4Raw, banding removal Unet Model from fastMRI, Unetwavelet model from DlDegibbs, and conventional BM3D denoising model. The utilized dataset is the publicly available M4RawV1.5 dataset, with the training subset containing 1024 samples. The original validation set (240 samples) has been partitioned into a validation set (112 samples) and a test set (128 samples), maintaining a balanced training set: validation set: test set ratio of 8:1:1. 

Denoising efficacy is measured using PSNR and SSIM metrics. Each algorithm undergoes the following evaluations:
* Systematic processing of the M4Raw test set (formatted in H5), providing mean and standard deviation values for PSNR and SSIM.
* Preservation of denoised effects for individual images (formatted in DICOM, without labels), followed by the calculation of aggregate processing time.

These evaluations aim to provide a comprehensive assessment of the practical effectiveness of deep learning models in the context of MRI denoising.

# Documentation and Reproducibility 
* **The M4Raw dataset and its corresponding NAFNet model**
  * [M4Raw: A multi-contrast, multi-repetition, multi-channel MRI k-space dataset for low-field MRI research](https://doi.org/10.1038/s41597-023-02181-4)
  * All of the M4Raw data can be downloaded from the [M4Raw dataset page](https://zenodo.org/records/8056074).
  * Detailed information about the M4Raw can be found at [the GitHub repo](https://github.com/mylyu/M4Raw).


* **fastMRI banding_removal**
  * [MRI Banding Removal via Adversarial Training](https://api.semanticscholar.org/CorpusID:210861018)
  * This project includes the `banding_removal` module from the [Facebook Research - fastMRI](https://github.com/facebookresearch/fastMRI/tree/main/banding_removal) project.


* **DlDegibbs Unetwavelet**
  * [Training a Neural Network for Gibbs and Noise Removal in Diffusion MRI](https://api.semanticscholar.org/CorpusID:150373937)
  * The code for this whitepaper is in the [/mmuckley/dldegibbs](https://github.com/mmuckley/dldegibbs) GitHub repo.


# Brief

Here is a brief overview of the project components:

- **M4Raw_tutorial.ipynb:** This notebook shows how to read the M4Raw dataset and apply some simple transformations to the data.
- **BM3D_process_all.ipynb:** This notebook demonstrates how to batch process the M4Raw test set using BM3D, and output the corresponding mean and standard deviation of PSNR and SSIM.
- **dicom_test.ipynb:** This notebook demonstrates how to process a new type of MRI image data using various algorithms and save the denoised results in well-formatted image layouts. Please note that the dataset utilized in this study comprises hospital MRI scan images in DICOM file format. Each patient's scan results are available in four modes: FLAIR-Axial, T1WI-Axial, T1WI-Sagittal, and T2WI-Axial. You can make corresponding modifications based on your own dataset directory.
  * The well-trained models mentioned above, along with the reorganized validation and test sets of M4RawV1.5, have been placed in a separate repository. If you're looking to get started quickly, feel free to visit [here](https://github.com/zychen029/MRI_Denoising_Models_Weights).

# Cite

If you find this code useful, please consider citing the original paper:

### M4Raw
[1] Lyu, M., Mei, L., Huang, S. et al. M4Raw: A multi-contrast, multi-repetition, multi-channel MRI k-space dataset for low-field MRI research. Sci Data 10, 264 (2023). https://doi.org/10.1038/s41597-023-02181-4
```latex
@article{lyu_m4raw_2023,
 title = {{M4Raw}: {A} multi-contrast, multi-repetition, multi-channel {MRI} k-space dataset for low-field {MRI} research},
 volume = {10},
 issn = {2052-4463},
 url = {https://doi.org/10.1038/s41597-023-02181-4},
 doi = {10.1038/s41597-023-02181-4},
 abstract = {Recently, low-field magnetic resonance imaging (MRI) has gained renewed interest to promote MRI accessibility and affordability worldwide. The presented M4Raw dataset aims to facilitate methodology development and reproducible research in this field. The dataset comprises multi-channel brain k-space data collected from 183 healthy volunteers using a 0.3 Tesla whole-body MRI system, and includes T1-weighted, T2-weighted, and fluid attenuated inversion recovery (FLAIR) images with in-plane resolution of {\textasciitilde}1.2 mm and through-plane resolution of 5 mm. Importantly, each contrast contains multiple repetitions, which can be used individually or to form multi-repetition averaged images. After excluding motion-corrupted data, the partitioned training and validation subsets contain 1024 and 240 volumes, respectively. To demonstrate the potential utility of this dataset, we trained deep learning models for image denoising and parallel imaging tasks and compared their performance with traditional reconstruction methods. This M4Raw dataset will be valuable for the development of advanced data-driven methods specifically for low-field MRI. It can also serve as a benchmark dataset for general MRI reconstruction algorithms.},
 number = {1},
 journal = {Scientific Data},
 author = {Lyu, Mengye and Mei, Lifeng and Huang, Shoujin and Liu, Sixing and Li, Yi and Yang, Kexin and Liu, Yilong and Dong, Yu and Dong, Linzheng and Wu, Ed X.},
 month = may,
 year = {2023},
 pages = {264},
}
```

### fastMRI banding_removal
[2] Defazio A, Murrell T, Recht M P. MRI Banding Removal via Adversarial Training[J]. 2020.D-OI:10.48550/arXiv.2001.08699.
```latex
@misc{zbontar2018fastMRI,
    title={{fastMRI}: An Open Dataset and Benchmarks for Accelerated {MRI}},
    author={Jure Zbontar and Florian Knoll and Anuroop Sriram and Tullie Murrell and Zhengnan Huang and Matthew J. Muckley and Aaron Defazio and Ruben Stern and Patricia Johnson and Mary Bruno and Marc Parente and Krzysztof J. Geras and Joe Katsnelson and Hersh Chandarana and Zizhao Zhang and Michal Drozdzal and Adriana Romero and Michael Rabbat and Pascal Vincent and Nafissa Yakubova and James Pinkerton and Duo Wang and Erich Owens and C. Lawrence Zitnick and Michael P. Recht and Daniel K. Sodickson and Yvonne W. Lui},
    journal = {ArXiv e-prints},
    archivePrefix = "arXiv",
    eprint = {1811.08839},
    year={2018}
}
```

### DlDegibbs Unetwavelet
[3] Muckley M J , Ades-Aron B , Papaioannou A ,et al.Training a Neural Network for Gibbs and Noise Removal in Diffusion MRI[J]. 2019.DOI:10.1002/mrm.28395.
```latex
@article{muckley2019training,
  title={Training a Neural Network for Gibbs and Noise Removal in Diffusion MRI},
  author={Muckley, Matthew J and Ades-Aron, Benjamin and Papaioannou, Antonios and Lemberskiy, Gregory and Solomon, Eddy and Lui, Yvonne W and Sodickson, Daniel K and Fieremans, Els and Novikov, Dmitry S and Knoll, Florian},
  journal={arXiv preprint arXiv:1905.04176},
  year={2019}
}
```

Please make sure to review and comply with the terms of the M4Raw, fastMRI and DlDegibbs project's license when using this code.


# Acknowledgements
Special thanks to the contributors of the original code libraries used in this project. Your work has been invaluable, and I appreciate the collaborative efforts of the entire community.
