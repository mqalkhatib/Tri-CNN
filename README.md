# Tri-CNN
This is an implementation of "Tri-CNN: A Three Branch Model for Hyperspectral Image Classification"

# Datasets
In our experiments, two of the most commonly used HSI datasets are adopted, namely, Pavia University and Salinas. Additionally, the Gulfport of Mississippi dataset is also used as well, although that it has not been widely used for HSI classification tasks, it is of great interest as it is small in size and consists of 72 spectral bands only.

# Requirements
python 3.8, Tensorflow 2.4.0

# Results
To quantitatively measure the proposed Tri-CNN model, three evaluation metrics are employed to verify the effectiveness of the algorithm, including Overall Accuracy (OA), Average Accuracy (AA) and Cohen's Kappa (k).
![image](https://user-images.githubusercontent.com/49251659/212429883-7f80d5e0-3b14-4733-b444-6201c9178de1.png)

Model was qualitatively evaluated by visually comparing the resulting class maps.
![image](https://user-images.githubusercontent.com/49251659/212430170-9cf311ba-967c-4105-86d6-59ee3a6c1268.png)

# Citation
@article{alkhatib2023tri,
  title={Tri-CNN: a three branch model for hyperspectral image classification},
  author={Alkhatib, Mohammed Q and Al-Saad, Mina and Aburaed, Nour and Almansoori, Saeed and Zabalza, Jaime and Marshall, Stephen and Al-Ahmad, Hussain},
  journal={Remote Sensing},
  volume={15},
  number={2},
  pages={316},
  year={2023},
  publisher={Multidisciplinary Digital Publishing Institute}
}


If you has any questions, please send e-mail to mqalkhatib@ieee.org
