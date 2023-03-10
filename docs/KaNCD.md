# KaNCD

The implementation of the KaNCD model in paper: [NeuralCD: A General Framework for Cognitive Diagnosis](https://ieeexplore.ieee.org/abstract/document/9865139)

KaNCD is an **K**nowledge-**a**ssociation based extension of the **N**eural**CD**M (alias NCDM in this package) model. In KaNCD, higher-order low dimensional latent traits of students, exercises and knowledge concepts are used respectively. 

The knowledge difficulty vector of an exercise is calculated from the latent trait of the exercise and the latent trait of each knowledge concept. 

![KDM_MF](F:\git_project\EduCDM\EduCDM\docs\_static\KDM_MF.png)

Similarly, the knowledge proficiency vector of a student is calculated from the latent trait of the student and the latent trait of each knowledge concept.

![KPM_MF](F:\git_project\EduCDM\EduCDM\docs\_static\KPM_MF.png)

Please refer to the paper for more details.