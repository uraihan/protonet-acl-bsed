# Few-Shot Learning for Bioacoustics Sound Event Detection 
This repo focuses on exploring a Meta Learning approach to perform Bioacoustics Sound Event Detection task. The training procedure is inspired by a paper from Nogueira, et al. (2024) "Prototypical Contrastive Network for Imbalanced Aerial Image Segmentation". It consists of using a Prototypical Network to learn initial prototypes and refining it further with a contrastive loss, in this case using the Supervised Angular Contrastive Loss (Supervised ACL) proposed by Wang, et al. (2023) which I explored in my master's thesis. For the prototypical network, I took the baseline model of DCASE 2024 Challenge Task 4 as an inspiration.

The purpose of this repo is to:
- Compare performance between meta learning (specifically PN) and transfer learning
trained with Supervised ACL.
- Further explore ways to improve Supervised ACL based on recommendations from this
paper (https://proceedings.mlr.press/v119/wang20k.html)

Secondary objective is to build a software solution based on this model. (which
I'm not sure when it will happen)
