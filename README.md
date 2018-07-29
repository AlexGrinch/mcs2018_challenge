# mcs2018_challenge
Third place solution for [Vision Labs Adversarial Attacks on Black Box Face Recognition System Challenge](https://competitions.codalab.org/competitions/19090).

Team Boyara Power (Oleksii Hrinchuk, Valentin Khrulkov, Elena Orlova)

# Solution description

Our solution consists of two parts. At first, we trained copycat network to imitate outputs of the Black Box. Then we attacked this substitute network with the standard White Box targeted attacks algorithm to get perturbed images and fool the original Black Box using them.

### Training copycat network
1. Take pretrained FaceNet based on Inception v1 architecture.
2. Replace all fully connected layers with one FC layer of 512 neurons, followed by BatchNorm and L2 normalization.
3. Finetune the obtained network by training it for 10 epochs with learning rate decay after every 3 epochs.
4. Train three such networks and combine them into ensemble (by averaging the resulting descriptors).

For better copycat network we used a number of data augmentation techniques, such as:
1. Augment the data with 4 possible corner crops, and by performing horizontal flip, zoom and shift.
2. Extend training set with previously computed submissions to better approximate the network in the proximity of given data.
3. Generate synthetic inputs based on identifying directions in which the models’ output is varying the most.

### Attacking copycat network
1. Best attacking algorithm in our experiments was targeted fast gradient method (FGM) accelerated with Nesterov momentum.
2. We also noticed that 998 out of 1000 people from target images were exactly the people from source images (which can be found by analysing pairwise L2 distances between source and target images). Instead of attacking just 5 given target images we were attacking 20 (original targets, corresponding sources, mirror reflections of both) to get more robust and generalizable attacker.

### Ablative study
| Model | Public score |
|-|-|
| Finetuned Facenet | 1.256 |
| + train augmentation | 1.114 |
| + ensemble & Nesterov | 1.007 |
| + BatchNorm | 0.981 |
