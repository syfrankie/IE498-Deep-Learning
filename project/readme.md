# Implementation of Cycle GAN
## Preparation 
Download the datasets from: https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/

Unzip and put into a directory called `dataset` \
Create two directories called `train` and `test` in the unzipped file\
Rename `trainA` & `trainB` into `X` & `Y` and put them into `train` \
Rename `testA` & `testB` into `X` & `Y` and put them into `test`

## Training
In `trn.py`: 
1. Change the directory name into the desired dataset in the `ImageDataset` function
2. Adjust the hyperparameters for the desired dataset(e.g.: `lr`, `num_epochs`, `lambda_cycle`, `lambda_iden`)
3. Choose the number of residual blocks for the generator with `num_blocks=9` in `Generator`
4. Run the code

## Testing
In `tst.py`:
1. Change the directory name into the desired dataset in the `ImageDataset` function
2. Change the number of residual blocks to be the same as the number in Training part with `num_blocks=9` in `Generator`
3. Create directories `X` & `Y` in order to store the testing results if they are not available in the `output` directory:
    ```
    os.makedirs('output/X')
    os.makedirs('output/Y')
4. Run the code