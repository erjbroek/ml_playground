This is just a playground for myself to experiment and compare different libraries for deeplearning, and finding out how to train the models on the gpu. To make sure the models work, i will be using mnist. Later, the plan is to actually use the models on complexer datasets like [CIFAR-10](https://en.wikipedia.org/wiki/CIFAR-10)


## mnist dataset
| Library    | Model | Accuracy | Loss  | Runtime | GPU Access |
|------------|-------|----------|-------|---------|------------|
| Tensorflow | cnn   | 99.22%   | 0.026 | 299.8s  | <span style="color:red">False</span>      |
| Tensorflow | mlp   | 98.17%   | 0.062 | 37.5s   | <span style="color:red">False</span>   
|-|||||   |
| Pytorch    | cnn   | 99.02%   | 0.0323 | 344.5s  | <span style="color:red">False</span>      |
| Pytorch    | cnn   | 99.18%   | 0.0311 | 53.6s  | <span style="color:green">True</span>      |
| Pytorch    | mlp   | 98.18%   | 0.072 | 64.1s  | <span style="color:red">False</span>      |
| Pytorch    | mlp   | 98.23%   | 0.070 | 34.2s  | <span style="color:green">True</span>      |

# Future plans:
- Test the model(s) on different datasets
- Implement image augmentation for the images for the cnn's, something like what i did for my other repository
- ![image](https://github.com/user-attachments/assets/4efdce9b-56dc-47df-9473-ad9127c61c90)

