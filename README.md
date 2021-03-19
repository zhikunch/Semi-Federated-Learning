# Semi-Federated-Learning
An implementation for paper "https://arxiv.org/pdf/2003.12795.pdf".
1. cl_main.py centralized machine learning from a whole dataset
2. fl_main.py conventional federated learning 
3. sfl_main.py implement the semi-federated learning
# How to Run
to run sfl_main.py, u can use the following command in the terminal

"python sfl_main.py --dataset mnist --num_channels 1 --model cnn --gpu 0 --iid --lr 0.001"
