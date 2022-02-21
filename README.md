# traffic-demand-prediction
used multi tricks to predict ride-sourcing demand

Abstract- With the development of the automated mobility technology and the shared economy, ride-sourcing services have been one of the first relevant application scenario using shared autonomous vehicles. Enhancing spatio-temporal ride-sourcing demand prediction performance is really important to efficiently dispatch shared autonomous vehicles, which can reduce vehicles idle time on streets. Indeed, empty vehicles could aggravate traffic congestion and air pollution. However, the existing researches focus on single-step prediction based on grids or hexagons, which not only limits the practical application value of the model, but also presents the problem of insufficient prediction ability. To fill this gap, we propose a novel ride-sourcing demand prediction framework using an Optimized Spatiotemporal Encoder-Decoder Neural Network (O-STEDN). For the encoder layer, a combination of Graph Convolutional Network (GCN) and a Read-first Long Short-Term Memory (RLSTM) network are used to better extract the spatio-temporal features simultaneously. For the decoder layer, a LSTM model and a dynamic spatio-temporal attention mechanism are applied to adaptively associate historical spatio-temporal features with the current prediction. Furthermore, a residual connection and a layer normalization tricks are added in both the encoder layer and the decoder layer to prevent the loss of feature information and internal covariate shift problems. The experiment results show that the proposed framework outperforms state-of-the-art models whether for single step demand prediction or multi-steps prediction, which could not only help transportation network companies to improve the ride-sourcing services, but also help policy makers to take traffic demand management measures dynamically.

key-words— ride-sourcing demand prediction; encoder-decoder model; graph convolutional network; read-first long short-term memory; dynamic spatio-temporal attention mechanism


Deep learning methods, including O-STEDN and baselines, the HyperParameters setting as followings:

        self.parser.add_argument('--is_training', type=bool, default=True, help='is training')
        self.parser.add_argument('--epochs', type=int, default=600, help='epoch')
        self.parser.add_argument('--step', type=int, default=1, help='step')
        self.parser.add_argument('--batch_size', type=int, default=32, help='batch size')
        self.parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate')
        self.parser.add_argument('--dropout', type=float, default=0.5, help='drop out')
        self.parser.add_argument('--site_num', type=int, default=162, help='total number of city')
        self.parser.add_argument('--features', type=int, default=5, help='numbers of the feature')
        self.parser.add_argument('--normalize', type=bool, default=True, help='normalize')
        self.parser.add_argument('--input_length', type=int, default=8, help='input length')
        self.parser.add_argument('--output_length', type=int, default=6, help='output length')
        
        self.parser.add_argument('--model_name', type=str, default='cnn_lstm', help='model string') 
        # you can change the model name to training or testing the model,
        # when the traning and testing stage before, you should set the defualt model name, that is {default='cnn_lstm'}:
        # first step is training, and second step is testing. training stage, you should input number 1, testing stage, you should input number 1.
        # if you have any problems you count, please do not hesitate to contact me, my e-mail address is: 2010768@tongji.edu.cn.
        
        self.parser.add_argument('--hidden1', type=int, default=128, help='number of units in hidden layer 1')
        self.parser.add_argument('--gcn_output_size', type=int, default=128, help='model string')
        self.parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight for L2 loss on embedding matrix')
        self.parser.add_argument('--max_degree', type=int, default=3, help='maximum Chebyshev polynomial degree')
        self.parser.add_argument('--hidden_size', type=int, default=128, help='hidden size')
        self.parser.add_argument('--hidden_layer', type=int, default=1, help='hidden layer')
        self.parser.add_argument('--training_set_rate', type=float, default=0.70, help='training set rate')
        self.parser.add_argument('--validate_set_rate', type=float, default=0.15, help='validate set rate')
        self.parser.add_argument('--test_set_rate', type=float, default=0.15, help='test set rate')

static methods, including HA and ARIMA, the HyperParameters setting as followings:

HA, historical experience data at the same time every day length we used is 7 days.
ARIMA, the parameters p, d and q values we used can selected from the followings:
    # p_values = [1, 2, 4, 6]
    # d_values = range(0, 3)
    # q_values = range(0, 3)
