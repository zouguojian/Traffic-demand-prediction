# traffic-demand-prediction

## Model one
### O-STEDN: An Optimized Spatiotemporal Encoder-Decoder Neural Network for Ridesourcing Demand Prediction

* Abstract: 
> Predicting the concentration of air pollutants is an effective method for preventing pollution 
incidents by providing an early warning of harmful substances in the air. Accurate prediction of 
air pollutant concentration can more effectively control and prevent air pollution. In this study, 
a big data correlation principle and deep learning technology are used for a proposed model of 
predicting PM2.5 concentration. The model comprises a deep learning network model based on a 
residual neural network (ResNet) and a convolutional long short-term memory (LSTM) network 
(ConvLSTM). ResNet is used to deeply extract the spatial distribution features of pollutant 
concentration and meteorological data from multiple cities. The output is used as input to 
ConvLSTM, which further extracts the preliminary spatial distribution features extracted from 
the ResNet, while extracting the spatiotemporal features of the pollutant concentration and 
meteorological data. The model combines the two features to achieve a spatiotemporal correlation
 of feature sequences, thereby accurately predicting the future PM2.5 concentration of the target
  city for a period of time. Compared with other neural network models and traditional models, 
  the proposed pollutant concentration prediction model improves the accuracy of predicting 
  pollutant concentration. For 1- to 3-hours prediction tasks, the proposed pollutant 
  concentration prediction model performed well and exhibited root mean square error (RMSE) 
  between 5.478 and 13.622. In addition, we conducted multiscale predictions in the target city 
  and achieved satisfactory performance, with the average RMSE value able to reach 22.927 even 
  for 1- to 15-hours prediction tasks.  

> code link [click](https://github.com/zouguojian/Traffic-demand-prediction/tree/main/OD)