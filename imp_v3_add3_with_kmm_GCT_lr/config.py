CFG = {
    # 'data_path': 'E:\\AmigoChou\\Q2\\Training\\O2M_amigo\\datasets\\feature_map\\apple\\all_in_one\\5',
    'kwargs': {'num_workers': 4},
    # 'batch_size': 8,
    # 'epoch': 150,
    # 'lr': 0.0001,
    # 'momentum': .9,
    
    'log_interval': 25,
    'l2_decay': 8e-5,
    'betas': [0.9, 0.999],
    
    # 'lambda': 1,
    
    'backbone': 'resnet18',
    'n_class': 2,
}
