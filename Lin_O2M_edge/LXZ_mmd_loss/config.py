CFG = {
    # 'data_path': 'E:\\AmigoChou\\Q2\\Training\\O2M_LXZ vs ZYM\\datasets\\guava\\sn_200_sp_200',
    'kwargs': {'num_workers': 4},
    # 'batch_size': 8,
    # 'lr': 0.0001,
    # 'lambda': 1000,
    # 'epoch': 200,

    # 'lr': 0.1,
    # 'lambda': 1,
    # 'epoch': 30,

    # 'momentum': .9,
    'log_interval': 25,
    'l2_decay': 1e-4,
    'betas': [0.9, 0.999],

    'backbone': 'resnet18',
    'n_class': 2,
}
