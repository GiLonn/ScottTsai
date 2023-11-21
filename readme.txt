backbone_multi: 定義特徵提取器with LiMAR module
call_resnet18_multi: 給resnet-18 架構相同部分預訓練參數 (不同部分參數則隨機)
ECANet: ECA 通道注意力機制 (一對多未使用)
estimate_mu : 計算超參數mu
instance_main:主程式 (包含train 跟test)
KMM_Lin:計算KMM
LabelSmoothing:標籤平滑程式
mmd_AMRAN : MMD and CMMD 計算
models:完整模型 (包含fc)
data_loader: 讀資料