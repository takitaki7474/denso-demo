from modules import argument, preprocess
import net
import numpy as np
import torch



VAL_DIR = "./data/val_data"
MODEL_PATH = "./models/model.pth"



if __name__=="__main__":
    args = argument.get_args()

    # 検証用データの生成
    val = preprocess.load_dataset(dataset_dir=VAL_DIR, imsize=(32,32), img_name=True)
    valloader = torch.utils.data.DataLoader(val, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # モデルの読み込み
    model = net.LeNet(2)
    model.load_state_dict(torch.load(MODEL_PATH))

    # 推論
    failures = []
    for (inputs, labels, img_name) in valloader:
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        failures += list(np.array(list(img_name))[predicted != labels]) # モデルが誤った画像名
    failures = list(set(failures))

    # 誤ったデータの出力
    print("\nfailures / total:   {0} / {1}\n".format(len(failures), len(val)))
    for failure in failures:
        print(failure)
