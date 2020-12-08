from modules import argument, plot, preprocess, training
from utils import log_loader
import net
import torch




TRAIN_DIR = "./data/train_data"
TEST_DIR = "./data/test_data"

LOG_SAVEPATH = "./assets/log.json"
MODEL_SAVEPATH = "./models/model.pth"

LOSS_SAVEPATH = "./assets/loss.png"
ACC_SAVEPATH = "./assets/accuracy.png"




if __name__=="__main__":
    args = argument.get_args()
    print("\nepochs: {0}".format(args.epochs))
    print("batch size: {0}".format(args.batch_size))
    print("learning rate: {0}\n".format(args.lr))

    # 訓練データの生成
    train = preprocess.load_dataset(dataset_dir=TRAIN_DIR, dataN=args.dataN, imsize=(32,32))
    trainloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print("number of train data: {0}".format(len(train)))

    # テストデータの生成
    test = preprocess.load_dataset(dataset_dir=TEST_DIR, dataN=args.dataN//4, imsize=(32,32))
    testloader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print("number of train data: {0}".format(len(test)))

    # 学習と評価
    model = net.LeNet(2)
    model = training.process(trainloader, testloader, model, args.epochs, args.lr, log_savepath=LOG_SAVEPATH)
    torch.save(model.state_dict(), MODEL_SAVEPATH)

    # 学習結果の可視化
    train_losses, test_losses, train_accs, test_accs = log_loader.load_log(path=LOG_SAVEPATH)
    plot.plot_loss(train_losses, test_losses, savepath=LOSS_SAVEPATH)
    plot.plot_acc(train_accs, test_accs, savepath=ACC_SAVEPATH)
