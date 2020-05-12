# MedianResNet

4年生ゼミ課題

## ノイズ除去

```
Usage: 
    python inference.py
    or
    python inference.py model_location img_location
Args:
    model_location: str, '../pretrained/fullyConvMedian.hdf5' as default
    img_location: str, '../data/test/rdata4.bmp' as default
```

|変数|説明|
|----|----|
|model_location|pretrainedに保存した重みファイルのパス|
|img_location|ノイズ除去対象ファイル|

## トレーニング

```
$ python train.py
```

* 0から学習する場合は，L.56~57のコメントアウトを外す
* 事前学習した重みを引き継ぐ場合は，L.61~69を使用
* L.63に事前学習済みの重みファイルパスを記述
* epoch数はL.125の`epochs`で指定

## ライブラリ

* keras==2.1.5
* tensorflow
* numpy
* scikit-image
* opencv-python

## TODO

* PyTorchへ書き換えたい