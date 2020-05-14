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
|model_location|pretrainedに保存した重みファイルへのパス|
|img_location|ノイズ除去対象ファイル|

## トレーニング

```
$ python train.py
```

* 0から学習する場合は，L.56~57のコメントアウトを外す
* 事前学習した重みを引き継ぐ場合は，L.61~69を使用
* L.63に事前学習済みの重みファイルパスを記述
* epoch数はL.125の`epochs`で指定

## 依存ライブラリ

* keras==2.1: ニューラルネットワークライブラリ
* tensorflow: ニューラルネットワークライブラリ（バッチ処理）
* numpy: 数値計算ライブラリ（行列生成）
* scikit-image: 画像処理ライブラリ（雑音画像生成）
* opencv-python: 画像処理ライブラリ（画像読み込み・保存，リサイズ，正規化）

## TODO

* PyTorchへ書き換えたい