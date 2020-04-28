# MedianResNet

4年生ゼミ課題

## ノイズ除去

```
$ python inference.py
```

|変数|説明|
|----|----|
|model_location|pretrainedに保存した重みファイルのパス|
|img_location|ノイズ除去対象ファイル|

## トレーニング

```
$ python train.py
```

* 1から学習する場合は，L.56~57のコメントアウトを外す
* 事前学習した重みを引き継ぐ場合は，L.61~69を使用
* L.63に事前学習済みの重みファイルのパスを記述


## TODO

Pytorchへの書き換え