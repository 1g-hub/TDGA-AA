# TDGA AutoAugment

Experiment for TDGA AutoAugment

TDGA
[遺伝アルゴリズムにおける熱力学的選択ルールの提案](https://www.jstage.jst.go.jp/article/iscie1988/9/2/9_2_82/_pdf)

## Requirements
- [Docker](https://www.docker.com/) >= 19.03
- [GNU Make](https://www.gnu.org/software/make/)
- [nvidia-drivers](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#how-do-i-install-the-nvidia-driver) (Only for GPU)
- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) (Only for GPU)

## 導入手順
### レポジトリのクローン
```bash
$ git clone https://github.com/1g-hub/TDGA-AA
$ cd TDGA-AA
```

### イメージのビルド
```bash
$ make build
```

## 使い方

### コンテナを起動して bash に入る
```
$ make bash
```

#### Example
```
$ python main.py --auto_augment=true --mag=5 --tinit=0.02 --tfin=0.02 --prob_mul=2
```

#### Main Argment

- dataset: 用いるデータセット
- network: 用いるネットワーク
- lr: 初期学習率
- weight_decay: SGDで用いる重み減衰
- seed: シード値
- batch_size: バッチサイズ
- epochs: 最終訓練の学習エポック
- pre_train_epochs: 事前学習の学習エポック
- auto_augment: 拡張を探索するか
- mag: 拡張の強度
- tinit, tfin: TDGA の初期温度と最終温度
- B: 最終訓練のサブ方策数
- Np: GA の1世代の個体数
- Ng: GA の世代数
- prob_mul: 確率の乗算値

### 各ファイル, フォルダの説明
- main.py: メインファイル．引数と一緒に呼び出す．
- tdga_augment.py: TDGAコントローラでサブ方策を探索する
- transforms_range_prob.py: 探索する拡張集合を定める
- utils.py: util 関数を色々まとめたもの
- models/: モデル定義がまとめられている
- data/: 学習データが入る．スクリプト内で自動的にダウンロードされる
- runs/: 学習結果のoutputフォルダ．自動的に生成される．


### output の説明
runs/ 以下に 1 回の試行に対応するフォルダが生成される．
以下，中身の説明
- augmentation.cp: 採用された拡張が圧縮されたファイル．他の実験で同一の方策を使い回したいときに．
- events ファイル: tensorboard 用のログ．train/test accuracy の遷移が確認できる．
- kwargs.json: 実験設定の確認
- subpolicies.txt: 探索された方策の確認
- figures/entropy.png: 世代ごとの各遺伝子座のエントロピー遷移
- figures/stats.png: 探索中の統計量
- figures/num_transforms.png: エントロピー値と拡張個数の図
- model/model.pt: val acc 最大のモデル重み

