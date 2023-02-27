# gradiation_CNN
　Geant4行なったシミュレーションから学習用と検証用のデータセットを作成し、CNNの学習を行うコード。
なお、[gradiation_Geant4](https://github.com/SmallForestk/gradiation_Geant4)から作るビルドファイルは`CNN`というディレクトリと同じ階層に作成し、`B4_stable`をビルドしたものを実行するときには[gradiation_Geant4](https://github.com/SmallForestk/gradiation_Geant4)の中にある`energy.sh`を使うとそのまま、データセットを作成するためのディレクトリに移動するようになっている。
 
## 1.学習・検証の流れ
### 1.1.データセットの作成
