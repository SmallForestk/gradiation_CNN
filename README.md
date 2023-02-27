# gradiation_CNN
　Geant4行なったシミュレーションから学習用と検証用のデータセットを作成し、CNNの学習を行うコード。
なお、[gradiation_Geant4](https://github.com/SmallForestk/gradiation_Geant4)から作るビルドファイルは`CNN`というディレクトリと同じ階層に作成し、`B4_stable`をビルドしたものを実行するときには[gradiation_Geant4](https://github.com/SmallForestk/gradiation_Geant4)の中にある`energy.sh`を使うとそのまま、データセットを作成するためのディレクトリに移動するようになっている。
 
## 1.学習・検証の流れ
### 1.1.データセットの作成
　まず、`CNN`のディレクトリ内にある`make_dir.sh`を実行する。
するとデータセットを保存するディレクトリとデータセット作成のためのrootファイルを保存するディレクトリが作成される。

　次に`make_3Dimage_h5.sh`を実行すると、`make_3Dhitmap_h5.py`が実行されて、学習用・検証用のデータセットがh5ファイルで`make_dir.sh`で作成したディレクトリに作成される。
 
### 1.2.学習
学習を行う際には
```
python ConvNN_3D_h5.py train_dataset 3 200 pi
```
というコマンドを実行するとCNNの学習が始まる。（なお、引数のうち"3"はLearning Rateを指定する引数でこの場合はLearning Rate=10^-3、"200"は学習を行うepoch数を指定する引数）
このコマンドを実行すると、`CNN/train_dataset/CNNparameter`にCNNの各epochごとでのパラメータが保存され、`CNN/train_dataset/Conv3D_result`に学習中のLossの変化などの結果が保存される。
また、学習時のログに学習が完了するとValidation Lossが最小となったepoch数が出力される。

### 1.3.検証
このCNNのコードでは検証は２つのデータを使って行われる。
一つは学習を行うと、学習用のデータセットをtrain, validaition, testに分割し、test用のデータセットをが保存されるので、それを用いて教師データが連続であるデータセットを再構成した場合の検証である。
もう一つは検証用の入射エネルギーが一定のデータセットを使用した検証である。

前者の検証を行う際には、
```
python ConvNN_3D_h5retest.py 3 79 pi
```
というコマンドによって実行される。
なお、このコマンドにおける"79"という数字は学習時に出力されたValidation Lossが最小となったepoch数を入力する場所となっており、学習の結果を見ながら値を変更する。

後者の検証を行う際には、`Conv3D_test.sh`を実行する。なお、このshellスクリプトにもValidation Lossが最小となったepoch数を入力する場所を指定する場所があり、
```
5    python ConvNN_3D_h5test.py "test_dataset/data_${energy}GeV" 3 79 pi "${energy}" "./test_dataset/data_${energy}GeV/hitmap3D/hitmap.h5"
```
という様になっているので"79"の部分を変更する。

### 1.4.データの分析
　1.3.検証までを行うと再構成結果が保存されるので、この結果を分析する際には、まず、
```
python ./analysis_3D.py 3 79 analysis_3D_3lr.root 3
```
というコマンドを実行する。
なお、ここでも"79"の部分はValidation Lossが最小となったepoch数を入力する場所となっているのでその都度変更する。
（他の数字は、学習の際のLearning Rateと使用したCNNの次元数を指定する値となっているが、変更はしなくて良い）
すると、`CNN`のディレクトリに`analysis_3D_3lr.root`というrootファイルが作成され、LinearityやResolutionのプロットがまとめられる。
まとめられた結果を確認する際には`analysis_3D_3lr.ipynb`を使用する。
`analysis_3D_3lr.ipynb`の中のコマンドを実行するとプロットやヒストグラムが表示される様になっている。
