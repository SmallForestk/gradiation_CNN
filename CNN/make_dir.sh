mkdir train_dataset
cd ./train_dataset
mkdir CNNparameter
mkdir Conv3D_result
mkdir hitmap3D
cd ../

mkdir test_dataset
cd ./test_dataset
for i in `seq 15`
do
	energy=`expr $i \* 2`
	mkdir "data_${energy}GeV"
	cd "./data_${energy}GeV"
	mkdir hitmap3D
	mkdir Conv3D_result
	cd ../
done
