python ../messege.py "start_training"
cp ../CNN_3cm/train_dataset/hitmap3D/hitmap.h5 /mnt/scratch/kobayashik

python ../messege.py start Learning Rate=10^-3
echo "execute Learning Rate=10^-3"
python ConvNN_3D_h5.py train_dataset 3 200 pi
python ../messege.py "end Learning Rate=10^-3"

rm /mnt/scratch/kobayashik/hitmap.h5
