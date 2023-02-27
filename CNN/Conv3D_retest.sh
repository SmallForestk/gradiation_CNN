cp ../CNN_3cm/train_dataset/hitmap3D/hitmap.h5 /mnt/scratch/kobayashik
python ConvNN_3D_h5retest.py 3 79 pi
rm /mnt/scratch/kobayashik/hitmap.h5
