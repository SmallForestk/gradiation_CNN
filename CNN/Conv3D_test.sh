for i in `seq 1 15`
do
    energy=`expr $i \* 2`
    echo "execute test Energy=${energy}"
    python ConvNN_3D_h5test.py "test_dataset/data_${energy}GeV" 3 79 pi "${energy}" "./test_dataset/data_${energy}GeV/hitmap3D/hitmap.h5"
done