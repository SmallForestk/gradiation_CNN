python make_3Dhitmap_h5.py ./pi_train.root 90 0 90 0 48 0 30 30
for i in `seq 1 15`
do
    energy=`expr $i \* 2`
    echo "execute in Cnn_${energy}GeV"
    python make_3Dhitmap_h5.py "./pi_${energy}GeV.root" 90 0 90 0 48 0 30 30
done
