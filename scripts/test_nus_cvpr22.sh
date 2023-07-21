
# # for device in val_Olympus_f0 val_Samsung_f0 # val_Sony_f0 
# # do
# # for q in 10 30 50 70 90
# # do
# # CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 python3 examples/test.py -g journal_jpeg_list -i 6 -q $q --dataset ./datasets/NUS --val $device --raw_space xyz_gamma
# # done
# # done


for drop_num in 0 1 2 3
do
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 python3 examples/test.py -g journal_cvpr22_J4_list -i 2 --dataset /home/Dataset/DatasetYufei/content_aware_reconstruction/SonyA57/ --val _ --cache --drop_num $drop_num

CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 python3 examples/test.py -g journal_cvpr22_J4_list -i 1 --dataset /home/Dataset/DatasetYufei/content_aware_reconstruction/SamsungNX2000/ --val _ --cache --file_type png --drop_num $drop_num

CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 python3 examples/test.py -g journal_cvpr22_J4_list -i 0 --dataset /home/Dataset/DatasetYufei/content_aware_reconstruction/OlympusEPL6/ --val _ --cache --drop_num $drop_num
done

# for gamma in 0 # 1 2 3 4 5
# do
# CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 python3 examples/test.py -g journal_cvpr22_J4_list -i 0 --dataset /home/Dataset/DatasetYufei/content_aware_reconstruction/OlympusEPL6/ --val _ --cache --gamma $gamma
# done