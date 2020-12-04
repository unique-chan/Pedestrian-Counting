# 동영상
# python3 count_single_view.py --source ./data/project.avi --view-img

# 이미지
#for i in $(seq 0 4); do
#  for j in $(seq 1 3); do
#    echo 비디오 ${i} - 클립 ${j}
#    python3 count_single_view.py --source ./data/3rd_track_1_0/t1_video_00${i}/t1_video_00${i}_${j} --view-img
#  done
#done

#python3 predict.py /media/chan/0552031b-9427-4239-a09d-09391a1f3061/PycharmProjects/그챌_데이터/3rd_track_1_0
#python3 predict_pipeline_changed.py /media/chan/0552031b-9427-4239-a09d-09391a1f3061/PycharmProjects/그챌_데이터/3rd_track_1_0
#python3 predict_no_sim.py /media/chan/0552031b-9427-4239-a09d-09391a1f3061/PycharmProjects/그챌_데이터/3rd_track_1_0
python3 predict.py /media/chan/0552031b-9427-4239-a09d-09391a1f3061/PycharmProjects/그챌_데이터/3rd_track_1_0

#python3 identification/tools/demo.py --gpu 0 --checkpoint identification/data/trained_model/checkpoint_step_50000.pth

# 학습
#python3 detection/train.py --data data/people_train/people_train.yaml --cfg yolov5x.yaml --cfg detection/yolov5/weights/people_train.pt --batch_size 8 --epochs 300
