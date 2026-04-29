extract_frames_from_video:
	uv run --active frame_extractor.py "/home/valdemar/GitHub/UAS2git/LSDP/2026-03-27 Drone flights/DJI_202603271447_003/DJI_20260327145405_0001_D.MP4" extracted_frames/ 10

extract_frames_from_calibration_video:
	uv run --active frame_extractor.py "/home/valdemar/GitHub/UAS2git/LSDP/2026-03-27 Drone flights/DJI_202604090836_003/DJI_20260409083842_0001_D.MP4" input/camera_calibration/ 10

run_on_all_frames:
	uv run visual_slam.py extracted_frames/

run_on_all_frames_no_gui:
	uv run visual_slam.py --skip_gui extracted_frames/ | tee output/all_frames.txt

run_on_selected_frames:
	uv run visual_slam.py selected_frames/

extract_camera_track_from_output:
	cat output/all_frames.txt | grep "Camera" | grep "True" | sort -n -k 2 | uniq > output/camera_track.txt