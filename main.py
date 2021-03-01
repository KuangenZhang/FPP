from algo import utils
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def main_analysis(val_dir='data/3_walking_small_Chen_12_17', analysis_step = 0, exp_num = 5):
    if 0 == analysis_step:
        ''' change the folder number to 0, 1, 2, 3, 4 '''
        # utils.organize_folders(val_dir)
        utils.organize_files(val_dir)
    elif 1 == analysis_step:
        '''remove files that are far from the reference time interval'''
        for c in range(exp_num):
            reference_file_list = np.array(sorted(glob.glob("{}/test{}/heel_strike/*.jpg".format(val_dir, c))))
            file_name_list = np.array(list(set(sorted(glob.glob("{}/test{}/*/*".format(val_dir, c))))
                                           - set(reference_file_list) - set(glob.glob("{}/test{}/results/*".format(val_dir, c)))
                                           ))
            for file_name in file_name_list:
                utils.rename_file_time(file_name)
            print(reference_file_list)
            # utils.remove_files_far_from_reference_time_interval(file_name_list, reference_file_list)
    elif 2 == analysis_step:
        ''' Calculate the camera pose, the valide time should be between the first heel strike and the last heel strike.
        '''
        for c in range(exp_num):
            utils.calc_camera_pose(val_dir, test_idx=c)
            utils.fuse_multi_clouds(val_dir, test_idx=c, remove_ground=False)
            print('test idx: ', c)
    elif 3 == analysis_step:
        ''' Remove ground and segmented the point cloud. '''
        for c in [0]:
        # for c in range(exp_num):
            utils.fuse_multi_clouds(val_dir, test_idx=c, remove_ground=True)
    elif 4 == analysis_step:
        ''' Synchronize the gaze and plot gaze on the orbbec rgb images'''
        for c in range(exp_num):
            utils.synchronize_gaze_2d(val_dir, test_idx=c, save_video=True)
    elif 5 == analysis_step:
        ''' Analyze the gaze and compare with the random foot placements'''
        for c in range(exp_num):
            utils.analyze_gaze_and_footplacement(val_dir, test_idx=c, render_every_gait=True)
    elif 6 == analysis_step:
        ''' Analyze the gaze and compare with the labeled foot placements'''
        for c in range(exp_num):
            utils.analyze_gaze_and_footplacement(val_dir, test_idx=c, render_every_gait=True, change_phase = True)
    elif 7 == analysis_step:
        '''Visualize classification and distance error of gaze'''
        utils.plot_prediction_accuracy(val_dir)


def main(default_val_dir = 'data/3-walking_small_Zhang_12_21',):
    val_dir = input("The path of the experiment data. Default_path is {}".format(default_val_dir)) or default_val_dir
    print(val_dir)
    ''' Change the folder number to 0, 1, 2, 3, 4 '''
    main_analysis(val_dir, analysis_step=0)
    input('Please copy the heel strike images from web_camera_1 folder to heel_strike folder and press enter to continue.')
    '''Remove files that are far from the reference time interval'''
    main_analysis(val_dir, analysis_step=1)
    ''' Calculate the camera pose, the valide time should be between the first heel strike and the last heel strike.'''
    input('Please remove the unstable orbbec rgb images and press enter to continue.')
    while 1:
        main_analysis(val_dir, analysis_step=2)
        if (input('Are SLAM results correct? Input anything to continue the next step '
                  'or press enter to repeat the SLAM.')):
            break
    ''' Remove ground and segmented the point cloud. '''
    input('Please copy clear rgb images to swing folder.')
    while 1:
        main_analysis(val_dir, analysis_step=3)
        if (input('Are segmentation results correct? Input anything to continue the next step '
                  'or press enter to repeat the segmentation.')):
            break
    ''' Synchronize the gaze and plot gaze on the orbbec rgb images'''
    while 1:
        main_analysis(val_dir, analysis_step=4)
        if (input('Are gaze_in_depth results correct? Input anything to continue the next step '
                  'or press enter to repeat the gaze synchronization.')):
            break
    ''' Analyze the gaze and compare with the random foot placements'''
    main_analysis(val_dir, analysis_step=5)
    input('Please compare images in the results/2d_gaze_and_foot_placements and web_camera_2 '
          'to label the indices of foot placements and check the foot placement position and press enter to continue.')
    ''' Analyze the gaze and compare with the labeled foot placements'''
    main_analysis(val_dir, analysis_step=6)
    '''Visualize classification and distance error of gaze'''
    print('Plot the classification accuracy and distance error.')
    main_analysis(val_dir, analysis_step=7)
    print('See the classification and distance error in exp3_cls_acc.pdf and exp3_location_error.pdf files. '
          'The program is finished! Hope you get a good result!')

def result_analysis(analysis_step, exp_num = 5):
    val_dir_list = [
        'data/3_walking_small_Chen_12_21',
        'data/3-walking_small_Liang_12_21',
        # 'data/3-walking_small_Lin_12_21',
        'data/3-walking_small_Zhang_12_21',
        'data/3-walking_small_Zhou_12_23',
        # 'data/3_walking_small_Li_9_17',
    ]
    if 0 == analysis_step:
        '''Label foot placement position'''
        for val_dir in val_dir_list:
            utils.mark_foot_position(val_dir)
    elif 1 == analysis_step:
        ''' Calculate the position of foot placements and plot them on the 2D gaze and environmental images'''
        for val_dir in val_dir_list:
            c = 4
            # for c in range(exp_num):
            print(val_dir + ', test: {}'.format(c))
            utils.analyze_gaze_and_footplacement(val_dir, test_idx=c, render_every_gait=True, change_phase = True)
            # # '''Visualize classification and distance error of gaze'''
            # utils.plot_prediction_accuracy(val_dir)
    elif 2 == analysis_step:
        # for val_dir in val_dir_list:
        #     utils.plot_prediction_accuracy(val_dir)
        '''Visualize classification and distance error of gaze'''
        utils.plot_all_prediction_accuracy(val_dir_list)
    elif 3 == analysis_step:
        utils.plot_gaze_foot_error_bar(val_dir_list)
    elif 4 == analysis_step:
        utils.analyze_gait_parameters(val_dir_list)
    elif 5 == analysis_step:
        ''' Render the video of 3D walking, 2D temporal gazes, 2D gazes, and the global camera. '''
        lead_time_vec = [-0.3,  -0.1, -0.3, -0.3, -0.7, -0.3,]
        window_length_vec = [0.7, 0.5, 0.7,  0.7,  0.5,  0.7,]
        initial_right_vec = [False, True, False, False, True, False]
        for i in range(len(val_dir_list)):
            val_dir = val_dir_list[i]
            c = 5
            utils.synchronize_gaze_2d(val_dir, test_idx=c, save_video=True)
            if 'Li_9_17' not in val_dir:
                utils.save_global_cam_video(val_dir, test_idx=c)
            utils.render_gaze_and_foot_placements_video(
                val_dir, test_idx=c, lead_time = lead_time_vec[i], window_length = window_length_vec[i])
            utils.save_temporal_gaze_video(
                val_dir, test_idx=c, lead_time = lead_time_vec[i],
                window_length = window_length_vec[i], initial_right = initial_right_vec[i])
    elif 6 == analysis_step:
        utils.combine_videos(val_dir_list, test_idx = 4)
    elif 7 == analysis_step:
        for i in range(len(val_dir_list)):
            val_dir = val_dir_list[i]
            c = 9
            video_name = 'results/videos/{}_gaze_video.mp4'.format(os.path.basename(val_dir))
            utils.synchronize_gaze_2d(val_dir, test_idx=c, save_video=True, video_name=video_name)



if __name__ == "__main__":
    result_analysis(analysis_step=5)

