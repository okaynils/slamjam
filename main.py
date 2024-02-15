import numpy as np
import cv2
from matplotlib import pyplot as plt
import open3d as o3d

pcd = o3d.geometry.PointCloud()
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='3D Visualization', width=800, height=600)

sample_point = np.asarray([[-1.61721649e+01,-1.25735164e-01,5.07755966e+01], [0,0,0]])
sample_color = np.asarray([[0, 0, 1], [0,1,0]])

pcd.points = o3d.utility.Vector3dVector(sample_point)
pcd.colors = o3d.utility.Vector3dVector(sample_color)

vis.add_geometry(pcd)

map_height, map_width = 600, 800
keypoint_map = np.zeros((map_height, map_width, 3), dtype=np.uint8)

video = cv2.VideoCapture('footage/driving.mp4')

if not video.isOpened(): 
    print("Error opening video file") 

import numpy as np
import cv2
from matplotlib import pyplot as plt

def calc_keypoints(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0), flags=0)
    return frame_with_keypoints, keypoints, descriptors

def match_keypoints(descriptors1, descriptors2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    return good

# calculate video properties
image_width, image_height = int(video.get(3)), int(video.get(4))
f = image_width  
cx, cy = image_width / 2, image_height / 2
K = np.array([[f, 0, cx],
              [0, f, cy],
              [0, 0, 1]])

prev_frame = None
prev_descriptors = None
prev_keypoints = None

global_3d_points = []

while(video.isOpened()):
    ret, frame = video.read()
    if ret:
        frame_with_keypoints, keypoints, descriptors = calc_keypoints(frame)

        if prev_frame is not None:
            # triangulation
            good_matches = match_keypoints(prev_descriptors, descriptors)
            points1 = np.float32([ prev_keypoints[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
            points2 = np.float32([ keypoints[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
            E, mask = cv2.findEssentialMat(points1, points2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, mask = cv2.recoverPose(E, points1, points2, K)

            # projections
            P0 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))
            P1 = np.dot(K, np.hstack((R, t)))
            
            points_4d_hom = cv2.triangulatePoints(P0, P1, points1, points2)
            points_3d = points_4d_hom / np.tile(points_4d_hom[-1, :], (4, 1))
            points_3d = points_3d[:3, :].T  # Convert from homogeneous coordinates to 3D

            if points_3d.size > 0:
                for point in points_3d:
                    global_3d_points.append(point)
                
            if len(global_3d_points) > 0:
                all_points_o3d = o3d.utility.Vector3dVector(np.array(global_3d_points))
                pcd.points = all_points_o3d
                
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()

                

        cv2.imshow('Frame with Keypoints', frame_with_keypoints)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):  # Press Q on keyboard to exit
            break

        prev_frame = frame
        prev_descriptors = descriptors
        prev_keypoints = keypoints
    else: 
        break

video.release()
cv2.destroyAllWindows()
vis.destroy_window()
