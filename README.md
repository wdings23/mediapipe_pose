Compute joint orientation, outputting 4x4 animation matrix amd axis-angle rotations matching the 3D landmark positions output from mediapipe. The core logic is in the function compute_joint_local_rotation_anim_matrix(). Comment in the function as given:

1) get the length of the bones from the landmark
2) get the total anim matrix of the bones in the rig
3) set the translation part of the total matrix to the length of the bones from the landmarks
4) use the above as inverse applied to the angle-axis rotation of the bind pose joint to the landmark joint
(4 from above brings the landmark to local space of the joint in order to calculate the rotation needed)
5) verify the rotated joint world position with landmark, should have dot product near 1.0 and close distance

ADDENDUM: need to chain up animation matrices till the current joint and use it for step 2 instead of total bind matrix of the bone in the rig
this ensures that the rotation will be applied on the animated joint pose orientation, not the bind joint pose orientation 



# Initial bind pose of the rig in Blender 3D.
![alt text](https://github.com/wdings23/mediapipe_pose/blob/main/screenshot1.jpg?raw=true)




# After applied local joint rotations on the rig in Blender 3D. See the script with the angle-axis values in the script window as output from running get_pose.py.  
![alt text](https://github.com/wdings23/mediapipe_pose/blob/main/screenshot0.jpg?raw=true)
