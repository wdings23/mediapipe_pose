#!/usr/bin/env python
# coding: utf-8

import mediapipe as mp
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math
import json
import time

class float3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    @staticmethod
    def add(v0, v1):
        return float3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z)

    def __add__(self, v):
        return float3(self.x + v.x, self.y + v.y, self.z + v.z)


    @staticmethod
    def subtract(v0, v1):
        return float3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z)

    def __sub__(self, v):
        return float3(self.x - v.x, self.y - v.y, self.z - v.z)

    @staticmethod
    def multiply(v0, v1):
        return float3(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z)

    def __mul__(self, v):
        return float3(self.x * v, self.y * v, self.z * v)

    def __idiv__(self, v):
        return float3(self.x / v, self.y / v, self.z / v)

    @staticmethod
    def dot(v0, v1):
        return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z

    @staticmethod
    def length(v):
        return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

    @staticmethod
    def normalize(v):
        magnitude = float3.length(v)
        ret = float3(0.0, 0.0, 0.0)
        ret.x = v.x / magnitude
        ret.y = v.y / magnitude
        ret.z = v.z / magnitude

        return ret

    @staticmethod
    def cross(v0, v1):
        return float3(
            v0.y * v1.z - v0.z * v1.y,
            v0.z * v1.x - v0.x * v1.z,
            v0.x * v1.y - v0.y * v1.x)

    @staticmethod
    def scalar_multiply(v, scalar):
        return float3(v.x * scalar, v.y * scalar, v.z * scalar)
     
##
class float4x4:

    ##
    def __init__(self, entries = [
        1.0, 0.0, 0.0, 0.0, 
        0.0, 1.0, 0.0, 0.0, 
        0.0, 0.0, 1.0, 0.0, 
        0.0, 0.0, 0.0, 1.0]):
        self.entries = entries

    ##
    def identity(self):
        self.entries = [
            1.0, 0.0, 0.0, 0.0, 
            0.0, 1.0, 0.0, 0.0, 
            0.0, 0.0, 1.0, 0.0, 
            0.0, 0.0, 0.0, 1.0
        ]

    ##
    @staticmethod
    def translate(translation):
        return float4x4([
            1.0, 0.0, 0.0, translation.x, 
            0.0, 1.0, 0.0, translation.y, 
            0.0, 0.0, 1.0, translation.z, 
            0.0, 0.0, 0.0, 1.0])

    ##
    @staticmethod
    def scale(scale):
        return float4x4([
            scale.x, 0.0, 0.0, 0.0, 
            0.0, scale.y, 0.0, 0.0, 
            0.0, 0.0, scale.z, 0.0, 
            0.0, 0.0, 0.0, 1.0])

    ##
    @staticmethod
    def rotate(q):
        return q.to_matrix()

    ##
    @staticmethod
    def multiply(m0, m1):
        entries = [
            1.0, 0.0, 0.0, 0.0, 
            0.0, 1.0, 0.0, 0.0, 
            0.0, 0.0, 1.0, 0.0, 
            0.0, 0.0, 0.0, 1.0]
        for i in range(0, 4):

            for j in range(0, 4):
                fResult = 0.0

                for k in range(0, 4):
                
                    iIndex0 = (i << 2) + k
                    iIndex1 = (k << 2) + j
                    fResult += (m0.entries[iIndex0] * m1.entries[iIndex1])
                
                entries[(i << 2) + j] = fResult
            
        return float4x4(entries)
        
    ##
    def __mul__(self, m):
        entries = [
            1.0, 0.0, 0.0, 0.0, 
            0.0, 1.0, 0.0, 0.0, 
            0.0, 0.0, 1.0, 0.0, 
            0.0, 0.0, 0.0, 1.0]
        for i in range(0, 4):

            for j in range(0, 4):
                fResult = 0.0

                for k in range(0, 4):
                
                    iIndex0 = (i << 2) + k
                    iIndex1 = (k << 2) + j
                    fResult += (self.entries[iIndex0] * m.entries[iIndex1])
                
                entries[(i << 2) + j] = fResult
            
        return float4x4(entries)

    ##
    @staticmethod
    def concat_matrices(matrices):
        curr_matrix = matrices[0]
        for i in range(1, len(matrices)):
            concat_matrices = float4x4.multiply(curr_matrix, matrices[i])
            curr_matrix = concat_matrices

        return float4x4(curr_matrix.entries)

    ##
    def apply(self, v):
        return float3(
            float3.dot(float3(self.entries[0], self.entries[1], self.entries[2]), v) + self.entries[3],
            float3.dot(float3(self.entries[4], self.entries[5], self.entries[6]), v) + self.entries[7],
            float3.dot(float3(self.entries[8], self.entries[9], self.entries[10]), v) + self.entries[11]
        )

    ##
    @staticmethod
    def from_angle_axis(axis, angle):
        fCosAngle = math.cos(angle)
        fSinAngle = math.sin(angle)
        fT = 1.0 - angle

        entries = [
            1.0, 0.0, 0.0, 0.0, 
            0.0, 1.0, 0.0, 0.0, 
            0.0, 0.0, 1.0, 0.0, 
            0.0, 0.0, 0.0, 1.0]
        fCosAngle = math.cos(angle)
        fSinAngle = math.sin(angle)
        fT = 1.0 - fCosAngle

        entries[0] = fT * axis.x * axis.x + fCosAngle
        entries[5] = fT * axis.y * axis.y + fCosAngle
        entries[10] = fT * axis.z * axis.z + fCosAngle

        fTemp0 = axis.x * axis.y * fT
        fTemp1 = axis.z * fSinAngle

        entries[4] = fTemp0 + fTemp1
        entries[1] = fTemp0 - fTemp1

        fTemp0 = axis.x * axis.z * fT
        fTemp1 = axis.y * fSinAngle

        entries[8] = fTemp0 - fTemp1
        entries[2] = fTemp0 + fTemp1

        fTemp0 = axis.y * axis.z * fT
        fTemp1 = axis.x * fSinAngle

        entries[9] = fTemp0 + fTemp1
        entries[6] = fTemp0 - fTemp1

        return float4x4(entries)

    ##
    @staticmethod
    def invert(m):
        inv = [
            1.0, 0.0, 0.0, 0.0, 
            0.0, 1.0, 0.0, 0.0, 
            0.0, 0.0, 1.0, 0.0, 
            0.0, 0.0, 0.0, 1.0] 
        invOut = [
            1.0, 0.0, 0.0, 0.0, 
            0.0, 1.0, 0.0, 0.0, 
            0.0, 0.0, 1.0, 0.0, 
            0.0, 0.0, 0.0, 1.0]
        
        inv[0] = (m.entries[5] * m.entries[10] * m.entries[15] -
            m.entries[5]  * m.entries[11] * m.entries[14] -
            m.entries[9]  * m.entries[6]  * m.entries[15] +
            m.entries[9]  * m.entries[7]  * m.entries[14] +
            m.entries[13] * m.entries[6]  * m.entries[11] -
            m.entries[13] * m.entries[7]  * m.entries[10])
        
        inv[4] = (-m.entries[4]  * m.entries[10] * m.entries[15] +
        m.entries[4]  * m.entries[11] * m.entries[14] +
        m.entries[8]  * m.entries[6]  * m.entries[15] -
        m.entries[8]  * m.entries[7]  * m.entries[14] -
        m.entries[12] * m.entries[6]  * m.entries[11] +
        m.entries[12] * m.entries[7]  * m.entries[10])
        
        inv[8] = (m.entries[4]  * m.entries[9] * m.entries[15] -
        m.entries[4]  * m.entries[11] * m.entries[13] -
        m.entries[8]  * m.entries[5] * m.entries[15] +
        m.entries[8]  * m.entries[7] * m.entries[13] +
        m.entries[12] * m.entries[5] * m.entries[11] -
        m.entries[12] * m.entries[7] * m.entries[9])
        
        inv[12] = (-m.entries[4]  * m.entries[9] * m.entries[14] +
        m.entries[4]  * m.entries[10] * m.entries[13] +
        m.entries[8]  * m.entries[5] * m.entries[14] -
        m.entries[8]  * m.entries[6] * m.entries[13] -
        m.entries[12] * m.entries[5] * m.entries[10] +
        m.entries[12] * m.entries[6] * m.entries[9])
        
        inv[1] = (-m.entries[1]  * m.entries[10] * m.entries[15] +
        m.entries[1]  * m.entries[11] * m.entries[14] +
        m.entries[9]  * m.entries[2] * m.entries[15] -
        m.entries[9]  * m.entries[3] * m.entries[14] -
        m.entries[13] * m.entries[2] * m.entries[11] +
        m.entries[13] * m.entries[3] * m.entries[10])
        
        inv[5] = (m.entries[0]  * m.entries[10] * m.entries[15] -
        m.entries[0]  * m.entries[11] * m.entries[14] -
        m.entries[8]  * m.entries[2] * m.entries[15] +
        m.entries[8]  * m.entries[3] * m.entries[14] +
        m.entries[12] * m.entries[2] * m.entries[11] -
        m.entries[12] * m.entries[3] * m.entries[10])
        
        inv[9] = (-m.entries[0]  * m.entries[9] * m.entries[15] +
        m.entries[0]  * m.entries[11] * m.entries[13] +
        m.entries[8]  * m.entries[1] * m.entries[15] -
        m.entries[8]  * m.entries[3] * m.entries[13] -
        m.entries[12] * m.entries[1] * m.entries[11] +
        m.entries[12] * m.entries[3] * m.entries[9])
        
        inv[13] = (m.entries[0]  * m.entries[9] * m.entries[14] -
        m.entries[0]  * m.entries[10] * m.entries[13] -
        m.entries[8]  * m.entries[1] * m.entries[14] +
        m.entries[8]  * m.entries[2] * m.entries[13] +
        m.entries[12] * m.entries[1] * m.entries[10] -
        m.entries[12] * m.entries[2] * m.entries[9])
        
        inv[2] = (m.entries[1]  * m.entries[6] * m.entries[15] -
        m.entries[1]  * m.entries[7] * m.entries[14] -
        m.entries[5]  * m.entries[2] * m.entries[15] +
        m.entries[5]  * m.entries[3] * m.entries[14] +
        m.entries[13] * m.entries[2] * m.entries[7] -
        m.entries[13] * m.entries[3] * m.entries[6])
        
        inv[6] = (-m.entries[0]  * m.entries[6] * m.entries[15] +
        m.entries[0]  * m.entries[7] * m.entries[14] +
        m.entries[4]  * m.entries[2] * m.entries[15] -
        m.entries[4]  * m.entries[3] * m.entries[14] -
        m.entries[12] * m.entries[2] * m.entries[7] +
        m.entries[12] * m.entries[3] * m.entries[6])
        
        inv[10] = (m.entries[0]  * m.entries[5] * m.entries[15] -
        m.entries[0]  * m.entries[7] * m.entries[13] -
        m.entries[4]  * m.entries[1] * m.entries[15] +
        m.entries[4]  * m.entries[3] * m.entries[13] +
        m.entries[12] * m.entries[1] * m.entries[7] -
        m.entries[12] * m.entries[3] * m.entries[5])
        
        inv[14] = (-m.entries[0]  * m.entries[5] * m.entries[14] +
        m.entries[0]  * m.entries[6] * m.entries[13] +
        m.entries[4]  * m.entries[1] * m.entries[14] -
        m.entries[4]  * m.entries[2] * m.entries[13] -
        m.entries[12] * m.entries[1] * m.entries[6] +
        m.entries[12] * m.entries[2] * m.entries[5])
        
        inv[3] = (-m.entries[1] * m.entries[6] * m.entries[11] +
        m.entries[1] * m.entries[7] * m.entries[10] +
        m.entries[5] * m.entries[2] * m.entries[11] -
        m.entries[5] * m.entries[3] * m.entries[10] -
        m.entries[9] * m.entries[2] * m.entries[7] +
        m.entries[9] * m.entries[3] * m.entries[6])
        
        inv[7] = (m.entries[0] * m.entries[6] * m.entries[11] -
        m.entries[0] * m.entries[7] * m.entries[10] -
        m.entries[4] * m.entries[2] * m.entries[11] +
        m.entries[4] * m.entries[3] * m.entries[10] +
        m.entries[8] * m.entries[2] * m.entries[7] -
        m.entries[8] * m.entries[3] * m.entries[6])
        
        inv[11] = (-m.entries[0] * m.entries[5] * m.entries[11] +
        m.entries[0] * m.entries[7] * m.entries[9] +
        m.entries[4] * m.entries[1] * m.entries[11] -
        m.entries[4] * m.entries[3] * m.entries[9] -
        m.entries[8] * m.entries[1] * m.entries[7] +
        m.entries[8] * m.entries[3] * m.entries[5])
        
        inv[15] = (m.entries[0] * m.entries[5] * m.entries[10] -
        m.entries[0] * m.entries[6] * m.entries[9] -
        m.entries[4] * m.entries[1] * m.entries[10] +
        m.entries[4] * m.entries[2] * m.entries[9] +
        m.entries[8] * m.entries[1] * m.entries[6] -
        m.entries[8] * m.entries[2] * m.entries[5])
        
        det = m.entries[0] * inv[0] + m.entries[1] * inv[4] + m.entries[2] * inv[8] + m.entries[3] * inv[12]
        if det <= 1.0e-5:
            for i in range(0, 16):
                invOut[i] = 1.0e9
        
        else:
            det = 1.0 / det
            for i in range(0, 16):
                invOut[i] = inv[i] * det
        
        return float4x4(invOut)


##
class quaternion:

    ##
    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    ##
    @staticmethod
    def from_angle_axis(axis, fAngle):
        return quaternion(
            axis.x * math.sin(fAngle * 0.5),
            axis.y * math.sin(fAngle * 0.5),
            axis.z * math.sin(fAngle * 0.5),
            math.cos(fAngle * 0.5)
        )
    
    @staticmethod
    def to_angle_axis(q):
        fDenom = math.sqrtf(1.0 - q.w * q.w)
        ret = float3(q.x, q.y, q.z)
        if fDenom > 0.00001:
            ret.x /= fDenom
            ret.y /= fDenom
            ret.z /= fDenom
           
        w = 2.0 * math.acos(q.w)
        
        return ret, w

    @staticmethod
    def multiply(q0, q1):
        return quaternion(
            q0.w * q1.x + q0.x * q1.w + q0.y * q1.z - q0.z * q1.y,
            q0.w * q1.y + q0.y * q1.w + q0.z * q1.x - q0.x * q1.z,
            q0.w * q1.z + q0.z * q1.w + q0.x * q1.y - q0.y * q1.x,
            q0.w * q1.w - q0.x * q1.x - q0.y * q1.y - q0.z * q1.z
        )
    

    ##
    def to_matrix(self):
        fXSquared = self.x * self.x
        fYSquared = self.y * self.y
        fZSquared = self.z * self.z
    
        fXMulY = self.x * self.y
        fXMulZ = self.x * self.z
        fXMulW = self.x * self.w
    
        fYMulZ = self.y * self.z
        fYMulW = self.y * self.w
    
        fZMulW = self.z * self.w
        
        afVal = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        afVal[0] = 1.0 - 2.0 * fYSquared - 2.0 * fZSquared
        afVal[1] = 2.0 * fXMulY - 2.0 * fZMulW
        afVal[2] = 2.0 * fXMulZ + 2.0 * fYMulW
        afVal[3] = 0.0
        
        afVal[4] = 2.0 * fXMulY + 2.0 * fZMulW
        afVal[5] = 1.0 - 2.0 * fXSquared - 2.0 * fZSquared
        afVal[6] = 2.0 * fYMulZ - 2.0 * fXMulW
        afVal[7] = 0.0
        
        afVal[8] = 2.0 * fXMulZ - 2.0 * fYMulW
        afVal[9] = 2.0 * fYMulZ + 2.0 * fXMulW
        afVal[10] = 1.0 - 2.0 * fXSquared - 2.0 * fYSquared
        afVal[11] = 0.0
        
        afVal[12] = afVal[13] = afVal[14] = 0.0
        afVal[15] = 1.0
        
        return float4x4(afVal)


class Joint:
    
    ##
    def __init__(self, name, rotation, scale, translation):
        self.name = name
        self.rotation = rotation
        self.scale = scale
        self.translation = translation
        self.children = []
        self.parent = None
        self.is_root_joint = False
        
        rotation_matrix = float4x4.rotate(rotation)
        translation_matrix = float4x4.translate(translation)
        scale_matrix = float4x4.scale(scale)
        
        #translation_rotation_matrix = float4x4.multiply(translation_matrix, rotation_matrix)
        #self.local_matrix = float4x4.multiply(translation_rotation_matrix, scale_matrix)

        self.local_matrix = float4x4.concat_matrices([translation_matrix, rotation_matrix, scale_matrix])

        self.total_matrix = float4x4()
        self.anim_matrix = float4x4()
        self.total_anim_matrix = float4x4()

        self.total_matrix.identity()
        self.anim_matrix.identity()
        self.total_anim_matrix.identity()

    ##
    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    ##
    def get_world_position(self):
        return float3(self.total_matrix.entries[3], self.total_matrix.entries[7], self.total_matrix.entries[11])

##
class Joint_Hierarchy:

    ##
    def __init__(self, joints = []):
        self.joints = joints
        self.root_joints = []

        self.joint_dict = {}

        for joint in joints:
            if joint.parent == None:
                self.root_joints.append(joint)
                joint.is_root_joint = True

            self.joint_dict[joint.name] = joint

        print('\n')
        for root_joint in self.root_joints:
            Joint_Hierarchy.traverse_joint(root_joint)
        print('\n')

    ##
    @staticmethod
    def traverse_joint(curr_joint):
        parent_joint = curr_joint.parent
        if parent_joint != None:
            curr_joint.total_matrix = float4x4.concat_matrices(
                [parent_joint.total_matrix,
                 curr_joint.local_matrix,
                 curr_joint.anim_matrix])
        else:
            curr_joint.total_matrix = float4x4.concat_matrices(
                [curr_joint.local_matrix,
                 curr_joint.anim_matrix])

        print('\"{}\" position ({}, {}, {})'.format(
            curr_joint.name, 
            curr_joint.total_matrix.entries[3],
            curr_joint.total_matrix.entries[7],
            curr_joint.total_matrix.entries[11]))

        if len(curr_joint.children) > 0:
            for child in curr_joint.children:
                Joint_Hierarchy.traverse_joint(child)

    ##
    def apply_scale(self, scale_value):
        for root_joint in self.root_joints:
            #root_joint.total_matrix.entries[0] = scale_value
            #root_joint.total_matrix.entries[5] = scale_value
            #root_joint.total_matrix.entries[10] = scale_value
            updated_anim_matrix = float4x4.concat_matrices([
                root_joint.anim_matrix,
                float4x4.scale(float3(scale_value, scale_value, scale_value))])
            root_joint.anim_matrix = updated_anim_matrix

        print('\n')
        for root_joint in self.root_joints:
            self.traverse_joint(root_joint)
        print('\n')

    ##
    def apply_rotation(self, axis, angle):
        for root_joint in self.root_joints:
            total_matrix = root_joint.total_matrix
            q = quaternion.from_angle_axis(axis, angle)
            angle_axis_rotation_matrix = q.to_matrix()
            #root_joint.total_matrix = float4x4.multiply(angle_axis_rotation_matrix, total_matrix)
            updated_anim_matrix = float4x4.concat_matrices([
                root_joint.anim_matrix,
                float4x4.rotate(q)])
            root_joint.anim_matrix = updated_anim_matrix

        print('\n')
        for root_joint in self.root_joints:
            self.traverse_joint(root_joint)
        print('\n')

    ##
    def apply_rotation_to_joint(self, joint_name, axis, angle):
        for joint in self.joints:
            if joint.name == joint_name:
                q = quaternion.from_angle_axis(axis, angle)
                angle_axis_rotation_matrix = q.to_matrix()
                updated_local_matrix = float4x4.multiply(angle_axis_rotation_matrix, joint.local_matrix)
                joint.local_matrix = updated_local_matrix 

        print('\n')
        for root_joint in self.root_joints:
            self.traverse_joint(root_joint)
        print('\n')

'''
##
def traverse_joint(curr_joint):
    parent_joint = curr_joint.parent
    if parent_joint != None:
        curr_joint.total_matrix = float4x4.multiply(parent_joint.total_matrix, curr_joint.local_matrix)

    print('\"{}\" position ({}, {}, {})'.format(
        curr_joint.name, 
        curr_joint.total_matrix.entries[3],
        curr_joint.total_matrix.entries[7],
        curr_joint.total_matrix.entries[11]))

    if len(curr_joint.children) > 0:
        for child in curr_joint.children:
            traverse_joint(child)
'''

##
def output_debug_blender_3d_spheres(
    landmark_positions,
    default_color,
    color_indices0,
    color0,
    color_indices1,
    color1):

    for i in range(0, len(landmark_positions)):
        position = landmark_positions[i]

        color = default_color

        if i in color_indices0:
            color = color0

        if i in color_indices1:
            color = color1

        print('draw_sphere([{}, {}, {}], 0.02, {}, {}, {}, 255) # {}'.format(
            position.x,
            position.y,
            position.z,
            color.x,
            color.y,
            color.z,
            i))


##
def test_rig(landmark_positions):
    
    json_file = open('c:\\Users\\dingwings\\demo-models\\media-pipe\\test-rig-1.gltf', 'r')
    file_content = json_file.read()
    json_file.close()
    rig = json.loads(file_content)
    nodes = rig['nodes']
    
    # create joints
    joints = []
    children_indices = []
    for node in nodes:
        child_indices = []
        if 'children' in node:
            child_indices = node['children']
        
        children_indices.append(child_indices)

        rotation = quaternion(0.0, 0.0, 0.0, 1.0)
        scale = float3(1.0, 1.0, 1.0)
        translation = float3(0.0, 0.0, 0.0)

        if 'rotation' in node:
            rotation = quaternion(node['rotation'][0], node['rotation'][1], node['rotation'][2], node['rotation'][3])
        
        if 'scale' in node:
            scale = float3(node['scale'][0], node['scale'][1], node['scale'][2])

        if 'translation' in node:
            translation = float3(node['translation'][0], node['translation'][1], node['translation'][2])

        curr_joint = Joint(
            name = node['name'],
            rotation = rotation,
            scale = scale,
            translation = translation)
            
        joints.append(curr_joint)
    
    axis_x = float3(1.0, 0.0, 0.0)
    axis_z = float3(0.0, 0.0, 1.0)

    # rotation angle of root joint
    hip = float3.subtract(landmark_positions[23], landmark_positions[24])
    hip_normalized = float3.normalize(hip)
    hip_axis = float3.cross(axis_z, hip_normalized)
    hip_angle = math.acos(float3.dot(axis_x, hip_normalized))

    shoulder_to_hip = float3.subtract(hip_right, shoulder_right)
    shoulder_to_hip_distance = float3.length(shoulder_to_hip)

    # add children
    for i in range(0, len(joints)):
        curr_joint = joints[i]
        for index in children_indices[i]:
            curr_joint.add_child(joints[index])

    hierarchy = Joint_Hierarchy(joints)

    rig_shoulder_to_hip_distance = hierarchy.joint_dict['left_shoulder'].get_world_position().y - hierarchy.joint_dict['pelvis'].get_world_position().y
    rig_ratio = shoulder_to_hip_distance / rig_shoulder_to_hip_distance

    landmark_ratio = 1.0 / rig_ratio

    scaled_landmark_positions = []
    for position in landmark_positions:
        scaled_landmark_positions.append(float3.scalar_multiply(position, landmark_ratio))

    output_debug_blender_3d_spheres(
        scaled_landmark_positions,
        float3(0.0, 0.0, 0.0),
        [11, 12],
        float3(0.0, 255.0, 0.0),
        [23, 24],
        float3(255.0, 0.0, 0.0))

    landmark_left_elbow_position = scaled_landmark_positions[14]

    print('rig scale = {}'.format(rig_ratio))
    print('rig angle = {} axis = ({}, {}, {})'.format((hip_angle * 180.0) / 3.14159, hip_axis.x, hip_axis.y, hip_axis.z))
    print('landmark scale = {}'.format(landmark_ratio))

    hierarchy.apply_rotation(hip_axis, hip_angle)

    left_shoulder_position = hierarchy.joint_dict['left_shoulder'].get_world_position()
    left_elbow_position = hierarchy.joint_dict['left_upper_arm'].get_world_position()

    left_elbow_position_normalized = float3.normalize(left_elbow_position)
    landmark_left_elbow_position_normalized = float3.normalize(landmark_left_elbow_position)


    


    shoulder_inverse_matrix = float4x4.invert(hierarchy.joint_dict['left_shoulder'].total_matrix)
    
    # brings to shoulder local coordinate system
    local_shoulder_elbow_pos = shoulder_inverse_matrix.apply(left_elbow_position)
    local_shoulder_landmark_pos = shoulder_inverse_matrix.apply(landmark_left_elbow_position)
    local_shoulder_elbow_pos_normalized = float3.normalize(local_shoulder_elbow_pos)
    local_shoulder_landmark_pos_normalized = float3.normalize(local_shoulder_landmark_pos)

    # angle axis
    local_axis = float3.cross(local_shoulder_elbow_pos_normalized, local_shoulder_landmark_pos_normalized)
    local_angle = math.acos(float3.dot(local_shoulder_elbow_pos_normalized, local_shoulder_landmark_pos_normalized))
    #local_axis = float3.cross(left_elbow_position_normalized, landmark_left_elbow_position_normalized)
    #local_angle = math.acos(float3.dot(left_elbow_position_normalized, landmark_left_elbow_position_normalized))
    local_axis_normalized = float3.normalize(local_axis)
    
    # rotation matrix
    local_shoulder_rotation_matrix = float4x4.from_angle_axis(local_axis_normalized, local_angle)
    check_result = local_shoulder_rotation_matrix.apply(local_shoulder_elbow_pos_normalized)
    
    print('draw_sphere([{}, {}, {}], 0.02, 255.0, 0.0, 0.0, 255) # landmark elbow'.format(local_shoulder_landmark_pos_normalized.x, local_shoulder_landmark_pos_normalized.y, local_shoulder_landmark_pos_normalized.z))
    print('draw_sphere([{}, {}, {}], 0.02, 0.0, 255.0, 0.0, 255) # elbow'.format(local_shoulder_elbow_pos_normalized.x, local_shoulder_elbow_pos_normalized.y, local_shoulder_elbow_pos_normalized.z))
    print('draw_sphere([{}, {}, {}], 0.02, 255.0, 255.0, 0.0, 255) # xform elbow'.format(check_result.x, check_result.y, check_result.z))


    ###########

    # apply local to shoulder vector to rotate to landmark matrix and shoulder total matrix
    
    # relative to shoulder's local space
    local_to_shoulder_position = float3(
        hierarchy.joint_dict['left_upper_arm'].local_matrix.entries[3],
        hierarchy.joint_dict['left_upper_arm'].local_matrix.entries[7],
        hierarchy.joint_dict['left_upper_arm'].local_matrix.entries[11])
    rotate_to_landmark_local = local_shoulder_rotation_matrix.apply(local_to_shoulder_position)

    # this should be 1 if they're at the same angle, don't necessarily need to be same position due the different length
    check_dp1 = float3.dot(float3.normalize(rotate_to_landmark_local), float3.normalize(local_shoulder_landmark_pos))
    
    # transform to world space descending from shoulder 
    should_be_same_angle_as_landmark_position = hierarchy.joint_dict['left_shoulder'].total_matrix.apply(rotate_to_landmark_local)

    # this should be close to 1 if they're simliar angle, making sure to bring them relative to shoulder position or else it's vector are 
    # from origin (0, 0, 0) which will not be the same
    check_dp2 = float3.dot(
        float3.normalize(should_be_same_angle_as_landmark_position - left_shoulder_position),
        float3.normalize(landmark_left_elbow_position - left_shoulder_position))

    ###########


    ###########

    # apply rotation to landmark in local space to local matrix
    local_matrix_with_local_rotation_to_landmark_matrix = local_shoulder_rotation_matrix * hierarchy.joint_dict['left_upper_arm'].local_matrix

    # apply to shoulder total matrix to move into world space
    world_space_matrix = hierarchy.joint_dict['left_shoulder'].total_matrix * local_matrix_with_local_rotation_to_landmark_matrix
    rotate_to_landmark_angle_position0 = float3(world_space_matrix.entries[3], world_space_matrix.entries[7], world_space_matrix.entries[11])

    # apply as above
    rotate_to_landmark_angle_space = float4x4.concat_matrices([
        hierarchy.joint_dict['left_shoulder'].total_matrix,
        local_shoulder_rotation_matrix,
        hierarchy.joint_dict['left_upper_arm'].local_matrix,
    ])

    rotate_to_landmark_angle_position1 = float3(rotate_to_landmark_angle_space.entries[3], rotate_to_landmark_angle_space.entries[7], rotate_to_landmark_angle_space.entries[11])

    v0 = float3.normalize(rotate_to_landmark_angle_position1 - left_shoulder_position)
    v1 = float3.normalize(landmark_left_elbow_position - left_shoulder_position)

    check_dp3 = float3.dot(
        v0,
        v1)

    ###########

    check_dp0 = float3.dot(
        float3.normalize(local_shoulder_landmark_pos_normalized),
        float3.normalize(check_result))
    
    left_elbow_joint_total_matrix3 = float4x4.concat_matrices([
        #hierarchy.joint_dict['root'].local_matrix,
        #hierarchy.joint_dict['root'].anim_matrix,
        #hierarchy.joint_dict['pelvis'].local_matrix,
        #hierarchy.joint_dict['pelvis'].anim_matrix,
        #hierarchy.joint_dict['left_clavicle'].local_matrix,
        #hierarchy.joint_dict['left_clavicle'].anim_matrix,

        #hierarchy.joint_dict['left_shoulder'].local_matrix,
        
        hierarchy.joint_dict['left_shoulder'].total_matrix,
        local_shoulder_rotation_matrix,

        #hierarchy.joint_dict['left_upper_arm'].local_matrix,  
        #hierarchy.joint_dict['left_upper_arm'].anim_matrix
    ])
    
    # check test_pos0 == landmark_left_elbow_position_normalized
    test_pos0 = local_shoulder_rotation_matrix.apply(left_elbow_position_normalized)

    new_elbow_pos = float3(left_elbow_joint_total_matrix3.entries[3], left_elbow_joint_total_matrix3.entries[7], left_elbow_joint_total_matrix3.entries[11])  
    
    landmark_normalized = float3.normalize(landmark_left_elbow_position)
    elbow_normalized = float3.normalize(left_elbow_position)
    new_elbow_normalized = float3.normalize(new_elbow_pos)
    
    
    check_dp1 = float3.dot(
        landmark_normalized,
        new_elbow_normalized)
    
    check_distance1 = float3.length(float3.subtract(new_elbow_pos, landmark_left_elbow_position))

    print('draw_sphere([{}, {}, {}], 0.02, 255.0, 0.0, 0.0, 255)'.format(landmark_normalized.x, landmark_normalized.y, landmark_normalized.z))
    print('draw_sphere([{}, {}, {}], 0.02, 0.0, 255.0, 0.0, 255)'.format(elbow_normalized.x, elbow_normalized.y, elbow_normalized.z))
    print('draw_sphere([{}, {}, {}], 0.02, 255.0, 255.0, 0.0, 255)'.format(new_elbow_normalized.x, new_elbow_normalized.y, new_elbow_normalized.z))

    hierarchy.apply_rotation_to_joint('left_upper_arm', elbow_axis_normalized, elbow_angle)

####### test_rig ########

##
def test_rig2(landmark_positions):
    json_file = open('c:\\Users\\dingwings\\demo-models\\media-pipe\\test-rig-1.gltf', 'r')
    file_content = json_file.read()
    json_file.close()
    rig = json.loads(file_content)
    nodes = rig['nodes']
    
    # create joints
    joints = []
    children_indices = []
    for node in nodes:
        child_indices = []
        if 'children' in node:
            child_indices = node['children']
        
        children_indices.append(child_indices)

        rotation = quaternion(0.0, 0.0, 0.0, 1.0)
        scale = float3(1.0, 1.0, 1.0)
        translation = float3(0.0, 0.0, 0.0)

        if 'rotation' in node:
            rotation = quaternion(node['rotation'][0], node['rotation'][1], node['rotation'][2], node['rotation'][3])
        
        if 'scale' in node:
            scale = float3(node['scale'][0], node['scale'][1], node['scale'][2])

        if 'translation' in node:
            translation = float3(node['translation'][0], node['translation'][1], node['translation'][2])

        curr_joint = Joint(
            name = node['name'],
            rotation = rotation,
            scale = scale,
            translation = translation)
            
        joints.append(curr_joint)
    
    # add children
    for i in range(0, len(joints)):
        curr_joint = joints[i]
        for index in children_indices[i]:
            curr_joint.add_child(joints[index])

    hierarchy = Joint_Hierarchy(joints)

    # 1) get the length of the bones from the landmark
    # 2) get the total matrix of the bones in the rig
    # 3) set the translation part of the total matrix to the length of the bones from the landmarks
    # 4) use the above as inverse applied to the angle-axis rotation of the bind pose joint to the landmark joint
    # (4 from above brings the landmark to local space of the joint in order to calculate the rotation needed)
    # 5) verify the rotated joint world position with landmark, should have dot product near 1.0 and close distance   

    shoulder_joint = hierarchy.joint_dict['left_shoulder']
    elbow_joint = hierarchy.joint_dict['left_upper_arm']

    landmark_shoulder_position = landmark_positions[12]
    landmark_elbow_position = landmark_positions[14]

    # length of the bones from landmark
    landmark_shoulder_to_elbow_length = float3.length(landmark_elbow_position - landmark_shoulder_position)

    # ratio of landmark bone length to rig bone length
    rig_shoulder_to_elbow_bone_length = float3.length(elbow_joint.translation)
    landmark_to_rig_ratio = landmark_shoulder_to_elbow_length / rig_shoulder_to_elbow_bone_length
    rig_scaled_local_joint_position = float3.scalar_multiply(elbow_joint.translation, landmark_to_rig_ratio) # this is translation in local space

    # total rig joint matrix with the translation of landmark 
    rig_shoulder_total_matrix_with_landmark_translation = shoulder_joint.total_matrix
    rig_shoulder_total_matrix_with_landmark_translation.entries[3] = landmark_shoulder_position.x
    rig_shoulder_total_matrix_with_landmark_translation.entries[7] = landmark_shoulder_position.y
    rig_shoulder_total_matrix_with_landmark_translation.entries[11] = landmark_shoulder_position.z

    # invert the above matrix to bring landmark to joint's local coordinate space
    inverse_rig_shoulder_total_matrix_with_landmark_translation = float4x4.invert(rig_shoulder_total_matrix_with_landmark_translation)

    # apply to the landmark and verify
    landmark_local_joint_position = inverse_rig_shoulder_total_matrix_with_landmark_translation.apply(landmark_elbow_position)
    verify_length = float3.length(landmark_local_joint_position)
    assert(verify_length - landmark_shoulder_to_elbow_length <= 1.0e-6)

    # get the local joint rotation matrix
    rig_local_joint_position_normalized = float3.normalize(rig_scaled_local_joint_position)
    landmark_local_joint_position_normalized = float3.normalize(landmark_local_joint_position)
    local_rotation_axis = float3.cross(rig_local_joint_position_normalized, landmark_local_joint_position_normalized)
    local_rotation_axis_normalized = float3.normalize(local_rotation_axis)
    local_rotation_angle = math.acos(float3.dot(rig_local_joint_position_normalized, landmark_local_joint_position_normalized))
    local_rotation_matrix = float4x4.from_angle_axis(local_rotation_axis_normalized, local_rotation_angle)
    
    local_rotation_axis = float3.cross(rig_local_joint_position_normalized, landmark_local_joint_position_normalized)
    local_rotation_axis_normalized = float3.normalize(local_rotation_axis)
    local_rotation_angle = math.acos(float3.dot(rig_local_joint_position_normalized, landmark_local_joint_position_normalized))
    local_rotation_matrix = float4x4.from_angle_axis(local_rotation_axis_normalized, local_rotation_angle)

    # verify rig's local space angle
    rig_rotated_local_position_normalized = local_rotation_matrix.apply(rig_local_joint_position_normalized)
    verify_dp = float3.dot(rig_rotated_local_position_normalized, landmark_local_joint_position_normalized)
    assert(((verify_dp - 1.0) * (verify_dp - 1.0)) <= 1.0e-6)

    # verify rig's local space distance
    rig_rotated_local_joint_position = local_rotation_matrix.apply(rig_scaled_local_joint_position)
    diff_position = rig_rotated_local_joint_position - landmark_local_joint_position
    diff_distance = float3.length(diff_position)
    assert(diff_distance < 1.0e-6)

    # verify world space angle using parent joint's position as origin, should be close to 1.0 ie. same angle
    # scale the bone length to match landmark for position verification
    scaled_elbow_local_matrix = elbow_joint.local_matrix
    scaled_elbow_local_matrix.entries[3] *= landmark_to_rig_ratio
    scaled_elbow_local_matrix.entries[7] *= landmark_to_rig_ratio
    scaled_elbow_local_matrix.entries[11] *= landmark_to_rig_ratio
    new_joint_matrix = float4x4.concat_matrices([
        shoulder_joint.total_matrix,
        local_rotation_matrix,
        scaled_elbow_local_matrix
    ])
    verify_joint_position = float3(new_joint_matrix.entries[3], new_joint_matrix.entries[7], new_joint_matrix.entries[11])
    shoulder_joint_position = float3(shoulder_joint.total_matrix.entries[3], shoulder_joint.total_matrix.entries[7], shoulder_joint.total_matrix.entries[11])
    verify_dp = float3.dot(
        float3.normalize(verify_joint_position - shoulder_joint_position),
        float3.normalize(landmark_elbow_position - shoulder_joint_position)
    )
    assert((verify_dp - 1.0) * (verify_dp - 1.0)  <= 1.0e-6)

    verify_distance = float3.length(verify_joint_position - landmark_elbow_position)
    assert(verify_distance <= 1.0e-6)

    #print('draw_sphere([{}, {}, {}], 0.02, 255.0, 0.0, 0.0, 255) # verify joint position'.format(
    #    verify_joint_position.x, verify_joint_position.y, verify_joint_position.z))

########### test_rig2 #########


##### compute_joint_local_rotation_anim_matrix #####
def compute_joint_local_rotation_anim_matrix(
    curr_joint, 
    landmark_positions,
    landmark_rig_mapping):

    # 1) get the length of the bones from the landmark
    # 2) get the total anim matrix of the bones in the rig
    # 3) set the translation part of the total matrix to the length of the bones from the landmarks
    # 4) use the above as inverse applied to the angle-axis rotation of the bind pose joint to the landmark joint
    # (4 from above brings the landmark to local space of the joint in order to calculate the rotation needed)
    # 5) verify the rotated joint world position with landmark, should have dot product near 1.0 and close distance

    # ADDENDUM: need to chain up animation matrices till the current joint and use it for step 2 instead of total bind matrix of the bone in the rig
    # this ensures that the rotation will be applied on the animated joint pose orientation, not the bind joint pose orientation 

    parent_joint = curr_joint.parent

    curr_joint_landmark_indices = landmark_rig_mapping[curr_joint.name]
    parent_joint_landmark_indices = landmark_rig_mapping[parent_joint.name]

    # get midpoint of parent joint larndmark position
    landmark_parent_joint_position = float3(0.0, 0.0, 0.0)
    for index in parent_joint_landmark_indices:
        landmark_position = landmark_positions[index]
        landmark_parent_joint_position += landmark_position
    landmark_parent_joint_position.x /= float(len(parent_joint_landmark_indices))
    landmark_parent_joint_position.y /= float(len(parent_joint_landmark_indices))
    landmark_parent_joint_position.z /= float(len(parent_joint_landmark_indices))

    # get midpoint of joint landmark position
    landmark_joint_position = float3(0.0, 0.0, 0.0) 
    for index in curr_joint_landmark_indices:
        landmark_position = landmark_positions[index]
        landmark_joint_position += landmark_position
    landmark_joint_position.x /= len(curr_joint_landmark_indices)
    landmark_joint_position.y /= len(curr_joint_landmark_indices)
    landmark_joint_position.z /= len(curr_joint_landmark_indices)

    # length of the bones from landmark
    landmark_parent_to_child_length = float3.length(landmark_joint_position - landmark_parent_joint_position)

    # ratio of landmark bone length to rig bone length
    rig_parent_to_curr_joint_bone_length = float3.length(curr_joint.translation)
    landmark_to_rig_ratio = landmark_parent_to_child_length / rig_parent_to_curr_joint_bone_length
    rig_scaled_local_joint_position = float3.scalar_multiply(curr_joint.translation, landmark_to_rig_ratio) # this is translation in local space

    # total rig joint matrix with the translation of landmark 
    rig_parent_joint_anim_matrix_with_landmark_translation = parent_joint.total_anim_matrix
    rig_parent_joint_anim_matrix_with_landmark_translation.entries[3] = landmark_parent_joint_position.x
    rig_parent_joint_anim_matrix_with_landmark_translation.entries[7] = landmark_parent_joint_position.y
    rig_parent_joint_anim_matrix_with_landmark_translation.entries[11] = landmark_parent_joint_position.z

    # invert the above matrix to bring landmark to joint's local coordinate space
    inverse_rig_parent_joint_total_matrix_with_landmark_translation = float4x4.invert(rig_parent_joint_anim_matrix_with_landmark_translation)

    # apply to the landmark and verify
    landmark_local_joint_position = inverse_rig_parent_joint_total_matrix_with_landmark_translation.apply(landmark_joint_position)
    verify_length = float3.length(landmark_local_joint_position)
    assert(verify_length - landmark_parent_to_child_length <= 1.0e-6)

    # get the local joint rotation matrix
    rig_local_joint_position_normalized = float3.normalize(rig_scaled_local_joint_position)
    landmark_local_joint_position_normalized = float3.normalize(landmark_local_joint_position)
    local_rotation_axis = float3.cross(rig_local_joint_position_normalized, landmark_local_joint_position_normalized)
    local_rotation_axis_normalized = float3.normalize(local_rotation_axis)
    local_rotation_angle = math.acos(float3.dot(rig_local_joint_position_normalized, landmark_local_joint_position_normalized))
    local_rotation_matrix = float4x4.from_angle_axis(local_rotation_axis_normalized, local_rotation_angle)
    
    # verify rig's local space angle
    rig_rotated_local_position_normalized = local_rotation_matrix.apply(rig_local_joint_position_normalized)
    verify_dp = float3.dot(rig_rotated_local_position_normalized, landmark_local_joint_position_normalized)
    assert(((verify_dp - 1.0) * (verify_dp - 1.0)) <= 1.0e-6)

    # verify rig's local space distance
    rig_rotated_local_joint_position = local_rotation_matrix.apply(rig_scaled_local_joint_position)
    diff_position = rig_rotated_local_joint_position - landmark_local_joint_position
    diff_distance = float3.length(diff_position)
    assert(diff_distance < 1.0e-6)

    # verify world space angle using parent joint's position as origin, should be close to 1.0 ie. same angle
    # scale the bone length to match landmark for position verification
    scaled_curr_joint_local_matrix = float4x4()
    for i in range(0, len(curr_joint.local_matrix.entries)):
        scaled_curr_joint_local_matrix.entries[i] = curr_joint.local_matrix.entries[i]
    scaled_curr_joint_local_matrix.entries[3] *= landmark_to_rig_ratio
    scaled_curr_joint_local_matrix.entries[7] *= landmark_to_rig_ratio
    scaled_curr_joint_local_matrix.entries[11] *= landmark_to_rig_ratio
    new_joint_matrix = float4x4.concat_matrices([
        parent_joint.total_anim_matrix,
        local_rotation_matrix,
        scaled_curr_joint_local_matrix
    ])
    verify_joint_position = float3(new_joint_matrix.entries[3], new_joint_matrix.entries[7], new_joint_matrix.entries[11])
    parent_joint_position = float3(parent_joint.total_anim_matrix.entries[3], parent_joint.total_anim_matrix.entries[7], parent_joint.total_anim_matrix.entries[11])
    verify_dp = float3.dot(
        float3.normalize(verify_joint_position - parent_joint_position),
        float3.normalize(landmark_joint_position - parent_joint_position)
    )
    assert((verify_dp - 1.0) * (verify_dp - 1.0)  <= 1.0e-6)

    verify_distance = float3.length(verify_joint_position - landmark_joint_position)
    assert(verify_distance <= 1.0e-6)

    #print('draw_sphere([{}, {}, {}], 0.02, 255.0, 255.0, 0.0, 255) # {} verify joint position'.format(
    #    verify_joint_position.x, verify_joint_position.y, verify_joint_position.z,
    #    curr_joint.name))

    new_joint_matrix2 = float4x4.concat_matrices([
        parent_joint.total_anim_matrix,
        local_rotation_matrix,
        curr_joint.local_matrix
    ])
    verify_joint_position2 = float3(new_joint_matrix2.entries[3], new_joint_matrix2.entries[7], new_joint_matrix2.entries[11])
    verify_dp2 = float3.dot(
        float3.normalize(verify_joint_position2 - parent_joint_position),
        float3.normalize(landmark_joint_position - parent_joint_position)
    )
    assert((verify_dp2 - 1.0) * (verify_dp2 - 1.0)  <= 1.0e-6)

    #print('draw_sphere([{}, {}, {}], 0.02, 0.0, 0.0, 255.0, 255) # {} verify joint position'.format(
    #    verify_joint_position2.x, verify_joint_position2.y, verify_joint_position2.z,
    #    curr_joint.name))

    return local_rotation_matrix, local_rotation_axis_normalized, local_rotation_angle

##### compute_joint_local_rotation_matrices #####

def compute_joint_local_rotation_matrices(
    rig, 
    landmark_positions,
    landmark_rig_mapping):

    # 1) get the length of the bones from the landmark
    # 2) get the total matrix of the bones in the rig
    # 3) set the translation part of the total matrix to the length of the bones from the landmarks
    # 4) use the above as inverse applied to the angle-axis rotation of the bind pose joint to the landmark joint
    # (4 from above brings the landmark to local space of the joint in order to calculate the rotation needed)
    # 5) verify the rotated joint world position with landmark, should have dot product near 1.0 and close distance   


    # TODO: need to chain up animation matrices till the current joint and use it for step 2 instead of total bind matrix of the bone in the rig
    # this ensures that the rotation will be applied on the animated joint pose orientation, not the bind joint pose orientation 

    joint_local_rotation_dict = {}

    for curr_joint in rig.joints:

        # skip root joints 
        if curr_joint.is_root_joint == True:
            continue

        parent_joint = curr_joint.parent

        # skip joint with no mapping
        if not curr_joint.name in landmark_rig_mapping or not parent_joint.name in landmark_rig_mapping:
            continue

        curr_joint_landmark_index = landmark_rig_mapping[curr_joint.name]
        parent_joint_landmark_index = landmark_rig_mapping[parent_joint.name]

        landmark_parent_joint_position = landmark_positions[parent_joint_landmark_index]
        landmark_joint_position = landmark_positions[curr_joint_landmark_index]

        # length of the bones from landmark
        landmark_parent_to_child_length = float3.length(landmark_joint_position - landmark_parent_joint_position)

        # ratio of landmark bone length to rig bone length
        rig_parent_to_curr_joint_bone_length = float3.length(curr_joint.translation)
        landmark_to_rig_ratio = landmark_parent_to_child_length / rig_parent_to_curr_joint_bone_length
        rig_scaled_local_joint_position = float3.scalar_multiply(curr_joint.translation, landmark_to_rig_ratio) # this is translation in local space

        # total rig joint matrix with the translation of landmark 
        rig_parent_joint_total_matrix_with_landmark_translation = parent_joint.total_matrix
        rig_parent_joint_total_matrix_with_landmark_translation.entries[3] = landmark_parent_joint_position.x
        rig_parent_joint_total_matrix_with_landmark_translation.entries[7] = landmark_parent_joint_position.y
        rig_parent_joint_total_matrix_with_landmark_translation.entries[11] = landmark_parent_joint_position.z

        # invert the above matrix to bring landmark to joint's local coordinate space
        inverse_rig_parent_joint_total_matrix_with_landmark_translation = float4x4.invert(rig_parent_joint_total_matrix_with_landmark_translation)

        # apply to the landmark and verify
        landmark_local_joint_position = inverse_rig_parent_joint_total_matrix_with_landmark_translation.apply(landmark_joint_position)
        verify_length = float3.length(landmark_local_joint_position)
        assert(verify_length - landmark_parent_to_child_length <= 1.0e-6)

        # get the local joint rotation matrix
        rig_local_joint_position_normalized = float3.normalize(rig_scaled_local_joint_position)
        landmark_local_joint_position_normalized = float3.normalize(landmark_local_joint_position)
        local_rotation_axis = float3.cross(rig_local_joint_position_normalized, landmark_local_joint_position_normalized)
        local_rotation_axis_normalized = float3.normalize(local_rotation_axis)
        local_rotation_angle = math.acos(float3.dot(rig_local_joint_position_normalized, landmark_local_joint_position_normalized))
        local_rotation_matrix = float4x4.from_angle_axis(local_rotation_axis_normalized, local_rotation_angle)
        
        # verify rig's local space angle
        rig_rotated_local_position_normalized = local_rotation_matrix.apply(rig_local_joint_position_normalized)
        verify_dp = float3.dot(rig_rotated_local_position_normalized, landmark_local_joint_position_normalized)
        assert(((verify_dp - 1.0) * (verify_dp - 1.0)) <= 1.0e-6)

        # verify rig's local space distance
        rig_rotated_local_joint_position = local_rotation_matrix.apply(rig_scaled_local_joint_position)
        diff_position = rig_rotated_local_joint_position - landmark_local_joint_position
        diff_distance = float3.length(diff_position)
        assert(diff_distance < 1.0e-6)

        # verify world space angle using parent joint's position as origin, should be close to 1.0 ie. same angle
        # scale the bone length to match landmark for position verification
        scaled_curr_joint_local_matrix = float4x4()
        for i in range(0, len(curr_joint.local_matrix.entries)):
            scaled_curr_joint_local_matrix.entries[i] = curr_joint.local_matrix.entries[i]
        scaled_curr_joint_local_matrix.entries[3] *= landmark_to_rig_ratio
        scaled_curr_joint_local_matrix.entries[7] *= landmark_to_rig_ratio
        scaled_curr_joint_local_matrix.entries[11] *= landmark_to_rig_ratio
        new_joint_matrix = float4x4.concat_matrices([
            parent_joint.total_matrix,
            local_rotation_matrix,
            scaled_curr_joint_local_matrix
        ])
        verify_joint_position = float3(new_joint_matrix.entries[3], new_joint_matrix.entries[7], new_joint_matrix.entries[11])
        parent_joint_position = float3(parent_joint.total_matrix.entries[3], parent_joint.total_matrix.entries[7], parent_joint.total_matrix.entries[11])
        verify_dp = float3.dot(
            float3.normalize(verify_joint_position - parent_joint_position),
            float3.normalize(landmark_joint_position - parent_joint_position)
        )
        assert((verify_dp - 1.0) * (verify_dp - 1.0)  <= 1.0e-6)

        verify_distance = float3.length(verify_joint_position - landmark_joint_position)
        assert(verify_distance <= 1.0e-6)

        # save local rotation matrix for parent
        joint_local_rotation_dict[parent_joint.name] = local_rotation_matrix

        #print('draw_sphere([{}, {}, {}], 0.02, 255.0, 255.0, 0.0, 255) # {} verify joint position'.format(
        #    verify_joint_position.x, verify_joint_position.y, verify_joint_position.z,
        #    curr_joint.name))

        new_joint_matrix2 = float4x4.concat_matrices([
            parent_joint.total_matrix,
            local_rotation_matrix,
            curr_joint.local_matrix
        ])
        verify_joint_position2 = float3(new_joint_matrix2.entries[3], new_joint_matrix2.entries[7], new_joint_matrix2.entries[11])
        verify_dp2 = float3.dot(
            float3.normalize(verify_joint_position2 - parent_joint_position),
            float3.normalize(landmark_joint_position - parent_joint_position)
        )
        assert((verify_dp2 - 1.0) * (verify_dp2 - 1.0)  <= 1.0e-6)

        #print('draw_sphere([{}, {}, {}], 0.02, 0.0, 0.0, 255.0, 255) # {} verify joint position'.format(
        #    verify_joint_position2.x, verify_joint_position2.y, verify_joint_position2.z,
        #    curr_joint.name))

    return joint_local_rotation_dict

##### compute_joint_local_rotation_matrices ######

##### load_rig #####

def load_rig(file_path):
    json_file = open(file_path, 'r')
    file_content = json_file.read()
    json_file.close()
    rig = json.loads(file_content)
    nodes = rig['nodes']
    
    # create joints
    joints = []
    children_indices = []
    for node in nodes:
        child_indices = []
        if 'children' in node:
            child_indices = node['children']
        
        children_indices.append(child_indices)

        rotation = quaternion(0.0, 0.0, 0.0, 1.0)
        scale = float3(1.0, 1.0, 1.0)
        translation = float3(0.0, 0.0, 0.0)

        if 'rotation' in node:
            rotation = quaternion(node['rotation'][0], node['rotation'][1], node['rotation'][2], node['rotation'][3])
        
        if 'scale' in node:
            scale = float3(node['scale'][0], node['scale'][1], node['scale'][2])

        if 'translation' in node:
            translation = float3(node['translation'][0], node['translation'][1], node['translation'][2])

        curr_joint = Joint(
            name = node['name'],
            rotation = rotation,
            scale = scale,
            translation = translation)
            
        joints.append(curr_joint)
    
    # add children
    for i in range(0, len(joints)):
        curr_joint = joints[i]
        for index in children_indices[i]:
            curr_joint.add_child(joints[index])

    rig = Joint_Hierarchy(joints)

    return rig

##### load_rig #####

##
def traverse_for_anim_matrix(
    curr_joint,
    landmark_positions,
    landmark_rig_mapping,
    local_anim_rotation_axis_angles):

    # parent joint total anim matrix, identity if no parent
    parent_joint_total_anim_matrix = float4x4()
    parent_joint_total_anim_matrix.identity()
    if(curr_joint.parent != None):
        parent_joint_total_anim_matrix = curr_joint.parent.total_anim_matrix

    # check if this joint needs to update its anim matrix
    parent_joint = curr_joint.parent
    if parent_joint != None and curr_joint.is_root_joint == False and curr_joint.name in landmark_rig_mapping and parent_joint.name in landmark_rig_mapping:
        local_anim_rotation_matrix, local_anim_rotation_axis, local_anim_rotation_angle = compute_joint_local_rotation_anim_matrix(
            curr_joint = curr_joint,
            landmark_positions = landmark_positions,
            landmark_rig_mapping = landmark_rig_mapping)
        
        # grandparent's anim matrix 
        grandparent_total_anim_matrix = float4x4()
        grandparent_total_anim_matrix.identity()
        if parent_joint.parent != None:
            grandparent_total_anim_matrix = parent_joint.parent.total_anim_matrix

        # recalculate parent's total anim matrix and update
        parent_joint.anim_matrix = local_anim_rotation_matrix
        parent_joint_total_anim_matrix = float4x4.concat_matrices([
            grandparent_total_anim_matrix,
            parent_joint.local_matrix,
            parent_joint.anim_matrix
        ])
        parent_joint.total_anim_matrix = parent_joint_total_anim_matrix

        local_anim_rotation_axis_angles[parent_joint.name] = [local_anim_rotation_axis, local_anim_rotation_angle]

    # compute the total anim matrix for the joint
    curr_joint.total_anim_matrix = float4x4.concat_matrices([
        parent_joint_total_anim_matrix,
        curr_joint.local_matrix,
        curr_joint.anim_matrix
    ])

    '''
    color = float3(0.0, 0.0, 0.0)
    color.x = 1.0
    color.y = 0.0
    color.z = 1.0
    print('draw_sphere([{}, {}, {}], 0.02, {}, {}, {}, 255) # {}'.format(
            curr_joint.total_anim_matrix.entries[3],
            curr_joint.total_anim_matrix.entries[7],
            curr_joint.total_anim_matrix.entries[11],
            color.x * 255.0,
            color.y * 255.0,
            color.z * 255.0,
            curr_joint.name))
    '''

    # traverse into children
    for child_joint in curr_joint.children:
        traverse_for_anim_matrix(
            curr_joint = child_joint, 
            landmark_positions = landmark_positions,
            landmark_rig_mapping = landmark_rig_mapping,
            local_anim_rotation_axis_angles = local_anim_rotation_axis_angles)

##### compute_joint_local_rotation_matrices2 #####

##
def compute_joint_local_rotation_matrices2(
    rig, 
    landmark_positions,
    landmark_rig_mapping):

    local_anim_rotation_axis_angles = {}
    for root_joint in rig.root_joints:
        traverse_for_anim_matrix(
            root_joint, 
            landmark_positions, 
            landmark_rig_mapping,
            local_anim_rotation_axis_angles)

    local_anim_matrix_info = {}
    for joint in rig.joints:
        local_anim_matrix_info[joint.name] = joint.anim_matrix

    return local_anim_matrix_info, local_anim_rotation_axis_angles

##### compute_joint_local_rotation_matrices2 ######


##
def test_rig3(landmark_positions):
    rig = load_rig('c:\\Users\\dingwings\\demo-models\\media-pipe\\test-rig-1.gltf')

    landmark_rig_mapping = {}
    landmark_rig_mapping['left_hand'] = 20
    landmark_rig_mapping['left_lower_arm'] = 16
    landmark_rig_mapping['left_upper_arm'] = 14
    landmark_rig_mapping['left_shoulder'] = 12

    landmark_rig_mapping['right_hand'] = 19
    landmark_rig_mapping['right_lower_arm'] = 15
    landmark_rig_mapping['right_upper_arm'] = 13
    landmark_rig_mapping['right_shoulder'] = 11

    landmark_rig_mapping['left_thigh'] = 24
    landmark_rig_mapping['left_leg'] = 26
    landmark_rig_mapping['left_ankle'] = 28
    landmark_rig_mapping['left_feet'] = 30

    landmark_rig_mapping['right_thigh'] = 23
    landmark_rig_mapping['right_leg'] = 25
    landmark_rig_mapping['right_ankle'] = 27
    landmark_rig_mapping['right_feet'] = 29

    return rig, compute_joint_local_rotation_matrices(
        rig = rig,
        landmark_positions = landmark_positions,
        landmark_rig_mapping = landmark_rig_mapping)

##
def test_rig4(rig, landmark_positions):
    
    landmark_rig_mapping = {}
    landmark_rig_mapping['left_hand'] = [16]
    landmark_rig_mapping['left_arm'] = [14]
    landmark_rig_mapping['left_shoulder'] = [12]

    landmark_rig_mapping['right_hand'] = [15]
    landmark_rig_mapping['right_arm'] = [13]
    landmark_rig_mapping['right_shoulder'] = [11]

    landmark_rig_mapping['left_thigh'] = [24]
    landmark_rig_mapping['left_leg'] = [26]
    landmark_rig_mapping['left_ankle'] = [28]
    landmark_rig_mapping['left_feet'] = [30]

    landmark_rig_mapping['right_thigh'] = [23]
    landmark_rig_mapping['right_leg'] = [25]
    landmark_rig_mapping['right_ankle'] = [27]
    landmark_rig_mapping['right_feet'] = [29]

    landmark_rig_mapping['pelvis'] = [23, 24]
    landmark_rig_mapping['left_clavicle'] = [23, 24, 11, 12]
    landmark_rig_mapping['neck'] = [9, 10, 11, 12]
    landmark_rig_mapping['head'] = [9, 10]

    local_anim_matrix_info, local_anim_rotation_axis_angles = compute_joint_local_rotation_matrices2(
        rig = rig,
        landmark_positions = landmark_positions,
        landmark_rig_mapping = landmark_rig_mapping)

    return local_anim_matrix_info, local_anim_rotation_axis_angles

##
def traverse_total_animation_joint(curr_joint):

    parent_total_anim_matrix = float4x4()
    parent_total_anim_matrix.identity()
    if curr_joint.parent != None:
        parent_total_anim_matrix = curr_joint.parent.total_anim_matrix

    curr_joint.total_anim_matrix = float4x4.concat_matrices([
        parent_total_anim_matrix,
        curr_joint.local_matrix,
        curr_joint.anim_matrix
    ])

    for child in curr_joint.children:
        traverse_total_animation_joint(child)

##
def output_debug_rig(
    rig):
    for root_joint in rig.root_joints:
        traverse_total_animation_joint(
            root_joint)

    color = float3(1.0, 1.0, 0.0)
    for joint in rig.joints:
        joint_position = float3(
            joint.total_anim_matrix.entries[3],
            joint.total_anim_matrix.entries[7],
            joint.total_anim_matrix.entries[11])
        
        if joint.name.find('shoulder') >= 0:
            color = float3(1.0, 0.0, 0.0)
        elif joint.name.find('arm') >= 0:
            color = float3(0.0, 1.0, 0.0)
        elif joint.name.find('hand') >= 0:
            color = float3(0.0, 0.0, 1.0)
        elif joint.name.find('thigh') >= 0:
            color = float3(1.0, 1.0, 0.0)
        elif joint.name.find('leg') >= 0:
            color = float3(1.0, 0.0, 1.0)
        elif joint.name.find('foot') >= 0:
            color = float3(0.0, 1.0, 1.0)
        else:
            color = float3(0.0, 0.0, 0.0)

        print('draw_sphere([{}, {}, {}], 0.02, {}, {}, {}, 255) # {}'.format(
            joint_position.x,
            joint_position.y,
            joint_position.z,
            color.x * 255.0,
            color.y * 255.0,
            color.z * 255.0,
            joint.name))


##
def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode = True,
        min_detection_confidence = 0.5, 
        min_tracking_confidence = 0.5)

    cap = cv.VideoCapture('d:\\test\\mediapipe\\4.mp4')

    # load rig
    rig = load_rig('c:\\Users\\dingwings\\demo-models\\media-pipe\\test-rig-5.gltf')

    frame_index = 0
    while cap.isOpened():

        # read frame of movie
        ret, frame = cap.read()
        #image_height, image_width, _ = frame.shape
        #plt.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        
        # process for landmarks
        results = pose.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

        # img = cv.imread('d:\\test\\openpose\\test-image-0.jpg')
        # image_height, image_width, _ = img.shape
        # plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        #results = pose.process(cv.cvtColor(img, cv.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            print('!!! no valid landmarks at frame: {} !!!'.format(frame_index))
            continue

        landmark_positions = []
        for i in range(0, len(results.pose_world_landmarks.landmark)):
            color = [0, 0, 0]

            if i == 23 or i == 24:
                color = [255, 0, 0]
            elif i == 11 or i == 12:
                color = [0, 255, 0]
            elif i == 13 or i == 14:
                color = [0, 0, 255]
            elif i == 15 or i == 16:
                color = [255, 255, 255]

            scaled_landmark = float3(results.pose_world_landmarks.landmark[i].x, results.pose_world_landmarks.landmark[i].y * -1.0, results.pose_world_landmarks.landmark[i].z) 
            landmark_positions.append(scaled_landmark)

            '''
            print('draw_sphere([{}, {}, {}], 0.02, {}, {}, {}, 255) # {}'.format(
                    scaled_landmark.x,
                    scaled_landmark.y,
                    scaled_landmark.z,
                    color[0],
                    color[1],
                    color[2],
                    i))
            '''

        '''
        v0 = float3.normalize(float3(103.0, 33.0, -823.0))
        v1 = float3.normalize(float3(-8.0, -2330.2, -14.0))
        axis = float3.normalize(float3.cross(v1, v0))
        angle = math.acos(float3.dot(v0, v1))
        xform_mat = float4x4.from_angle_axis(axis, angle)
        length0 = float3.length(v0)
        length1 = float3.length(v1)
        check_v = float3.normalize(xform_mat.apply(v1))

        print('draw_sphere([{}, {}, {}], 0.02, 255.0, 0.0, 0.0, 255)'.format(v0.x, v0.y, v0.z))
        print('draw_sphere([{}, {}, {}], 0.02, 0.0, 255.0, 0.0, 255)'.format(v1.x, v1.y, v1.z))
        print('draw_sphere([{}, {}, {}], 0.02, 255.0, 255.0, 0.0, 255)'.format(check_v.x, check_v.y, check_v.z))
        '''

        joint_local_rotation_matrices, joint_anim_local_rotation_axis_angles = test_rig4(rig, landmark_positions)
        #output_debug_rig(rig)

        '''
        # debug script to rotate the joints in blender 3d
        print('obj = bpy.data.objects[\'root\']')
        for key in joint_anim_local_rotation_axis_angles:
            axis_angle = joint_anim_local_rotation_axis_angles[key]
            if axis_angle[1] > 0.0:
                print('bone = obj.pose.bones[\'{}\']'.format(key))
                print('bone.rotation_mode = \'AXIS_ANGLE\'')
                print('bone.rotation_axis_angle = [{}, {}, {}, {}]'.format(
                    axis_angle[1],
                    axis_angle[0].x,
                    axis_angle[0].y,
                    axis_angle[0].z
                ))
        '''

        #plt.imshow(cv.cvtColor(annotated_image, cv.COLOR_BGR2RGB))
        
        # draw annoted image
        index = 0
        annotated_image = frame.copy()
        bg_image = np.zeros(frame.shape, dtype = np.uint8)
        annotated_image = np.where(True, annotated_image, bg_image)
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        #cv.imwrite('d:\\test\\mediapipe\\annotated_image' + str(index) + '.png', annotated_image)
        
        cv.imshow('frame', annotated_image)

        frame_index += 1
        key = cv.waitKey(1)


    cap.release()
    cv.destroyAllWindows()

##
if __name__ == '__main__':
    main()

