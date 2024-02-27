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
    def to_angle_axis(m):
        epsilon = 0.01
        epsilon2 = 0.1
        
        if ((math.fabs(m.entries[0]-m.entries[4])< epsilon)
        and (math.fabs(m.entries[2]-m.entries[8])< epsilon)
        and (math.fabs(m.entries[6]-m.entries[9])< epsilon)): 
            
            if ((math.abs(m.entries[1]+m.entries[4]) < epsilon2)
            and (math.abs(m.entries[2]+m.entries[8]) < epsilon2)
            and (math.abs(m.entries[6]+m.entries[9]) < epsilon2)
            and (math.abs(m.entries[0]+m.entries[5]+m.entries[10]-3) < epsilon2)):
                return [0,1,0,0]
            
            angle = math.PI
            xx = (m.entries[0]+1)/2
            yy = (m.entries[5]+1)/2
            zz = (m.entries[10]+1)/2
            xy = (m.entries[1]+m.entries[4])/4
            xz = (m.entries[2]+m.entries[8])/4
            yz = (m.entries[6]+m.entries[9])/4
            if ((xx > yy) and (xx > zz)):
                if (xx < epsilon):
                    x = 0
                    y = 0.7071
                    z = 0.7071
                else:
                    x = math.sqrt(xx)
                    y = xy/x
                    z = xz/x
                
            elif (yy > zz):
                if (yy< epsilon):
                    x = 0.7071
                    y = 0
                    z = 0.7071
                else:
                    y = math.sqrt(yy)
                    x = xy/y
                    z = yz/y
            else:
                if (zz< epsilon):
                    x = 0.7071
                    y = 0.7071
                    z = 0
                else:
                    z = math.sqrt(zz)
                    x = xz/z
                    y = yz/z

            return [angle, x, y, z]
        
        s = math.sqrt((m.entries[9] - m.entries[6])*(m.entries[9] - m.entries[6])
            +(m.entries[2] - m.entries[8])*(m.entries[2] - m.entries[8])
            +(m.entries[4] - m.entries[1])*(m.entries[4] - m.entries[1]))
        if (math.fabs(s) < 0.001):
             s=1
            
        angle = math.acos(( m.entries[0] + m.entries[5] + m.entries[10] - 1)/2)
        x = (m.entries[9] - m.entries[6])/s
        y = (m.entries[2] - m.entries[8])/s
        z = (m.entries[4] - m.entries[1])/s

        return [angle, x, y, z]

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
    if (parent_joint != None and 
       curr_joint.is_root_joint == False and 
       curr_joint.name in landmark_rig_mapping and 
       parent_joint.name in landmark_rig_mapping and
       parent_joint.name != 'pelvis'):

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

    # special case for rotating pelvis
    if curr_joint.name == 'pelvis':
        # rotation in y axis
        rotation_axis_y = float3.normalize(landmark_positions[23] - landmark_positions[24])
        plane_v = float3(1.0, 0.0, 0.0)
        angle_x = math.acos(float3.dot(rotation_axis_y, plane_v))
        rotation_matrix_y = float4x4.from_angle_axis(float3(0.0, 1.0, 0.0), angle_x)

        # rotation in x axis
        rotation_axis_x = float3.normalize(landmark_positions[12] - landmark_positions[24])
        plane_v = float3(0.0, 1.0, 0.0)
        angle_y = math.cos(float3.dot(rotation_axis_x, plane_v))
        rotation_matrix_x = float4x4.from_angle_axis(float3(1.0, 0.0, 0.0), angle_y)

        curr_joint.anim_matrix = float4x4.concat_matrices([
            rotation_matrix_y,
            rotation_matrix_x
        ])

        axis_angle_array = float4x4.to_angle_axis(curr_joint.anim_matrix)
        local_anim_rotation_axis_angles[curr_joint.name] = [float3(axis_angle_array[0], axis_angle_array[1], axis_angle_array[2]), axis_angle_array[3]]

        
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
    landmark_rig_mapping['left_foot'] = [32]

    landmark_rig_mapping['right_thigh'] = [23]
    landmark_rig_mapping['right_leg'] = [25]
    landmark_rig_mapping['right_ankle'] = [27]
    landmark_rig_mapping['right_foot'] = [31]

    landmark_rig_mapping['pelvis'] = [23, 24]
    landmark_rig_mapping['left_pelvis'] = [23, 24]
    landmark_rig_mapping['right_pelvis'] = [23, 24]

    landmark_rig_mapping['right_clavicle'] = [23, 24, 11, 12]
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

    cap = cv.VideoCapture('d:\\test\\mediapipe\\5.mp4')

    # load rig
    rig = load_rig('c:\\Users\\dingwings\\demo-models\\media-pipe\\test-rig-6.gltf')

    # reset key-frame file
    file = open('d:\\test\\mediapipe\\blender-key-frames.py', 'w')
    file.close()

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

        
        # debug script to rotate the joints in blender 3d and set key frame every 5 frames
        if frame_index % 5 == 0:
            print('bpy.ops.object.mode_set(mode=\'POSE\')', file = open('d:\\test\\mediapipe\\blender-key-frames.py', 'a'))
            print('obj = bpy.data.objects[\'root\']', file = open('d:\\test\\mediapipe\\blender-key-frames.py', 'a'))
            for key in joint_anim_local_rotation_axis_angles:
                axis_angle = joint_anim_local_rotation_axis_angles[key]
                
                print('bone = obj.pose.bones[\'{}\']'.format(key), file = open('d:\\test\\mediapipe\\blender-key-frames.py', 'a'))
                print('bone.rotation_mode = \'AXIS_ANGLE\'', file = open('d:\\test\\mediapipe\\blender-key-frames.py', 'a'))
                print('bone.rotation_axis_angle = [{}, {}, {}, {}]'.format(
                    axis_angle[1],
                    axis_angle[0].x,
                    axis_angle[0].y,
                    axis_angle[0].z
                ), file = open('d:\\test\\mediapipe\\blender-key-frames.py', 'a'))
                print('obj.data.bones[\'{}\'].select = True'.format(key), file = open('d:\\test\\mediapipe\\blender-key-frames.py', 'a'))
                print('bone.keyframe_insert(data_path = \'rotation_axis_angle\', frame = {})'.format(frame_index + 1), file = open('d:\\test\\mediapipe\\blender-key-frames.py', 'a'))
                print('obj.data.bones[\'{}\'].select = False'.format(key), file = open('d:\\test\\mediapipe\\blender-key-frames.py', 'a'))

            print('\n\n\n', file = open('d:\\test\\mediapipe\\blender-key-frames.py', 'a'))

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
        
        # back to identity since we're traversing the entire rig every frame for the animation matrices
        for joint in rig.joints:
            joint.total_anim_matrix.identity()
            joint.anim_matrix.identity()

        cv.imshow('frame', annotated_image)

        if frame_index >= 600:
            break

        frame_index += 1
        key = cv.waitKey(1)


    cap.release()
    cv.destroyAllWindows()

##
if __name__ == '__main__':
    main()

