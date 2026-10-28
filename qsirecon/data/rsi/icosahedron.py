# Based roughly on the C++ code at
# http://blog.andreaskahler.com/2009/06/creating-icosphere-mesh-in-code.html

import math
import numpy as np

class Point3D( object ):
  def __init__( self, x, y, z ):
    self.x, self.y, self.z = x, y, z

  def distFrom( self, point ):
    return math.sqrt( (self.x-point.x)**2
                    + (self.y-point.y)**2
                    + (self.z-point.z)**2 )

  def normalize( self ):
    length = math.sqrt( (self.x)**2
                      + (self.y)**2
                      + (self.z)**2 )
    return Point3D(self.x/length, self.y/length, self.z/length)

  def average( self, point ):
    return Point3D((self.x + point.x)/2,
                   (self.y + point.y)/2,
                   (self.z + point.z)/2)

  def exists( self, points, tolerence = 1.0e-6 ):
    minIndex = -1
    minDistance = 1.0
    for i in range(len(points)):
      if self.distFrom(points[i]) < tolerence:
        minIndex = i
    return minIndex


def make_icosahedron(order=3):
  vertex = []
  face = []

  t = (1.0 + math.sqrt(5.0)) / 2.0;

  vertex.append(Point3D(-1,  t,  0).normalize())
  vertex.append(Point3D( 1,  t,  0).normalize())
  vertex.append(Point3D(-1, -t,  0).normalize())
  vertex.append(Point3D( 1, -t,  0).normalize())

  vertex.append(Point3D( 0, -1,  t).normalize())
  vertex.append(Point3D( 0,  1,  t).normalize())
  vertex.append(Point3D( 0, -1, -t).normalize())
  vertex.append(Point3D( 0,  1, -t).normalize())

  vertex.append(Point3D( t,  0, -1).normalize())
  vertex.append(Point3D( t,  0,  1).normalize())
  vertex.append(Point3D(-t,  0, -1).normalize())
  vertex.append(Point3D(-t,  0,  1).normalize())

  #for i in range(len(vertex)):
  #  minDistance = 1.0e6
  #  for j in range(len(vertex)):
  #    if i != j:
  #      distance = vertex[i].distFrom(vertex[j])
  #      if distance < minDistance:
  #        minDistance = distance

  #  print(minDistance)

  # 5 faces around point 0
  face.append([0, 11, 5])
  face.append([0, 5, 1])
  face.append([0, 1, 7])
  face.append([0, 7, 10])
  face.append([0, 10, 11])

  # 5 adjacent faces 
  face.append([1, 5, 9])
  face.append([5, 11, 4])
  face.append([11, 10, 2])
  face.append([10, 7, 6])
  face.append([7, 1, 8])

  # 5 faces around point 3
  face.append([3, 9, 4])
  face.append([3, 4, 2])
  face.append([3, 2, 6])
  face.append([3, 6, 8])
  face.append([3, 8, 9])

  # 5 adjacent faces 
  face.append([4, 9, 5])
  face.append([2, 4, 11])
  face.append([6, 2, 10])
  face.append([8, 6, 7])
  face.append([9, 8, 1])

  for ocount in range(order):
    newface = []

    for f in face:
      vindex0 = f[0]
      vindex1 = f[1]
      vindex2 = f[2]

      v0 = vertex[vindex0]
      v1 = vertex[vindex1]
      v2 = vertex[vindex2]

      v3 = Point3D.average(v0, v1).normalize()
      index = v3.exists(vertex)
      if index >= 0:
        vindex3 = index
      else:
        vertex.append(v3)
        vindex3 = len(vertex)-1

      v4 = Point3D.average(v1, v2).normalize()
      index = v4.exists(vertex)
      if index >= 0:
        vindex4 = index
      else:
        vertex.append(v4)
        vindex4 = len(vertex)-1

      v5 = Point3D.average(v2, v0).normalize()
      index = v5.exists(vertex)
      if index >= 0:
        vindex5 = index
      else:
        vertex.append(v5)
        vindex5 = len(vertex)-1


      newface.append([vindex0, vindex3, vindex5])
      newface.append([vindex3, vindex1, vindex4])
      newface.append([vindex4, vindex2, vindex5])
      newface.append([vindex3, vindex4, vindex5])

    face = newface
    #print(len(face), len(vertex))

    #for i in range(len(vertex)):
    #  minDistance = 1.0e6
    #  for j in range(len(vertex)):
    #    if i != j:
    #      distance = vertex[i].distFrom(vertex[j])
    #      if distance < minDistance:
    #        minDistance = distance

    #  print(minDistance)

  npvertex = np.zeros(shape=(len(vertex),3))
  for i in range(len(vertex)):
    npvertex[i,:] = [vertex[i].x, vertex[i].y, vertex[i].z]

  return npvertex
