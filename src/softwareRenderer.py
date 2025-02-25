# Just a little exercise that I did get intuition for:
# homogenous coordinates
# basic render pipeline
# perspective projection
# rotation matrices


import math,pygame,time,pygame.gfxdraw
import numpy as np


class vec4():
    x,y,z,w = 0,0,0,0
    def __init__(self,x=0,y=0,z=0,w=0):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def __getitem__(self, key):
        if (key==0):
            return self.x
        if (key==1):
            return self.y
        if (key==2):
            return self.z
        if (key==3):
            return self.w
        return 0

    def __setitem__(self, key, value):
        if (key==0):
            self.x=value
        if (key==1):
            self.y=value
        if (key==2):
            self.z=value
        if (key==3):
            self.w=value
            
    def __str__(self):
        return f"({self.x:.2f},{self.y:.2f},{self.z:.2f},{self.w:.2f})"

    def __mul__(self,o):    
        if (type(o) == float or type(o) == int):
            return vec4(self.x*o,self.y*o,self.z*o,self.w*o)
        
        if (type(o) == vec4):
            return (self.x*o.x) + (self.y*o.y) + (self.z*o.z) + (self.w*o.w)
        
    def __add__(self,o):
        if (type(o) == vec4):
            return vec4(
                self.x+o.x,
                self.y+o.y,
                self.z+o.z,
                self.w+o.w)
        
    def __sub__(self,o):
        if (type(o) == vec4):
            return vec4(self.x-o.x,self.y-o.y,self.z-o.z,self.w-o.w)
        
    def cross3D(self,o):
        return vec4(
            self.y*o.z - self.z*o.y,
            self.z*o.x - self.x*o.z,
            self.x*o.y - self.y*o.x,
            0)
        
    def mag(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2 + self.w**2)
    def norm(self):
        return self*(1/self.mag())
    


class mat4x4():
    mat = []
    def __init__(self,mat = [vec4(),vec4(),vec4(),vec4()]):
        self.mat = mat
        
    def __getitem__(self, key):
        return self.mat[key]

    def transpose3x3(self):
        return mat4x4([
            vec4(self.mat[0][0],self.mat[1][0],self.mat[2][0],self.mat[0][3]),
            vec4(self.mat[0][1],self.mat[1][1],self.mat[2][1],self.mat[1][3]),
            vec4(self.mat[0][2],self.mat[1][2],self.mat[2][2],self.mat[2][3]),
            self.mat[3]
        ])
            
    def __mul__(self,o):
        #matrix matrix
        if (type(o) == mat4x4):
            outMat = mat4x4([vec4(0,0,0,0),vec4(0,0,0,0),vec4(0,0,0,0),vec4(0,0,0,0)])
            for i in range(4):
                for j in range(4):
                    s = 0
                    for k in range(4):
                         s += o[i][k]*self[k][j]
                    outMat[i][j] = s
            
            return outMat
        #matrix vector
        if (type(o) == vec4):
            outVec = vec4(0,0,0,0)
            for i in range(4):
                for k in range(4):
                    outVec[i] += self[i][k]*o[k]
            return outVec
                  
    def __str__(self):
        string = ""
        for i in range(4):
            string += str(self[i])+"\n"
        return string

    #Proper Euler ZYX
    def rotation(x,y,z):
        rx = mat4x4([
            vec4(1,0,0,0),
            vec4(0,math.cos(x),-math.sin(x),0),
            vec4(0,math.sin(x),math.cos(x),0),
            vec4(0,0,0,1)])

        ry = mat4x4([
            vec4(math.cos(y),0,math.sin(y),0),
            vec4(0,1,0,0),
            vec4(-math.sin(y),0,math.cos(y),0),
            vec4(0,0,0,1)])
        
        rz = mat4x4([
            vec4(math.cos(z),-math.sin(z),0,0),
            vec4(math.sin(z),math.cos(z),0,0),
            vec4(0,0,1,0),
            vec4(0,0,0,1)])

        return rz*ry*rx

    def translation(x,y,z):
        return mat4x4([
        vec4(1,0,0,x),
        vec4(0,1,0,y),
        vec4(0,0,1,z),
        vec4(0,0,0,1)])

    def transform(pitch,yaw,roll,x,y,z):
        T = mat4x4.translation(-x,-y,-z)
        R = mat4x4.rotation(pitch,yaw,roll).transpose3x3()
        return (R*T)

    def invtransform(pitch,yaw,roll,x,y,z):
        T = mat4x4.translation(x,y,z)
        R = mat4x4.rotation(pitch,yaw,roll)
        return (T*R)
    
    def perspective(fov,aspect,near,far):
        scale = 1/math.tan(fov/2)
        clip = -(far+near)/(far-near)
        clip2 = -(2*far*near)/(far-near)
        return mat4x4(
            [vec4(aspect*scale,0,0,0),
            vec4(0,scale,0,0),
            vec4(0,0,clip,clip2),
            vec4(0,0,-1,0)]
        )

def clamp(a,b,c):
    return min(max(a,b),c)
        
def degToRad(x):
    return x * math.pi/180



SIZE = (800,800)
pygame.font.init() 
obj_file = "../models/bunny.obj"
my_font = pygame.font.SysFont('Helvetica', 12)
screen = pygame.display.set_mode(SIZE)
run = True

#camera position
pos = vec4(0,0,0,0)
#pitch/yaw
p = 0
y = 0
#time
t = 0

#fps related
fps = 0
last = time.time()

#specifies weather to center and hide mouse
lockMouse = False
#shader display mode
frag = 0

#render related
drawNormal = False
backfaceCulling = True
lightDirection = vec4(1,1,1,0).norm()
lightColor = (255,255,255)
globalIllumination = (20,20,20)

#geometry related
verts = []
edges = []
faces = []
faceSizes = []
norm = []


with open(obj_file,"r") as f:
    for line in f.readlines():
        if (line.startswith("v")):
            v = line.strip("\n").split(" ")
            verts.append(vec4(float(v[1]),float(v[2]),float(v[3]),1))
        if (line.startswith("f")):
            f = line.strip("\n").split(" ")
            f = [int(f[1])-1,int(f[2])-1,int(f[3])-1]
            edges.append([f[0],f[1]])
            edges.append([f[1],f[2]])
            edges.append([f[2],f[1]])
            faces.append(f)

for f in faces:
    a=verts[f[0]]
    b=verts[f[1]]
    c=verts[f[2]]
    E1 = (b-a)
    E2 = (c-b)
    faceSizes.append(0.3 * max(max(E1.mag(),E2.mag()),(a-c).mag()))
    normal = (E1.cross3D(E2)).norm()

    norm.append(normal)

while run:


    delta = max(time.time()-last,0.001)
    pygame.display.set_caption(f"Software Renderer ({int(1/(delta) ):}fps)")
    last = time.time()


    ObjR = mat4x4.rotation(math.sin(t)*0.1,degToRad(t*10),math.sin(t)*0.1)
    ObjT = mat4x4.translation(0,math.sin(t*0.5)*0.1,0)
    ObjMat = ObjR * ObjT#mat4x4.transform(math.sin(t)*0.1,degToRad(t*10),math.sin(t)*0.1,0,math.sin(t*0.5)*0.1,0)
    t+=delta

    far = 100
    P = mat4x4.perspective(degToRad(60),1,0.01,far)

    SceneMat = mat4x4.invtransform(degToRad(p),degToRad(y),0,pos.x,pos.y,pos.z)
    camRotation = mat4x4.rotation(degToRad(p),degToRad(y),0).transpose3x3()
   
    
    #Input handling
    keys = pygame.key.get_pressed()

    if (keys[pygame.K_UP]):
        p -= 40*delta
    if (keys[pygame.K_DOWN]):
        p += 40*delta

    #if (keys[pygame.K_LEFT]):
    #    y -= 40*delta
    #if (keys[pygame.K_RIGHT]):
    #    y += 40*delta

    if(lockMouse):
        pygame.mouse.set_pos((400,400))
        rel = pygame.mouse.get_rel()
        p += (rel[1]/800) * delta * 360
        y += (rel[0]/800) * delta * 360

    if (pygame.mouse.get_pressed()[0]):
        if (not lockMouse):
            lockMouse = True
            pygame.mouse.set_visible(False)
            pygame.event.set_grab(True)
    

    forward = (camRotation*vec4(0,0,1,0)).norm()
    view = forward
    right = (camRotation*vec4(-1,0,0,0)).norm()

    if (keys[pygame.K_w]):
        pos+=forward*delta*2
    if (keys[pygame.K_s]):
        pos-=forward*delta*2
    if (keys[pygame.K_a]):
        pos-=right*delta*2
    if (keys[pygame.K_d]):
        pos+=right*delta*2

    for event in pygame.event.get():
        if (event.type == pygame.QUIT):
            run = False
        if (event.type == pygame.KEYDOWN):
            if event.key == pygame.K_r:
                #debug statements go here!
                pass
            if event.key == pygame.K_n:
                drawNormal = not drawNormal
            if event.key == pygame.K_ESCAPE:
                lockMouse = False
                pygame.mouse.set_visible(True)
                pygame.event.set_grab(False)
            if event.key == pygame.K_f:
                frag+=1
                if (frag>2):
                    frag = 0

    #Render 
    
    screen.fill((0,0,0))
    
    def toScr(pt):
        pt = pt
        if (pt[3] != 0):
            pt[0] /= pt[3]
            pt[1] /= pt[3]
            pt[2] /= far
            pt[3] /= pt[3]
        #HDC to px
        return vec4(400+(pt[0]*400),400+(-pt[1]*400),pt[2],0)




    cameraSpace = []
    profile = time.time()
    mat = (ObjMat*SceneMat)*P

    #local -> world -> camera -> screen

    camMat = np.array([
        [mat[0][0],mat[0][1],mat[0][2],mat[0][3]],
        [mat[1][0],mat[1][1],mat[1][2],mat[1][3]],
        [mat[2][0],mat[2][1],mat[2][2],mat[2][3]],
        [mat[3][0],mat[3][1],mat[3][2],mat[3][3]],
    ])
        
    for point in verts:
        #my mat4x4:
        #pt = toScr(mat*vec4(point.x,point.y,point.z,1))
        #np mat4x4 (row major value arrays and c bindings are faster obv): 
        pt = np.matmul(camMat,np.array([point.x,point.y,point.z,1]))
        pt = toScr(pt)
        cameraSpace.append(pt)

    #profile matrix multiplication time
    profile = time.time() - profile
    surf_MMText = my_font.render(f"MatMul {profile*1000:0.2f}ms", False, (255, 255, 255))
    profile = time.time()

    #painter's algorithm
    poly = []
    for face in faces:
        #[index of polygon, minimum depth of all points]
        poly.append((len(poly), min(min(cameraSpace[face[0]].z,cameraSpace[face[1]].z),cameraSpace[face[2]].z) ))

    #sort by depth
    poly.sort(key = lambda x:-x[1])

    def validate(pt):
        pt.x = int(pt.x)
        pt.y = int(pt.y)
        return pt.z>0 and abs(pt.x)<3000 and abs(pt.y) < 3000

    #commented out bc it's performance heavy
    #invObj = np.array([
    #    [ObjMat[0][0],ObjMat[0][1],ObjMat[0][2],ObjMat[0][3]],
    #    [ObjMat[1][0],ObjMat[1][1],ObjMat[1][2],ObjMat[1][3]],
    #    [ObjMat[2][0],ObjMat[2][1],ObjMat[2][2],ObjMat[2][3]],
    #    [ObjMat[3][0],ObjMat[3][1],ObjMat[3][2],ObjMat[3][3]],
    #])
    #invObj = np.linalg.inv(invObj)
    #localPos = np.matmul(invObj,np.array([pos.x,pos.y,pos.z,1]))
    #localPos = vec4(localPos[0],localPos[1],localPos[2],1)
    
    for fIndex in poly:
        poly = faces[fIndex[0]]
        faceSize = faceSizes[fIndex[0]]
        
        A = cameraSpace[poly[0]]
        B = cameraSpace[poly[1]]
        C = cameraSpace[poly[2]]        
        
        localNormal = norm[fIndex[0]]
        worldNormal = ObjR*localNormal
        barycenter = (verts[poly[0]]+verts[poly[1]]+verts[poly[2]])*(1.0/3.0)

        #Didn't get this to work, I think it's because my coordinate spaces are all goofy
        #lineOfSight = (barycenter - localPos).norm()
        #dot = lineOfSight*localNormal

        dot = view*worldNormal
        if (drawNormal):
            #using my mat4x4
            #D = (verts[poly[0]]+verts[poly[1]]+verts[poly[2]])*(1.0/3.0)
            #E = toScr(mat*(D+surfNorm*0.2))
            #D = toScr(mat*D)
            
            #using np mat4x4
            E = toScr(np.matmul(camMat,np.array([barycenter.x + localNormal.x*faceSize,barycenter.y + localNormal.y*faceSize ,barycenter.z + localNormal.z*faceSize,1])))
            D = toScr(np.matmul(camMat,np.array([barycenter.x,barycenter.y,barycenter.z,1])))

        #determine triangle color
        color = (0,0,0)
        if (frag==0):
            # world space normal
            color = (128 + 127*worldNormal.x,128 + 127*worldNormal.y,128 + 127*worldNormal.z)
        elif (frag == 1):
            #view angle dot
            color = (dot*255,(1-dot)*255,0)
        elif (frag == 2):
            lDot = clamp(worldNormal*lightDirection,0,1)
            color = (
                     clamp(globalIllumination[0]+lDot*lightColor[0],0,255),
                     clamp(globalIllumination[1]+lDot*lightColor[1],0,255),
                     clamp(globalIllumination[2]+lDot*lightColor[2],0,255))

        #draw triangles and normals
        #validate is necessary to cast x/y to ints and ensure they are not exceedingly large or small
        if (validate(A) and validate(B) and validate(C) and (dot >= 0 or not backfaceCulling)):
            pygame.gfxdraw.filled_trigon(screen,A.x,A.y,B.x,B.y,C.x,C.y,color)
            #pygame.gfxdraw.trigon(screen,A.x,A.y,B.x,B.y,C.x,C.y,(0,0,0))
        if (drawNormal and validate(D) and validate(E)):
            pygame.gfxdraw.line(screen,D.x,D.y,E.x,E.y,(0,0,255))


    #profile drawing time
    profile = time.time() - profile
    surf_DText = my_font.render(f"Draw {profile*1000:0.2f}ms", False, (255, 255, 255))

    #display matrix multiplication step execution time
    screen.blit(surf_MMText,(30,45))
    #display drawing execution time
    screen.blit(surf_DText,(30,60))

    #display position coordinates
    text_surface = my_font.render(f"({pos.x:0.2f},{pos.y:0.2f},{pos.z:0.2f})", False, (255, 255, 255))
    screen.blit(text_surface,(30,80))
    #display view direction
    text_surface = my_font.render(f"({view.x:0.2f},{view.y:0.2f},{view.z:0.2f})", False, (255, 255, 255))
    screen.blit(text_surface,(30,95))
    #display instructions
    instructions = "CONTROLS\n\nmouse:  view direction\nw/s:  forward/back \na/d:  left/right\nesc:  unlock cursor\nleft click:  lock cursor\nf:  switch shading\nn:  display normals"
    line = 0
    for s in instructions.split("\n"):
        text_surface = my_font.render(s, False, (255, 255, 255))
        line += 15
        screen.blit(text_surface,(SIZE[0]-130,10 + line))
    
            
    pygame.display.update()
    pygame.display.flip()

pygame.quit()


        
